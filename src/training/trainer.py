import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import config
from .scheduler import WarmupCosineScheduler


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        mask = target != self.padding_idx
        loss = loss.masked_select(mask).mean()
        
        return loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab_size: int,
        learning_rate: float = config.LEARNING_RATE,
        warmup_steps: int = config.WARMUP_STEPS,
        epochs: int = config.EPOCHS,
        gradient_accumulation_steps: int = config.GRADIENT_ACCUMULATION_STEPS,
        label_smoothing: float = config.LABEL_SMOOTHING,
        checkpoint_dir: Path = config.CHECKPOINTS_DIR,
        device: torch.device = config.DEVICE
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        self.criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            padding_idx=config.PAD_IDX,
            smoothing=label_smoothing
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=config.BETAS,
            weight_decay=0.01
        )
        
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            code_ids = batch['code_ids'].to(self.device)
            summary_ids = batch['summary_ids'].to(self.device)
            
            tgt_input = summary_ids[:, :-1]
            tgt_output = summary_ids[:, 1:]
            
            logits = self.model(code_ids, tgt_input)
            
            loss = self.criterion(logits, tgt_output)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.scheduler.get_lr():.2e}"
            })
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                code_ids = batch['code_ids'].to(self.device)
                summary_ids = batch['summary_ids'].to(self.device)
                
                tgt_input = summary_ids[:, :-1]
                tgt_output = summary_ids[:, 1:]
                
                logits = self.model(code_ids, tgt_input)
                loss = self.criterion(logits, tgt_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        return checkpoint['epoch']
    
    def train(self):
        print(f"Training on {self.device}")
        print(f"Total epochs: {self.epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            print()
        
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses


