import torch
from torch.utils.data import DataLoader
import config
from src.data.dataset import CodeSummaryDataset, collate_fn
from src.data.tokenizer import CodeTokenizer
from src.models.transformer import CodeSummarizationTransformer
from src.training.trainer import Trainer


def main():
    print("Loading tokenizer...")
    tokenizer = CodeTokenizer()
    tokenizer.load(config.TOKENIZER_PATH)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    print("Loading training data...")
    train_data = torch.load(config.TRAIN_DATA_PATH)
    val_data = torch.load(config.VAL_DATA_PATH)
    
    max_train = getattr(config, 'MAX_TRAIN_SAMPLES', len(train_data['codes']))
    max_val = getattr(config, 'MAX_VAL_SAMPLES', len(val_data['codes']))
    
    train_dataset = CodeSummaryDataset(
        codes=train_data['codes'][:max_train],
        summaries=train_data['summaries'][:max_train],
        tokenizer=tokenizer
    )
    
    val_dataset = CodeSummaryDataset(
        codes=val_data['codes'][:max_val],
        summaries=val_data['summaries'][:max_val],
        tokenizer=tokenizer
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print("Initializing model...")
    model = CodeSummarizationTransformer(vocab_size=vocab_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size
    )
    
    print("\nStarting training...")
    train_losses, val_losses = trainer.train()
    
    print("\nTraining history:")
    for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
        print(f"Epoch {epoch+1}: Train={t_loss:.4f}, Val={v_loss:.4f}")


if __name__ == "__main__":
    main()


