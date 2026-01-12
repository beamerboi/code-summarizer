import math


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
    def _compute_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lrs[0] * (self.current_step / self.warmup_steps)
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


