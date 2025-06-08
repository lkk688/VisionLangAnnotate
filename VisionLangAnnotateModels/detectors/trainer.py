import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import time
from typing import Dict, Any, Optional, Tuple
import logging
from .base_model import BaseModel
from .evaluation import ModelEvaluator

class ModelTrainer:
    """Handles training functionality for multi-modal detection models."""
    
    def __init__(self, base_model: BaseModel, evaluator: ModelEvaluator):
        self.base_model = base_model
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
        
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None,
              num_epochs: int = 10, lr: float = 1e-4, weight_decay: float = 1e-4,
              save_dir: str = "./checkpoints", save_every: int = 5,
              validate_every: int = 1, use_amp: bool = True,
              scheduler_type: str = "cosine", warmup_epochs: int = 1,
              gradient_clip_norm: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Train the model with the given parameters."""
        
        # Setup training components
        optimizer = self._setup_optimizer(lr, weight_decay)
        scheduler = self._setup_scheduler(optimizer, num_epochs, scheduler_type, warmup_epochs)
        scaler = GradScaler() if use_amp else None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics tracking
        train_losses = []
        val_metrics = []
        best_val_loss = float('inf')
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_one_epoch(
                train_dataloader, optimizer, scheduler, scaler,
                use_amp, gradient_clip_norm, epoch
            )
            train_losses.append(train_loss)
            
            # Validation phase
            if val_dataloader and epoch % validate_every == 0:
                val_loss = self._validate_epoch(val_dataloader, use_amp)
                val_metrics.append({'epoch': epoch, 'val_loss': val_loss})
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(save_dir, epoch, optimizer, scheduler, 
                                        train_loss, val_loss, is_best=True)
                
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            # Regular checkpoint saving
            if epoch % save_every == 0:
                self._save_checkpoint(save_dir, epoch, optimizer, scheduler, train_loss)
        
        # Final checkpoint
        self._save_checkpoint(save_dir, num_epochs-1, optimizer, scheduler, train_losses[-1])
        
        return {
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss
        }
    
    def _train_one_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                        scheduler: Any, scaler: Optional[GradScaler], use_amp: bool,
                        gradient_clip_norm: float, epoch: int) -> float:
        """Train for one epoch."""
        
        self.base_model.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with optional mixed precision
            if use_amp and scaler:
                with autocast():
                    loss = self._compute_loss(batch)
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.base_model.model.parameters(), gradient_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self._compute_loss(batch)
                loss.backward()
                
                # Gradient clipping
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.base_model.model.parameters(), gradient_clip_norm)
                
                optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        return total_loss / num_batches
    
    def _validate_epoch(self, dataloader: DataLoader, use_amp: bool) -> float:
        """Validate for one epoch."""
        
        self.base_model.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                if use_amp:
                    with autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a batch based on model type."""
        
        if self.base_model.model_type in ['detr', 'rtdetr']:
            # DETR-style models expect specific input format
            images = batch['images']
            targets = batch['targets']
            
            outputs = self.base_model.model(images)
            
            # Compute DETR loss
            loss_dict = self.base_model.model.criterion(outputs, targets)
            losses = sum(loss_dict[k] * self.base_model.model.weight_dict.get(k, 1) 
                        for k in loss_dict.keys() if k in self.base_model.model.weight_dict)
            
            return losses
            
        elif self.base_model.model_type in ['yolo', 'yolov8', 'yolov9', 'yolov10']:
            # YOLO-style models
            images = batch['images']
            targets = batch['targets']
            
            # YOLO models typically handle loss computation internally
            loss = self.base_model.model(images, targets)
            
            if isinstance(loss, dict):
                return sum(loss.values())
            return loss
            
        else:
            raise ValueError(f"Unsupported model type: {self.base_model.model_type}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the appropriate device."""
        
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.base_model.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.base_model.device) for v in value]
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _setup_optimizer(self, lr: float, weight_decay: float) -> optim.Optimizer:
        """Setup optimizer based on model type."""
        
        if self.base_model.model_type in ['detr', 'rtdetr']:
            # DETR models often use AdamW with different learning rates for backbone and transformer
            param_groups = [
                {'params': [p for n, p in self.base_model.model.named_parameters() 
                           if 'backbone' not in n and p.requires_grad], 'lr': lr},
                {'params': [p for n, p in self.base_model.model.named_parameters() 
                           if 'backbone' in n and p.requires_grad], 'lr': lr * 0.1}
            ]
            return optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            # Default optimizer for other models
            return optim.AdamW(self.base_model.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, num_epochs: int,
                        scheduler_type: str, warmup_epochs: int) -> Optional[Any]:
        """Setup learning rate scheduler."""
        
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
        elif scheduler_type == "warmup_cosine":
            # Simple warmup + cosine implementation
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    return 0.5 * (1 + torch.cos(torch.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            return None
    
    def _save_checkpoint(self, save_dir: str, epoch: int, optimizer: optim.Optimizer,
                        scheduler: Any, train_loss: float, val_loss: float = None,
                        is_best: bool = False) -> None:
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.base_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'model_type': self.base_model.model_type,
            'model_name': self.base_model.model_name
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        self.logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True,
                       load_scheduler: bool = True) -> Dict[str, Any]:
        """Load model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.base_model.device)
        
        # Load model state
        self.base_model.model.load_state_dict(checkpoint['model_state_dict'])
        
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'train_loss': checkpoint.get('train_loss', 0),
            'val_loss': checkpoint.get('val_loss', 0)
        }
        
        self.logger.info(f"Loaded checkpoint from epoch {info['epoch']}")
        return info