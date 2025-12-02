#!/usr/bin/env python3
"""
Training script for code generation models.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model
from data import CodeDataLoader, create_toy_dataset, save_dataset
from evaluation import CodeEvaluator
from utils import set_seed, get_device, Config

logger = logging.getLogger(__name__)


class CodeGenerationTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for code generation models."""
    
    def __init__(self, config: Config, model, tokenizer, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.evaluator = CodeEvaluator()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, outputs
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Tokenize prompts and codes
        prompts = batch['prompts']
        codes = batch['codes']
        
        # Create input sequences (prompt + code)
        sequences = [f"{prompt}\n{code}" for prompt, code in zip(prompts, codes)]
        
        # Tokenize
        inputs = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for language modeling)
        labels = inputs['input_ids'].clone()
        
        # Forward pass
        loss, _ = self.forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Tokenize prompts and codes
        prompts = batch['prompts']
        codes = batch['codes']
        
        # Create input sequences
        sequences = [f"{prompt}\n{code}" for prompt, code in zip(prompts, codes)]
        
        # Tokenize
        inputs = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors='pt'
        )
        
        # Create labels
        labels = inputs['input_ids'].clone()
        
        # Forward pass
        loss, _ = self.forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config.training.warmup_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def create_data_loaders(config: Config, tokenizer):
    """Create data loaders for training and validation."""
    # Create toy dataset if data files don't exist
    if not os.path.exists(config.data.train_path):
        logger.info("Creating toy dataset...")
        os.makedirs(os.path.dirname(config.data.train_path), exist_ok=True)
        
        # Create train/val/test splits
        all_data = create_toy_dataset(1000)
        train_data = all_data[:800]
        val_data = all_data[800:900]
        test_data = all_data[900:]
        
        save_dataset(train_data, config.data.train_path)
        save_dataset(val_data, config.data.val_path)
        save_dataset(test_data, config.data.test_path)
    
    # Create data loader
    data_loader = CodeDataLoader(
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.data.max_length
    )
    
    # Create datasets
    train_dataset = data_loader.create_dataset(config.data.train_path)
    val_dataset = data_loader.create_dataset(config.data.val_path)
    
    return train_dataset, val_dataset


def train_model(config_path: str):
    """Train a code generation model."""
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    # Set seed
    set_seed(config.device.seed)
    
    # Get device
    device = get_device()
    
    # Create model
    model = create_model(config.model.type, config.model.name, device)
    tokenizer = model.tokenizer
    
    # Create data loaders
    train_dataset, val_dataset = create_data_loaders(config, tokenizer)
    
    # Create trainer
    trainer_module = CodeGenerationTrainer(
        config=config,
        model=model.model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output.checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config.logging.log_dir,
        name=config.model.type
    )
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        devices=1,
        accelerator='auto',
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches
    )
    
    # Train the model
    trainer.fit(trainer_module)
    
    logger.info("Training completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train code generation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    train_model(args.config)


if __name__ == '__main__':
    main()
