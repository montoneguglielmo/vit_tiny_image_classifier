import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from model import get_vit_tiny_model
from utils import get_dataloaders, TrainingConfig
import os
from typing import Dict, Tuple, List, Optional

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    device: Optional[torch.device] = None,
    return_history: bool = False
) -> Tuple[Dict[str, List[float]], nn.Module] | float:
    """
    Train a model with the given configuration and return training history and best model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on (defaults to CUDA if available, else CPU)
        return_history: If True, returns history and model. If False, returns only final validation loss.
    
    Returns:
        If return_history is True:
            Tuple containing:
            - Dictionary with training history (train_loss, val_loss per epoch)
            - Best model based on validation loss
        If return_history is False:
            Final validation loss (float)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    
    # Create checkpoints directory if needed
    if config.save_checkpoints or config.save_best_model:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        train_loss = running_loss/len(train_loader)
        history['train_loss'].append(train_loss)
        print(f"Epoch [{epoch+1}/{config.epochs}], Training Loss: {train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss = val_loss/len(val_loader)
        history['val_loss'].append(val_loss)
        print(f"Epoch [{epoch+1}/{config.epochs}], Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if enabled
        if config.save_checkpoints:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_classes': config.num_classes
            }
            torch.save(checkpoint, f'{config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')
        
        # Save best model if enabled
        if config.save_best_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            if config.save_checkpoints:
                torch.save(checkpoint, f'{config.checkpoint_dir}/best_model.pth')
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    print("Training complete!")
    
    # Save final model if enabled
    if config.save_final_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': config.epochs,
            'num_classes': config.num_classes,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }, 'vit_tiny_model.pth')
        print("Final model saved successfully!")
    
    # Load best model if available
    if best_model is not None:
        model.load_state_dict(best_model)
    
    if return_history:
        return history, model
    else:
        return history['val_loss'][-1]

# Example usage
if __name__ == "__main__":
    # Default configuration
    config = TrainingConfig()
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=config.batch_size)
    
    # Get model
    model = get_vit_tiny_model(config.num_classes)
    
    # Train model
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        return_history=True
    )
