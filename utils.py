import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 3e-4
    num_classes: int = 10
    checkpoint_dir: str = 'checkpoints'
    save_checkpoints: bool = True
    save_best_model: bool = True
    save_final_model: bool = True

def get_dataloaders(batch_size=16, num_workers=2, val_split=0.1, seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    # Load training data and split into train and validation
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_size = int(len(train_ds) * val_split)
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    
    # Load test data
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Create dataloaders with consistent shuffling
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader