import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from model import get_vit_tiny_model
from utils import get_dataloaders
import argparse

def evaluate_model(model_path='vit_tiny_model.pth', batch_size=8):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model
    checkpoint = torch.load(model_path)
    num_classes = checkpoint['num_classes']
    
    # Initialize model and load weights
    model = get_vit_tiny_model(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    # Get test dataloader
    _, test_loader = get_dataloaders(batch_size=batch_size)
    
    # Initialize metrics
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Correct predictions: {correct}/{total}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a Vision Transformer model')
    parser.add_argument('--model_path', type=str, default='vit_tiny_model.pth',
                      help='Path to the model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    
    args = parser.parse_args()
    evaluate_model(model_path=args.model_path, batch_size=args.batch_size) 