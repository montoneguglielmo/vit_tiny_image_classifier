import timm
import torch.nn as nn

def get_vit_tiny_model(num_classes: int):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model