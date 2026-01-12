"""Stage 1: binary classification model builder (placeholder)."""
import torch.nn as nn
import torchvision.models as models

def build_model(backbone='resnet18', num_classes=2, pretrained=True):
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    raise NotImplementedError('Backbone not implemented')
