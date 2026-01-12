"""Stage 2: multi-label model builder (placeholder)."""
import torch.nn as nn
import torchvision.models as models

def build_model(backbone='resnet50', num_labels=5, pretrained=True):
    if backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_labels)
        return model
    raise NotImplementedError('Backbone not implemented')
