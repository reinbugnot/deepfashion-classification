import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet50
import timm

class CustomResNet(nn.Module):
    def __init__(self, output_dim, pretrained=True):
        super(CustomResNet, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = resnet50(pretrained=pretrained)
        
        # Freeze all layers except the last three
        for name, param in self.model.named_parameters():
            if not any(layer in name for layer in ['layer4', 'fc']):
                param.requires_grad = False
        
        # Replace the last linear layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_dim)
    
    def forward(self, x):
        return self.model(x)

class CustomSEResNeXt(nn.Module):
    def __init__(self, num_classes=26):
        super(CustomSEResNeXt, self).__init__()

        self.model = timm.create_model('seresnext50_32x4d', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
    
    
class CustomSEResNeXt_v2(nn.Module):
    def __init__(self, num_classes=26, dropout_p=0.25):
        super(CustomSEResNeXt_v2, self).__init__()

        self.model = timm.create_model('seresnext50_32x4d', pretrained=True)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the final bottleneck block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 512)
        self.dropout1 = nn.Dropout(dropout_p)
        self.hidden = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc_pred = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.hidden(self.dropout1(x)))
        x = self.fc_pred(self.dropout2(x))
        return x


