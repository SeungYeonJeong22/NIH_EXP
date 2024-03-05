import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class FPN(nn.Module):
    def __init__(self, num_classes=14):
        super(FPN, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        
        # Replace final fully connected layer
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        
        # Define FPN layers
        self.bottom_up_layers = list(self.backbone.features)
        self.top_down_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(1024, 256, kernel_size=1)
        ])
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1)
        ])
        
    def forward(self, x):
        # Bottom-up pathway
        bottom_up_features = self.bottom_up_forward(x)
        
        # Top-down pathway
        top_down_features = self.top_down_forward(bottom_up_features)
        
        # Lateral connections
        lateral_features = [
            lateral_layer(bottom_up_features[i]) + top_down_features[i]
            for i, lateral_layer in enumerate(self.lateral_layers)
        ]
        
        return lateral_features
    
    def bottom_up_forward(self, x):
        bottom_up_features = []
        for layer in self.bottom_up_layers:
            x = layer(x)
            bottom_up_features.append(x)
        return bottom_up_features
    
    def top_down_forward(self, bottom_up_features):
        top_down_features = []
        prev_features = None
        for i, layer in enumerate(self.top_down_layers):
            if i != 0:
                prev_features = top_down_features[-1]
            top_down_features.append(layer(prev_features if prev_features is not None else bottom_up_features[-1 - i]))
        return top_down_features[::-1]


class MLPBolck(nn.Module):
    def __init__(self):
        super(MLPBolck, self).__init__()
        
        self.network = nn.Sequential(nn.BatchNorm2d(),
                                    nn.MaxPool2d(),
                                    nn.Dropout(0.5),
                                    nn.Linear(2048, 1024))
    def forward(self, x):
        return self.network(x)


class ChestXNetFeaturePyramidModel(nn.Module):
    def __init__(self):
        super(ChestXNetFeaturePyramidModel, self).__init__()
        
        
        self.fpn = FPN()
        