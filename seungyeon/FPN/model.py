import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import gc

class CheXNet(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super(CheXNet, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.densenet(x)
        out = self.classifier(features)
        out = self.sigmoid(out)
        return features, out

# Feature Pyramid Network (FPN) 정의
class FPN(nn.Module):
    def __init__(self, num_classes=15):
        super(FPN, self).__init__()
        self.chexnet = CheXNet(num_classes=num_classes)

        self.conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features, _ = self.chexnet(x)
        
        # FPN top-down pathway
        p5 = self.conv1(features.unsqueeze(-1).unsqueeze(-1))  # Change to (N, C, 1, 1)
        p4 = self.conv2(p5)
        p3 = self.conv3(p4)
        p2 = self.conv4(p3)
        p1 = self.conv5(p2)
        
        # Multi-layer perceptron for each output
        p_concat = torch.cat((p1, p2, p3, p4, p5), dim=1)
        p_concat = p_concat.view(p_concat.size(0), -1)
        
        out = self.classifier(p_concat)
        out = self.sigmoid(out)
        
        return out