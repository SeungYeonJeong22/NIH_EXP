'''RetinaFPN in PyTorch.

See the paper "Focal Loss for Dense Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RetinaFPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(RetinaFPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer2 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        
        
        # Custom 
        self.fc1 = nn.Linear(28, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.final_bn = nn.BatchNorm1d(1024)
        self.output = nn.Linear(1024, 15)
        
        # #Todo -> 2048 x 최대한 짧게만
        # self.fc1 = nn.Linear(256, 2048)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.final_bn = nn.BatchNorm1d(1024)
        # self.output = nn.Linear(1024, 15)
        
        ### Tmp
        self.tmp_flt = nn.Flatten()
        self.tmp_fc = nn.Linear(256 * 28 * 28, 1024)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        
        # return p3, p4, p5, p6, p7
        
        # Custom
        x = self.tmp_flt(p3)
        x = self.tmp_fc(x)
        
        # x = F.relu(self.fc1(p3))
        # x = F.relu(self.fc2(x))
        # x = self.final_bn(self.conv1(x))
        x = torch.sigmoid(self.output(x))
        
        return x
        
class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features=2048, device='cpu'):
        super(MLPBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.global_max_pooling = nn.AdaptiveMaxPool2d(in_features)  # Global Max Pooling
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features, out_features)
        
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.global_max_pooling(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x
    

def RetinaFPN101():
    # return RetinaFPN(Bottleneck, [2,4,23,3])
    return RetinaFPN(Bottleneck, [2,2,2,2])