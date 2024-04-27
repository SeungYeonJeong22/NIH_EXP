import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import gc

class FPN(nn.Module):
    def __init__(self, device='cpu', num_classes=15):
        super(FPN, self).__init__()
        self.device = device
        self.backbone = models.densenet121(pretrained=True)
        
        kernel_size = 1
        stride = 1
        
        # l1~l4는 p2~p5를 만들기 위한 lateral layer에 쓰이는 conv
        # lateral layer 밑에서부터 l1,l2,l3,l4임
        # l4: p5를 만들어주는 2x2 conv (top down layer에서 쓰이는 layer)
        self.lateral_layer_l4 = nn.Conv2d(1024, 1024, kernel_size=kernel_size)
        
        # 순차적으로 denseblock3->옆으로 가는 conv
        # 순차적으로 denseblock2->옆으로 가는 conv
        # 순차적으로 denseblock1->옆으로 가는 conv
        self.lateral_layer_l3 = nn.Conv2d(512, 512, kernel_size=kernel_size)
        self.lateral_layer_l2 = nn.Conv2d(256, 256, kernel_size=kernel_size)
        self.lateral_layer_l1 = nn.Conv2d(128, 128, kernel_size=kernel_size)
        
        self.lateral_layers = nn.ModuleList([
            self.lateral_layer_l3,
            self.lateral_layer_l2,
            self.lateral_layer_l1
        ])
        
        self.conv2dlayer_l6 = nn.Conv2d(1024, 512, kernel_size=kernel_size, stride=stride)
        self.conv2dlayer_l7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride),
            nn.ReLU()
        )
        
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.batch_layer = nn.BatchNorm2d(512)
        self.last_fc_layer = nn.Linear(1024, 1024)
        
        # Replace final fully connected layer
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        
        # Define FPN layers
        self.bottom_up_layers = list(self.backbone.features)
        self.top_down_layers = nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=kernel_size),
            nn.Conv2d(512, 256, kernel_size=kernel_size),
            nn.Conv2d(256, 128, kernel_size=kernel_size),
        ])
        
        self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        # Bottom-up pathway
        bottom_up_features = self.bottom_up_forward(x)
        
        # Top-down pathway
        top_down_features, p5_features = self.top_down_forward(bottom_up_features)
        
        p4_features = self.upsampling(self.lateral_layer_l3(bottom_up_features[2]) + top_down_features[1])
        p3_features = self.upsampling(self.lateral_layer_l2(bottom_up_features[1]) + self.top_down_layers[1](p4_features))
        p2_features = self.upsampling(self.lateral_layer_l1(bottom_up_features[0]) + self.top_down_layers[2](p3_features))
        
        p6_features = self.conv2dlayer_l6(bottom_up_features[-1])
        p7_features = self.conv2dlayer_l7(p6_features)
        
        p_features = []
        p_features.append(p5_features)
        p_features.append(p4_features)
        p_features.append(p3_features)
        p_features.append(p2_features)
        p_features.append(p6_features)
        p_features.append(p7_features)
        
        mlp_output = []
        for feats in p_features:
            mlp_block = MLPBlock(feats.shape[1], device=self.device)
            mlp_block.to(device=self.device)
            mlp_output.append(mlp_block(feats))
            
        mlp_block = MLPBlock(bottom_up_features[-1].shape[1], device=self.device)
        mlp_block.to(device=self.device)
        
        # torch.cuda.empty_cache()
        # gc.collect()
        
        mlp_output.append(mlp_block(bottom_up_features[-1]))
        
        concatenated_features = torch.cat(mlp_output, dim=1)
        
        fc_network = FCNetwork(concatenated_features.size(1), device=self.device)
        fc_network.to(device=self.device)
        
        outputs = fc_network(concatenated_features)
        
        return outputs
    
    
    def bottom_up_forward(self, x):
        bottom_up_features = []
        # backbone feature를 전부 다 돌면서

        for idx, layer in enumerate(self.bottom_up_layers):
            x = layer(x)
            # # DenseBlock을 지났으면 bottom_up_features에 레이어 통과한 것을 추가
            # if str(layer).__contains__("DenseBlock"):
            #     bottom_up_features.append(x)
            
            # Transition_layer을 지났으면 bottom_up_features에 레이어 통과한 것을 추가
            if str(layer).__contains__("Transition"):
                bottom_up_features.append(x)
                
            if idx == len(self.bottom_up_layers)-1:
                bottom_up_features.append(x)
        return bottom_up_features
    
    
    def top_down_forward(self, bottom_up_features):
        top_down_features = []
        prev_features = None
        # p5 까지 해주려면 top_down_layer 개수 3개에 하나 더 해줌
        # for i, layer in enumerate(len(self.top_down_layers)+1):
        for i in range(len(self.top_down_layers)+1):
            if i == 0:
                # nn.conv2d (2x2)를 지난 p5
                prev_features = self.lateral_layer_l4(bottom_up_features[-1])
                top_down_features.append(prev_features)
                p5_features = prev_features
            else:
                # p4~p2
                prev_features = top_down_features[-1]
                # top_down_features.append(layer(prev_features))
                top_down_features.append(self.top_down_layers[i-1](prev_features))
        # return top_down_features[::-1]
        return top_down_features, p5_features


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features=2048, device='cpu'):
        super(MLPBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.gap = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features, out_features)
        
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x

class FCNetwork(nn.Module):
    def __init__(self, concatenated_feature_size, device='cpu'):
        super(FCNetwork, self).__init__()
        # Assuming the concatenated feature size from all MLP blocks is concatenated_feature_size
        self.fc1 = nn.Linear(concatenated_feature_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.final_bn = nn.BatchNorm1d(1024)
        self.output = nn.Linear(1024, 15)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final_bn(x)
        x = torch.sigmoid(self.output(x))
        return x