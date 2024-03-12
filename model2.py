import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class FPN(nn.Module):
    def __init__(self, num_classes=14):
        super(FPN, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        
        # l1~l4는 p2~p5를 만들기 위한 lateral layer에 쓰이는 conv
        # lateral layer 밑에서부터 l1,l2,l3,l4임
        # l4: p5를 만들어주는 2x2 conv (top down layer에서 쓰이는 layer)
        # self.lateral_layer_l4 = nn.Conv2d(1024, 1024, kernel_size=2, padding=1, dilation=1)
        self.lateral_layer_l4 = nn.Conv2d(1024, 1024, kernel_size=2)
        
        # 순차적으로 denseblock3->옆으로 가는 conv
        # 순차적으로 denseblock2->옆으로 가는 conv
        # 순차적으로 denseblock1->옆으로 가는 conv
        self.lateral_layer_l3 = nn.Conv2d(512, 512, kernel_size=2)
        self.lateral_layer_l2 = nn.Conv2d(256, 256, kernel_size=2)
        self.lateral_layer_l1 = nn.Conv2d(128, 128, kernel_size=2)
        
        self.lateral_layers = nn.ModuleList([
            self.lateral_layer_l3,
            self.lateral_layer_l2,
            self.lateral_layer_l1
        ])
        
        self.conv2dlayer_l6 = nn.Conv2d(512, 512, kernel_size=2, padding=1, stride=2)
        self.conv2dlayer_l7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, padding=1, stride=2),
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
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 128, kernel_size=1),
        ])
        
        
    def forward(self, x):
        # Bottom-up pathway
        bottom_up_features = self.bottom_up_forward(x)
        
        # Top-down pathway
        top_down_features, p5_features = self.top_down_forward(bottom_up_features)
        
        # p4~p2 까지의 feature
        # 각각의 dense block output에 conv2d를 적용한 것과 top-down feature를 합한 feature
        # p4_p2_features = [
        #     # top_down은 p5 이후부터
        #     self.upsampling(lateral_layer(bottom_up_features[2-i]) + top_down_features[i+1])
        #     for i, lateral_layer in enumerate(self.lateral_layers)
        # ]
        
        p4_features = self.upsampling(self.lateral_layer_l3(bottom_up_features[2]) + top_down_features[1])
        p3_features = self.upsampling(self.lateral_layer_l2(bottom_up_features[1]) + self.top_down_layers[1](p4_features))
        p2_features = self.upsampling(self.lateral_layer_l1(bottom_up_features[0]) + self.top_down_layers[2](p3_features))
        
        p6_features = [self.conv2dlayer_l6(bottom_up_features[-1])]
        p7_features = [self.conv2dlayer_l7(p6_features)]
        
        p_features = []
        # p_features.extend(p4_p2_features)
        p_features.extend(p5_features)
        p_features.extend(p4_features)
        p_features.extend(p3_features)
        p_features.extend(p2_features)
        p_features.extend(p6_features)
        p_features.extend(p7_features)
        
        mlp_output = []
        for feats in p_features:
            mlp_output.append(self.MLPBolck(feats))
        mlp_output.append(self.MLPBolck(bottom_up_features))
        
        outputs = torch.concat(mlp_output)
        outputs = self.batch_layer(outputs)
        outputs = self.last_fc_layer(outputs)
        outputs = self.backbone.classifier(outputs)
        
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


class MLPBolck(nn.Module):
    def __init__(self, in_features):
        super(MLPBolck, self).__init__()
        # BatchNorm2d에 상수가 아니라 in_features로 받아야 할 듯함
        self.network = nn.Sequential(nn.BatchNorm2d(1024),
                                    nn.MaxPool2d(),
                                    nn.Dropout(0.5),
                                    nn.Linear(2048, 1024))
    def forward(self, x):
        return self.network(x)
    
    def __len__(self, x):
        return len(x)

# import torch
# import torch.nn as nn
# import torchvision.models as models

# class FPN(nn.Module):
#     def __init__(self, backbone='densenet121'):
#         super(FPN, self).__init__()
        
#         # Backbone 설정
#         if backbone == 'densenet121':
#             self.backbone = models.densenet121(pretrained=True)
#             # DenseNet의 transition layer의 커널 크기를 (2, 2)로 변경
#             for name, module in self.backbone.named_modules():
#                 if 'transition' in name and name.endswith("conv"):
#                     module.kernel_size = (2, 2)
        
#         # Bottom-up Pathway 설정
#         self.bottom_up_pathway = nn.ModuleList([
#             self.backbone.features.conv0,
#             self.backbone.features.denseblock1,
#             self.backbone.features.transition1,
#             self.backbone.features.denseblock2,
#             self.backbone.features.transition2,
#             self.backbone.features.denseblock3,
#             self.backbone.features.transition3,
#             self.backbone.features.denseblock4,
#         ])
#         self.bottom_up_layers = list(self.backbone.features)
        
#         # Top-down Pathway 설정
#         self.top_down_pathway = nn.ModuleList([
#             nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1)),
#             nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1)),
#         ])
        
#         # Lateral Layer 설정
#         self.lateral_layer = nn.ModuleList([
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2)),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2)),
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(2, 2)),
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(2, 2)),
#         ])
        
#         # Upsample 설정
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
#     def forward(self, x):
#         # Bottom-up Pathway
#         # bottom_up_features = []
#         # for layer in self.bottom_up_pathway:
#         #     x = layer(x)
#         #     if str(layer).__contains__("Transition"):
#         #         print("bottom_up_features : ")
#         #         bottom_up_features.append(x)
                
#         bottom_up_features = []
#         # backbone feature를 전부 다 돌면서

#         for idx, layer in enumerate(self.bottom_up_layers):
#             x = layer(x)
#             # # DenseBlock을 지났으면 bottom_up_features에 레이어 통과한 것을 추가
#             # if str(layer).__contains__("DenseBlock"):
#             #     bottom_up_features.append(x)
            
#             # Transition_layer을 지났으면 bottom_up_features에 레이어 통과한 것을 추가
#             if str(layer).__contains__("Transition"):
#                 bottom_up_features.append(x)
                
#             if idx == len(self.bottom_up_layers)-1:
#                 bottom_up_features.append(x)                
        
#         # Top-down Pathway
#         top_down_features = []
#         for i, (lateral_layer, top_down_layer) in enumerate(zip(self.lateral_layer, self.top_down_pathway)):
#             if i == 0:
#                 top_down_feature = top_down_layer(bottom_up_features[-1])
#             else:
#                 top_down_feature = top_down_layer(self.upsample(top_down_features[-1]))
#             lateral_feature = lateral_layer(bottom_up_features[-(i + 2)])
#             top_down_features.append(top_down_feature + lateral_feature)
        
#         return top_down_features