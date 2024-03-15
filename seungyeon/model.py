import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# CheXNet (DenseNet121) 모델 로드 및 수정
class CheXNet(nn.Module):
    def __init__(self, num_classes):
        super(CheXNet, self).__init__()
        # DenseNet121 로드
        self.densenet121 = models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features
        self.num_classes = num_classes
        
        # CheXNet의 마지막 분류층을 변경
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)

    # Feature Pyramid Network (FPN) 설정
    def get_model_fpn(self):
        # 기본 모델로 ResNet50 불러오기
        backbone = models.resnet50(pretrained=True)
        backbone_out_channels = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 분류층 제거

        # FPN용 AnchorGenerator 생성
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

        # RoI 풀링 설정
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        
        # Faster R-CNN 모델에 위에서 정의한 백본, 앵커 생성기, RoI 풀링 사용하여 모델 생성
        model = FasterRCNN(backbone,
                        num_classes=self.num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)

        return model