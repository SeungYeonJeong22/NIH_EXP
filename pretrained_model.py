'''
나중에 CheXNet 훈련시켜서 모델 저장하기 위함
'''

import torch
import torch.nn
import torch.optim as optim
import torchvision.models as models


chexNet = models.densenet121(pretrained=True)
