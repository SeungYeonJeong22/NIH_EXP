import torch
import torch.nn
import torch.optim as optim
import torchvision.models as models


chexNet = models.densenet121(pretrained=True)
