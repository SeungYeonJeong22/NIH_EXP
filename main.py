import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

from dataset import CustomDataset
from model import FPN

from warnings import filterwarnings
filterwarnings("ignore")


data = pd.read_csv("Unique_Label_ver2.csv")
finding_labels = sorted(data['Finding Labels'].unique())
label_map = {label: idx for idx, label in enumerate(finding_labels)}
data['Finding Labels'] = data['Finding Labels'].map(label_map)

train_X, test_X, train_y, test_y = train_test_split(data["Image Index"].values, data["Finding Labels"].values, 
                                                    test_size=0.3, random_state=0, stratify=list(data["Finding Labels"].values))

data_list = {"Train":[], "Test":[]}
data_list['Train'].extend([[i,l] for i,l in zip(train_X, train_y)])
data_list['Test'].extend([[i,l] for i,l in zip(train_X, train_y)])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                   # 이미지 크기 조정
    transforms.RandomResizedCrop(224),               # 무작위 크롭 및 크기 조정
    transforms.RandomHorizontalFlip(),               # 무작위 수평 뒤집기
    transforms.RandomAffine(degrees=20, shear=0.1),  # 무작위 회전 및 기울이기
    # transforms.v2.RandomZoomOut(0.1),                 # 랜덤 줌 아웃 (v2에만 있는 것 같음)
    transforms.ToTensor(),                           # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),       # 이미지 크기 조정
    transforms.ToTensor(),               # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

train_dataset = CustomDataset(data_list["Train"], transform=train_transform)
test_dataset = CustomDataset(data_list["Test"], transform=test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FPN()
model = model.to(device=device)
model = nn.DataParallel(model).cuda()

# import torchsummary
# print("torchsummary : ", torchsummary.summary(model, (3, 224,224)))

import torch
# print("torch.cuda.current_device() : ", torch.cuda.current_device())

for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), desc="Trainig"):
    inputs = inputs.to(device)
    output = model(inputs)