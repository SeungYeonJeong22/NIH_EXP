from dataset import CustomDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from warnings import filterwarnings
filterwarnings("ignore")


data = pd.read_csv("Unique_Label_ver2.csv")
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

for batch_idx, (inputs, labels) in enumerate(train_loader):
    model(inputs)
    print("Batch:", batch_idx, "Input shape:", inputs.shape)

