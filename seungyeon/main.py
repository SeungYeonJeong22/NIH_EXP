import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch import optim

from dataset import CustomDataset
from model import FPN
from metric import compute_metrics
import pickle

# from metric import calculate_metrics, compute_metrics

from warnings import filterwarnings
filterwarnings("ignore")

data_list = {'Train':[], 'Test':[]}

train_path = "../../data/traindata.pickle"
test_path = "../../data/testdata.pickle"

train = pickle.load(open(train_path, 'rb'))
test = pickle.load(open(test_path, 'rb'))

for i in range(len(train)):
    data_list['Train'].extend([[train[i][0], train[i][-1]]])
    
for i in range(len(test)):
    data_list['Test'].extend([[test[i][0], test[i][-1]]])

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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu" # MacBook-air-m2 사용

print("device : ", device)

model = FPN(device=device)

backbone = torch.load('model.pth.tar', map_location='cpu')
state_dict = backbone['state_dict']

# 'module.' 접두사 처리
new_state_dict = {}
for key, value in state_dict.items():
    # 'module.'을 제거하고 새로운 딕셔너리에 저장
    new_key = key.replace('module.densenet121.', '')
    new_state_dict[new_key] = value

model = model.to(device=device)
model.load_state_dict(new_state_dict, strict=False)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_test_loss = float('inf')
best_epoch = 0
num_epochs = 30

save_model_path = "save_model"

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
    
    
learning_curve_file = "learning_curve.txt"
train_outputs_file = "train_outputs.txt"
test_outputs_file = "test_outputs.txt"

if os.path.exists(learning_curve_file):
    os.remove(learning_curve_file)
if os.path.exists(train_outputs_file):
    os.remove(train_outputs_file)
if os.path.exists(test_outputs_file):
    os.remove(test_outputs_file)    


with open(learning_curve_file, 'a') as f:
    f.write("\t\ttrain_loss\ttest_loss\n")
f.close()

with open(train_outputs_file, 'a') as f:
    f.write("\t\tTrain outputs\t\t\n")
f.close()

with open(test_outputs_file, 'a') as f:
    f.write("\t\tTest outputs\t\t\n")
f.close()

def make_txt_file(file_name, epoch=0, train_loss=0, test_loss=0, outputs=None):
    #learning curve
    if outputs == None:
        with open(file_name, 'a') as f:
            f.write(f"Epoch:{epoch}\t{train_loss}\t{test_loss}\n")
        f.close()
    else:
        # output file
        with open(file_name, 'a') as f:
            f.write(f"outputs: {outputs}\n")
            f.write("----"*20)
            f.write("\n")
        f.close()
    

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}: Train"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        make_txt_file(train_outputs_file, outputs=outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")
    
    # Test loop
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch}: Test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            make_txt_file(test_outputs_file, outputs=outputs)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # Save outputs and labels for calculating metrics later
            all_outputs.append(outputs)
            all_labels.append(labels)            

    test_loss = test_loss / len(test_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}")

    # Save the model if validation loss is improved
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f'{save_model_path}/FPN_based_ChexNet_model.pth')
        print("Saved model at epoch", best_epoch+1)
        
    # Concatenate all outputs and labels from all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    accuracy, precision, recall, f1 = compute_metrics(all_outputs.detach().cpu().numpy(), all_labels.detach().cpu().numpy())
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    make_txt_file(learning_curve_file, epoch=epoch, train_loss=train_loss, test_loss=test_loss)
