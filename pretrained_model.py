'''
CheXNet 훈련시켜서 모델 저장하기 위함
'''
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import CustomDataset
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data = pd.read_csv("Unique_Label_ver2.csv")
finding_labels = sorted(data['Finding Labels'].unique())
label_map = {label: idx for idx, label in enumerate(finding_labels)}
data['Finding Labels'] = data['Finding Labels'].map(label_map)

train_X, test_X, train_y, test_y = train_test_split(data["Image Index"].values, data["Finding Labels"].values, 
                                                    test_size=0.3, random_state=0, stratify=list(data["Finding Labels"].values))

data_list = {"Train":[], "Test":[]}
data_list['Train'].extend([[i,l] for i,l in zip(train_X, train_y)])
data_list['Test'].extend([[i,l] for i,l in zip(train_X, train_y)])

classes = sorted(data['Finding Labels'].unique())

# Load datasets
train_dataset = CustomDataset(data_list["Train"], transform=train_transform)

# Split train_dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define DenseNet-121 model
model = torchvision.models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(classes))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_val_loss = float('inf')
best_epoch = 0
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}: Pretrain Model Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}: Pretrain Model Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
    
    # Save the model if validation loss is improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'pretrained_model/ChexNet_model.pth')
        print("Saved model at epoch", best_epoch+1)