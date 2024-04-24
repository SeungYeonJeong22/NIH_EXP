from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_name, label = self.data_list[idx]
        image_path = os.path.join(f"../../data/nih_resize_all/{image_name}")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label