import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import pickle
# from sklearn.externals import joblib


cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
        "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]

category_map = {cate[i]:i+1 for i in range(15)}

class NIHDataset(data.Dataset):
    # In the following, the pickle files represent the serialized version of dictionary objects.
    # Specifically, each of such dict has image index as keys and a list of three elements as values (img name, img file, img labelvector):
    #           traindata = {
    #                           [0]:["00000003_002", np_img, np_labels],
    #                           [1]:["00000006_003", np_img, np_labels], ...
    #                           ...
    #                           [86523]:["00085471_001", np_img, np_labels], ...
    #                       }
    def __init__(self, data_path,input_transform=None,
                 used_category=-1,train=True):
        self.data_path = data_path
        if train == True:
            print(f"NIHDataset - train: opening pickle file...")
            
            # img_resize 한 후 만든 피클 데이터
            with open('traindata2.pickle', 'rb') as h:
                self.data = pickle.load(h)
        else:
            print(f"NIHDataset - test: opening pickle file...")
            with open('testdata2.pickle', 'rb') as h:
                self.data = pickle.load(h)
                
        print(f"NIHDataset - shuffling data...")
        random.shuffle(self.data)
        self.category_map = category_map
        self.input_transform = input_transform
        self.used_category = used_category


    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][1]).convert("RGB")
        label = np.array(self.data[index][2]).astype(np.float64)
        if self.input_transform:
            img = self.input_transform(img)
        return img, label

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(15)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0 
        return label

    def __len__(self):
        return len(self.data)




