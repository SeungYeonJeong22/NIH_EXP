# -*- coding: utf-8 -*-
import os 
import numpy as np
from PIL import Image 
import pandas as pd
from glob import glob
import pickle
from tqdm import tqdm

cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
       "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]

mode = "test"

output_pickle_path = f"{mode}data2.pickle"
df = pd.read_csv(f'{mode}data.csv')
df = df.loc[:,["Image Index","Finding Labels"]]

## construct TEST dataset pickle file ############################################################
test_images_path = '../data/nih_resize_all/'

data = {}
idx = 0
for _,imgname in tqdm(enumerate(glob(test_images_path+"*.png")), total=len(os.listdir(test_images_path)), desc=output_pickle_path):
    png_name = os.path.basename(imgname)    
    df_local=df.loc[df["Image Index"] == png_name]
    labels = df_local["Finding Labels"].values
    try:
        labels = labels[0].split("|") #eg, "[Cardiomegaly,"Effusion"]
        onehotencoding = [int(elem in labels) for elem in cate] #eg, [1,0,0,0,...,0,1]    
        img = Image.open(imgname)
        img = np.array(img) 
        data[idx]=[png_name, img, np.array(onehotencoding)]
        idx += 1
    except:
        continue

with open(output_pickle_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Check if the saved pickle dataset can be correctly loaded and inspect some key-values pairs
with open(output_pickle_path, 'rb') as h:
    data = pickle.load(h)

print(len(data))
print(data[0][0])
print(data[0][2])
print()
print(data[3][0])
print(data[3][1])

## Then, repeat all of that for the trainval data.