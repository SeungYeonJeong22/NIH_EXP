# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:27:33 2023

@author: gianl
"""
import os 

import numpy as np
from PIL import Image 
import pandas as pd
from glob import glob
import pickle

cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
        "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]

### Questa è quella originale del dataset
df = pd.read_csv('Y:/raid/home/gianlucacarloni/causal_medimg/Data_Entry_2017_v2020.csv')
# Questa invece è la versione tmp diciamo di debug con solo qualche paziente
# df = pd.read_csv('./Data_Entry_2017_v2020_debug.csv')

df = df.loc[:,["Image Index","Finding Labels"]]
## construct TEST dataset pickle file ############################################################

# test_images_path = 'Y:/raid/home/gianlucacarloni/causal_medimg/images/test/'
test_images_path = './images/test/'

test_data = {}
for idx,imgname in enumerate(glob(test_images_path+"*.png")):
        
    png_name = os.path.basename(imgname)    
    df_local=df.loc[df["Image Index"] == png_name]
    labels = df_local["Finding Labels"].values
    labels = labels[0].split("|") #eg, "[Cardiomegaly,"Effusion"]
    onehotencoding = [int(elem in labels) for elem in cate] #eg, [1,0,0,0,...,0,1]    
    img = Image.open(imgname)
    img = np.array(img) 
    test_data[idx]=[png_name, img, np.array(onehotencoding)]

with open('testdata_new.pickle', 'wb') as handle:
# with open('test_new_debug.pickle', 'wb') as handle: ###################### TODO ####
    # pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_data, handle)

### Check if the saved pickle dataset can be correctly loaded and inspect some key-values pairs
# with open('trainvaldata_dgx.pickle', 'rb') as h:
#    data = pickle.load(h)
# print(len(data))
# print(data[0][0])
# print(data[0][1])
# print(data[0][2])
# print()
# print(data[3][0])
# print(data[3][1])
# print(data[3][2])