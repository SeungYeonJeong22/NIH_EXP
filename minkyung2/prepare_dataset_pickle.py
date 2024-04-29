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
from tqdm import tqdm

cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
        "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]

### Questa è quella originale del dataset
df = pd.read_csv('/userHome/userhome3/minkyung/NIH_EXP/minkyung2/Data_Entry_2017.csv')
# Questa invece è la versione tmp diciamo di debug con solo qualche paziente
# df = pd.read_csv('./Data_Entry_2017_v2020_debug.csv')

df = df.loc[:,["Image Index","Finding Labels"]]
## construct TEST dataset pickle file ############################################################

# test_images_path = 'Y:/raid/home/gianlucacarloni/causal_medimg/images/test/'
# trainval_images_path = './images/trainval/'
test_images_path = './images_resize/test/'

test_data = {}
failed_images = []

for idx,imgname in tqdm(enumerate(glob(test_images_path+"*.png")), total=len(glob(test_images_path+"*.png")), desc="Processing Images"):
    try :
        img = Image.open(imgname)
        img = np.array(img)
        png_name = os.path.basename(imgname)    
        df_local = df.loc[df["Image Index"] == png_name]
        labels = df_local["Finding Labels"].values
        labels = labels[0].split("|") #eg, "[Cardiomegaly,"Effusion"]
        onehotencoding = [int(elem in labels) for elem in cate] #eg, [1,0,0,0,...,0,1]    
        #img = Image.open(imgname)
        #img = np.array(img) 
        test_data[idx]=[png_name, img, np.array(onehotencoding)]   
    except IOError as e :
        print(f"Error reading image {imgname} : {e}")
        failed_images.append(imgname)

try:
    with open('testdata_new.pickle', 'wb') as handle:
    # with open('test_new_debug.pickle', 'wb') as handle: ###################### TODO ####
        # pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, handle)
except Exception as e:
    print("pickle dump test_new.pickle Error!!")
    print(e)

with open('failed_images.txt', 'w') as f :
    for item in failed_images :
        f.write("%s\n" % item)
    f.close()

### Check if the saved pickle dataset can be correctly loaded and inspect some key-values pairs
# with open('trainvaldata_new.pickle', 'rb') as h:
#    data = pickle.load(h)
# print(len(data))
# print(data[0][0])
# print(data[0][1])
# print(data[0][2])
# print()
# print(data[3][0])
# print(data[3][1])
# print(data[3][2])