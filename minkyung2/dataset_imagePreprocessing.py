# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:37:07 2024

A simple Python script that uses the PIL library to loop over all PNG images
in a specified subfolder, opens them in grayscale mode, and adjusts the contrast
using the formula 255.0*((x-x_min)/(x_max - x_min))

@author: gianl
"""

from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define the subfolder containing the images
subfolder = 'Y:/raid/home/gianlucacarloni/causal_medimg/images/trainval/'

avg_img=np.zeros((1024,1024))
# Loop over all PNG images in the subfolder
for filename in tqdm(os.listdir(subfolder)):
    if filename.endswith('.png') and ("BAD" not in filename):
        # Construct the full file path
        file_path = os.path.join(subfolder, filename)
    
        # Open the image in grayscale mode
        with Image.open(file_path).convert('L') as img:
            
            if img.size != (1024,1024):
                img=img.resize((1024,1024))

            # Convert the image to a numpy array for contrast adjustment
            img_array = np.array(img)            
    
            # Apply the contrast adjustment formula
            x_min = img_array.min()
            x_max = img_array.max()
            img_array = 255.0 * ((img_array - x_min) / (x_max - x_min))
            
            avg_img += img_array/86114
    
            # Convert the adjusted array back to an image
            # img_contrast = Image.fromarray(img_array.astype('uint8'), 'L')
    
            # Save or display the adjusted image
            # img_contrast.show()  # Uncomment to display the image
            #
            # file_path_final = os.path.basename(file_path)
            # file_path_final = file_path_final.replace(".png", "_hs.png") # HistogramStretched version of the image
            # img_contrast.save(file_path_final+"_hs")  
            # 
            
            
avg_img = Image.fromarray(avg_img.astype('uint8'), 'L')            
avg_img.save(os.path.join(subfolder,"0_AVG_IMAGE.png"))
            
print('Contrast adjustment completed for all images.')
