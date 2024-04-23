# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:02:24 2024

@author: gianl
"""
# https://iust-projects.ir/post/cv02/

#computing
import numpy as np
from scipy.fft import fft2, fftshift
#image readining, storing, manipulation
from PIL import Image
# graphics and plot
import matplotlib.pyplot as plt

#%%
cxr_a = Image.open(r'Y:/raid/home/gianlucacarloni/causal_medimg/dataset_mimicCXR_JPG/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10001217/s58913004/5e54fc9c-37c49834-9ac3b915-55811712-9d959d26.jpg')
cxr_b = Image.open(r'Y:/raid/home/gianlucacarloni/causal_medimg/dataset_mimicCXR_JPG/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10001176/s53186264/1fe73f8e-036bd24e-4578c891-33c1746e-864884a7.jpg')


cxr_a = Image.open(r'C:/Users/gianl/Documents/fake-lung11.png').convert('L')
dim=3056

dim = max(max(cxr_a.size), max(cxr_b.size))
cxr_a=cxr_a.resize((dim,dim))
cxr_b=cxr_b.resize((dim,dim))


plt.figure(figsize=(10,10))
plt.subplot(121)
plt.title('A: image')
plt.imshow(cxr_a, cmap='gray')
plt.subplot(122)
plt.title('B: image')
plt.imshow(cxr_b, cmap='gray')




#%%
f_a = fftshift(fft2(cxr_a))
f_b = fftshift(fft2(cxr_b))

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.title('A: fft2')
plt.imshow(np.log(np.abs(f_a)), cmap='gray')
plt.subplot(122)
plt.title('B: fft2')
plt.imshow(np.log(np.abs(f_b)), cmap='gray')



#%% amplitude = np.sqrt(real ** 2 + imaginary ** 2)
### phase = np.arctan2(imaginary / real).
amp_a = np.sqrt(np.real(f_a) ** 2 + np.imag(f_a) ** 2)
pha_a = np.arctan2(np.imag(f_a), np.real(f_a))
amp_b = np.sqrt(np.real(f_b) ** 2 + np.imag(f_b) ** 2)
pha_b = np.arctan2(np.imag(f_b), np.real(f_b))

cut_dim=512
cut=(dim//2-cut_dim, dim//2 +cut_dim)
amp_a_cut_b = np.copy(amp_a)
amp_a_cut_b[cut[0]:cut[1], cut[0]:cut[1]]=amp_b[cut[0]:cut[1], cut[0]:cut[1]]

plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('A: amplitude')
plt.imshow(np.log(amp_a+1e-10), cmap='gray')
plt.subplot(222)
plt.title('A: phase')
plt.imshow(pha_a, cmap='gray')
plt.subplot(223)
plt.title('B: amplitude')
plt.imshow(np.log(amp_b+1e-10), cmap='gray')
plt.subplot(224)
plt.title('B: phase')
plt.imshow(pha_b, cmap='gray')

plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.title('Amplitude A')
plt.imshow(np.log(amp_a+1e-10), cmap='gray')
plt.subplot(132)
plt.title('Amplitude A with 512-cut from B')
plt.imshow(np.log(amp_a_cut_b+1e-10), cmap='gray')
plt.subplot(133)
plt.title('Amplitude B')
plt.imshow(np.log(amp_b+1e-10), cmap='gray')

#%% amplitude_phase
# a = np.multiply(amp_a, np.exp(1j * pha_a))
# a = np.fft.ifft2(a)  

b_a = np.multiply(amp_b, np.exp(1j * pha_a))
b_a = np.fft.ifft2(b_a)  
a_b = np.multiply(amp_a, np.exp(1j * pha_b))
a_b = np.fft.ifft2(a_b)  

#
a_cut_b = np.multiply(amp_a_cut_b, np.exp(1j * pha_a))
a_cut_b = np.fft.ifft2(a_cut_b)  


plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('A')
plt.imshow(cxr_a, cmap='gray')
plt.subplot(322)
plt.title('B')
plt.imshow(cxr_b, cmap='gray')
plt.subplot(323)
plt.title('B_A: Amplitude from B, phase from A')
plt.imshow(np.abs(b_a), cmap='gray')
plt.subplot(324)
plt.title('A_B: Amplitude from A, phase from B')
plt.imshow(np.abs(a_b), cmap='gray')
plt.subplot(325)
plt.title('Amplitude from A cut-B, phase from A')
plt.imshow(np.abs(a_cut_b), cmap='gray')
plt.subplot(326)
plt.title('6: Difference 3 and 5')
plt.imshow(np.abs(a_b-a_cut_b), cmap='gray')
