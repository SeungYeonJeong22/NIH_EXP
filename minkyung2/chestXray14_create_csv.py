# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:34:06 2024

@author: gianl
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#%% creazione del CSV appropriato

# csv=pd.read_csv(r'Y:/raid/home/gianlucacarloni/causal_medimg/Data_Entry_2017_v2020_my.csv')

# names_test=pd.read_csv(r'Y:/raid/home/gianlucacarloni/watchYourNeurons/chestXray14/filenames_test.txt',header=None)
# names_trainval=pd.read_csv(r'Y:/raid/home/gianlucacarloni/watchYourNeurons/chestXray14/filenames_trainval.txt',header=None)

# csv_test = csv[csv["Image Index"].isin(names_test[0])]
# csv_test.to_csv(r'Y:/raid/home/gianlucacarloni/causal_medimg/Data_Entry_my_test.csv',index=False)

# csv_trainval = csv[csv["Image Index"].isin(names_trainval[0])]
# csv_trainval.to_csv(r'Y:/raid/home/gianlucacarloni/causal_medimg/Data_Entry_my_trainval.csv',index=False)

#%% Analisi degli attributi:
    #Patient AGE --- histogram
# TRAIN
df = pd.read_csv(r'Y:/raid/home/gianlucacarloni/causal_medimg/Data_Entry_my_trainval.csv')
df_age = df['Patient Age']
age_min=df_age.min()
age_max=df_age.max()

plt.hist(df_age,bins=age_max-age_min,color='k')
plt.xticks(np.arange(age_min, age_max+1, 10.0))
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.title(f'MIN {age_min}, MEDIAN {df_age.median()}, MAX {age_max}')
plt.show()

plt.figure()
plt.hist(df_age,bins=age_max-age_min,color='k',cumulative=True)
plt.xticks(np.arange(age_min, age_max+1, 10.0))
plt.yticks(np.arange(0, len(df_age)+1, 5000))
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.title(f'MIN {age_min}, MEDIAN {df_age.median()}, MAX {age_max}')
plt.show()

# # TEST
# df = pd.read_csv(r'Y:/raid/home/gianlucacarloni/causal_medimg/Data_Entry_my_test.csv')
# df_age = df['Patient Age']
# age_min=df_age.min()
# age_max=df_age.max()
# plt.hist(df_age,bins=age_max-age_min,color='k')
# plt.xticks(np.arange(age_min, age_max+1, 10.0))
# plt.xlabel('Patient Age')
# plt.ylabel('Count')
# plt.title(f'MIN {age_min}, MEDIAN {df_age.median()}, MAX {age_max}')
# plt.show()
# plt.figure()
# plt.hist(df_age,bins=age_max-age_min,color='k',cumulative=True)
# plt.xticks(np.arange(age_min, age_max+1, 10.0))
# plt.yticks(np.arange(0, len(df_age)+1, 5000))
# plt.xlabel('Patient Age')
# plt.ylabel('Count')
# plt.title(f'MIN {age_min}, MEDIAN {df_age.median()}, MAX {age_max}')
# plt.show()

    # Follow-up # (number)
df_fu = df['Follow-up #']
fu_min=df_fu.min()
fu_max=df_fu.max()  
plt.hist(df_fu,bins=fu_max-fu_min,color='k')
plt.xticks(np.arange(fu_min, fu_max+1, 10.0))
plt.xlabel('Follow-up Number')
plt.ylabel('Count')
plt.title(f'MIN {fu_min}, MEDIAN {df_fu.median()}, MAX {fu_max}')
plt.show()

# for the time being, just exclude the ones having #1
df_fu = df[df['Follow-up #']>0]
df_fu = df_fu['Follow-up #']
fu_min=df_fu.min()
fu_max=df_fu.max()  
plt.hist(df_fu,bins=fu_max-fu_min,color='k',cumulative=True)
plt.xticks(np.arange(fu_min, fu_max+1, 10.0))
plt.xlabel('Follow-up Number (>0)')
plt.ylabel('Count')
plt.title(f'MIN {fu_min}, MEDIAN {df_fu.median()}, MAX {fu_max}')
plt.show()
