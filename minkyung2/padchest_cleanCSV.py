# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:14:15 2024

From the original "PADCHEST_chest_x_ray_images_labels_160K_01.02.19", I created a clean version
via MS Excel utilities, but for ease of use I here utilize python to get rid of the lateral views,
organize the output GT label, etc, and create a new final version.

Moreover, we can here perform some analyses regarding frequency and distribution of certain fields. 

@author: gianl
"""

import os
import numpy as np
import pandas as pd
import re

df = pd.read_csv('Y:/raid/home/gianlucacarloni/causal_medimg/dataset_padchest/padchest_cleanGC.csv')

# df.columns
# Out[3]: 
# Index(['ImageID', 'ImageDir', 'StudyID', 'PatientID', 'PatientAge',
#        'PatientSex_DICOM', 'ViewPosition_DICOM', 'Projection', 'Pediatric',
#        'Modality_DICOM', 'Manufacturer_DICOM', 'SpatialResolution_DICOM',
#        'BitsStored_DICOM', 'WindowCenter_DICOM', 'WindowWidth_DICOM',
#        'Rows_DICOM', 'Columns_DICOM', 'XRayTubeCurrent_DICOM',
#        'Exposure_DICOM', 'ExposureInuAs_DICOM', 'ExposureTime',
#        'RelativeXRayExposure_DICOM', 'Labels'],
#       dtype='object')

#To select rows in a pandas DataFrame where a specific letter does not appear in the string values of a column,
df = df[~ df['ViewPosition_DICOM'].astype(str).str.contains('l', case=False)]
# and then select only AP, PA, and AP_horizontal
df = df[df['Projection'].astype(str).str.contains('AP|horizontal|PA', case=True, regex=True)]




tmp_labels = df['Labels']
import ast
# Convert string representations of lists to actual lists
tmp_labels = tmp_labels.apply(ast.literal_eval)
# Now extract the set of unique labels
unique_labels = set(label for sublist in tmp_labels for label in sublist)
# new_labels = df['Labels']
#
df['Labels']=df['Labels'].apply(ast.literal_eval)


##

# Remove rows where column labels has no findings
df = df[df['Labels'].apply(lambda x: x != [''])]

## these are the final desired labels of the possible findings, mostly shared among all the CXR datasets
desired_labels = [
    'Aortic Enlargement',
	'Atelectasis',
    'Enlarged Cardiomediastinum', 
    'Calcification',
	'Cardiomegaly',
	'Consolidation',
	'Edema',
 	'Effusion', #Pleural, not pericardial
    'Emphysema',
    'Fibrosis', #/PulmunaryFibrosis',
    'Fracture', #Fracture/RibFracture/ClavicleFracure', 'humeral fracture','vertebral fracture','clavicle fracture','rib fracture',
    'Hernia',
    'Interstitial Lung Disease (ILD)',
    'Infiltration',
    'Lung Cavity',
    'Lung Cyst',
    'Lung Lesion',
    'Lung Opacity', 
    'Lung Tumor', 
    'Mass',
	'Mediastinal Shift',
	'No Finding',
	'Nodule',
	'Pleural Other',
	'Pleural Thickening',
	'Pneumonia',
	'Pneumothorax',
	'Tubercolosis',
	'Support Devices',
    'Chest Drain Tube', #TODO create a new desired_label called "Chest Drain Tube" specializing Support Devices
    ]



labels_pulmunary = {
    #related to lung opacity
    'lung opacity':['Lung Opacity'],
    'atelectasis':['Atelectasis'],
    'atelectasis basal':['Atelectasis'],
    'laminar atelectasis':['Atelectasis'],
    'lobar atelectasis':['Atelectasis'],
    'round atelectasis':['Atelectasis'],
    'segmental atelectasis':['Atelectasis'],
    'total atelectasis':['Atelectasis'],
    #
    'consolidation':['Consolidation','Lung Opacity'],
    'air trapping':['Emphysema'],##partially overlaps
    'emphysema':['Emphysema'], #-----diverso da subcutaneous
    'edema':['Edema'],
    'pulmonary edema':['Edema','Lung Opacity'],
    'pulmunary edema':['Edema','Lung Opacity'],
    'infiltrates':['Infiltration'],
    'infiltration':['Infiltration'],
    'ground glass pattern':['Lung Opacity'],
    'fibrotic band':['Fibrosis'],##partially overlaps
    'pulmunary fibrosis':['Fibrosis'],
    'pulmonary fibrosis':['Fibrosis'],
    'fibrosis':['Fibrosis'],
    'alveolar pattern':['Lung Opacity'],##may indicate pneumonia,
    'miliary opacities':['Lung Opacity'],
    #related to masses or nodules
    'lung lesion':['Lung Lesion'],
    'nodule':['Nodule','Lung Lesion'],
    'multiple nodules':['Nodule','Lung Lesion'],
    'mass':['Mass','Nodule','Lung Lesion','Lung Tumor'],
    'pulmonary mass':['Mass','Nodule','Lung Lesion','Lung Tumor'],
    'pulmunary mass':['Mass','Nodule','Lung Lesion','Lung Tumor'],
    'pleural mass':['Mass','Nodule','Lung Lesion','Lung Tumor'],
    'calcified granuloma':['Calcification','Mass'],
    'calcification':['Calcification'],
    'granuloma':['Mass'],
    'lymphangitis carcinomatosa':['Lung Tumor'],
    'lung tumor':['Lung Tumor'],
    'lung tumour':['Lung Tumor'],
    'cyst':['Lung Cyst'],
    #related to pleural findings
    'effusion':['Effusion'],
    'pleural effusion':['Effusion'],
    'loculated pleural effusion':['Effusion'],
    'pleural thickening':['Pleural Thickening'],
    'apical pleural thickening':['Pleural Thickening'],
    'calcified pleural thickening':['Pleural Thickening','Calcification'],
    'pleural other':['Pleural Other'], 
    'pleural plaques':['Pleural Other'],   
    'calcified pleural plaques':['Pleural Other','Calcification'],  
    'pneumothorax':['Pneumothorax'],
    'hydropneumothorax':['Pneumothorax','Effusion'],
    'fissure thickening':['Pleural Other'],
    'minor fissure thickening':['Pleural Other'],
    'major fissure thickening':['Pleural Other'],
    'loculated fissural effusion':['Pleural Other','Effusion'],
    #related to lung function
    'COPD signs':['Emphysema','Interstitial Lung Disease (ILD)'], #partial overlap and may be present, respectively
    'bullas':['Emphysema'],##----
    'kerley lines':['Edema'], #usually associated with    
    'hyperinflated lung':['Emphysema'], #common COPD feature, can be caused by emphysema
    'hypoexpansion':['Atelectasis'], #might be a cause,
    'hypoexpansion basal':['Atelectasis'], #might be a cause,:
    #related to infectious findings
    'tuberculosis':['Tubercolosis'],  
    'tuberculosis sequelae':['Tubercolosis'],  
    'pneumonia':['Pneumonia', 'Lung Opacity', 'Infiltration'],#depending on severity
    'atypical pneumonia':['Pneumonia', 'Lung Opacity', 'Infiltration'],
    'abscess':['Lung Cavity'],
    'empyema':['Effusion','Pleural Other'],
    #other    
    'calcified fibroadenoma':['Calcification'],  #not lung related 
    'bronchovascular markings':['No Finding','Interstitial Lung Disease (ILD)','Lung Opacity','Pneumonia'],#can be normal or a sign for another condition
    'mediastinal mass':['Mass'], #------mass, broad term, could be lung mass but needs differentiation
    'interstitial lung disease':['Interstitial Lung Disease (ILD)'],
    'interstitial pattern':['Interstitial Lung Disease (ILD)'],
    'reticular interstitial pattern':['Interstitial Lung Disease (ILD)'],
    'reticulonodular interstitial pattern':['Interstitial Lung Disease (ILD)'],
    'lepidic adenocarcinoma':['Lung Tumor'], #-----tumor    
    'asbestosis signs':['Interstitial Lung Disease (ILD)'], #---reticular pattern
    'cavitation':['Lung Cavity'], #--cavity
    'lung cavity':['Lung Cavity'],
    'calcified densities':['Calcification'],
    'normal':['No Finding'],
    'no finding':['No Finding'],
    ####### relating to bones#################
    'fracture':['Fracture'],
    'callus rib fracture':['Fracture'],
    'vertebral fracture':['Fracture'],
    'humeral fracture':['Fracture'],
    'clavicle fracture':['Fracture'],
    'rib fracture':['Fracture'],
    ##related to the heart
    'heart valve calcified':['Calcification'],    
    'cardiomegaly':['Cardiomegaly'],
    'heart insufficiency':['Edema'], #(indirect) possibly leading to.
    #related to the aorta
    'aortic enlargement':['Aortic Enlargement'],
    'aortic button enlargement':['Aortic Enlargement'],
    'aortic elongation':['Aortic Enlargement'],
    'descendent aortic elongation':['Aortic Enlargement'],
    'supra aortic elongation':['Aortic Enlargement'],
    'ascendent aortic elongation':['Aortic Enlargement'],
    'aortic aneurysm':['Aortic Enlargement'],    
    #other
    'lipomatosis':['Mediastinal Shift'], #(indirect) creates a mass effect that potentially causes mediastinal shift      
    'mediastinic lipomatosis':['Mediastinal Shift'], #(indirect) 
    ############related to catheters and tubes
    'support devices':['Support Devices'],
    'chest drain tube':['Support Devices','Chest Drain Tube'],##for Pleural Effusion, Pneumothorax
    'ventriculoperitoneal drain tube':['Support Devices'],   
    'catheter':['Support Devices'],
    'reservoir central venous catheter':['Support Devices'],
    'central venous catheter via jugular vein':['Support Devices'],
    'central venous catheter via umbilical vein':['Support Devices'],
    'central venous catheter':['Support Devices'],
    'central venous catheter via subclavian vein':['Support Devices'],
    #
    'gastrostomy tube':['Support Devices'], #bring food and feeding
    'NSG tube':['Support Devices'], #bring food and feeding
    'nephrostomy tube':['Support Devices'], #drains fluid from kidney
    'endotracheal tube':['Support Devices'],
    'tracheostomy tube':['Support Devices'],    
    #related to implants#########
    'bone cement':['Support Devices','Fracture'],#potentially
    'osteosynthesis material':['Support Devices','Fracture'],#potentially
    'metal':['Support Devices','Fracture'], #,'Calcification'],#potentially, resembling calcium
    'pacemaker':['Support Devices'], #,'Calcification'],#potentially resembling calcium
    'double J stent':['Support Devices'],
    'chamber device':['Support Devices'],
    'single chamber device':['Support Devices'],
    'dual chamber device':['Support Devices'],
    'electrical device':['Support Devices'],
    'artificial heart valve':['Support Devices'], 
    'artificial aortic heart valve':['Support Devices'],
    'artificial mitral heart valve':['Support Devices'],
    'enlarged cardiomediastinum':['Enlarged Cardiomediastinum'],
    'mediastinal shift':['Mediastinal Shift']
    }


# add a new column made of empy lists, eventually to be populated
df['Labels_clean']=np.empty((len(df), 0)).tolist()

def check_wildcarding(unique_labels, dictionary, strings):    
    results={}
    for string in strings:
        matching=set(sorted([s.strip() for s in unique_labels if re.compile(rf'.*{re.escape(string)}.*').search(s)]))
        if len(matching)>1:
            results[string]=matching
    return results

terms_with_multiple_wildcards = check_wildcarding(unique_labels=unique_labels, dictionary=labels_pulmunary, strings=labels_pulmunary.keys())





def safe_remove(lst, value):
    if value in lst:
        lst.remove(value)

def find_matching_keys(dictionary, search_term):
    # This pattern will match any keys that contain the search term, regardless of what comes before or after.
    pattern = re.compile(rf'.*{re.escape(search_term)}.*')
    matching_keys = [key for key in dictionary if pattern.search(key)]  #the matching keys, if any
    
    # Ad hoc modifications:
    if search_term == 'pneumothorax':
        safe_remove(matching_keys,'hydropneumothorax')
    elif search_term =='emphysema':
        safe_remove(matching_keys,'subcutaneous emphysema')
    elif search_term =='nodule':
        safe_remove(matching_keys,'pseudonodule')
    elif search_term =='mass':
        safe_remove(matching_keys,'mediastinal mass')
        safe_remove(matching_keys,'soft tissue mass')
        safe_remove(matching_keys,'breast mass')
    elif search_term =='granuloma':
        safe_remove(matching_keys,'calcified granuloma')
    elif search_term == 'normal':
        safe_remove(matching_keys,'abnormal foreign body')
    
    if len(matching_keys)>0:
        return matching_keys
    else:
        raise KeyError

# populate:
for index, row in df.iterrows():
    labels_list = row['Labels'] #e.g. ['pneumonia','pleural effusion','chronic changes','emphysema','heart insufficiency','interstitial pattern','costophrenic angle blunting']
    labels_clean_set = set()  # Set to keep track of unique labels
    for label in labels_list:
        if label != '': #be aware that some cell contains this strange empty string by default from the original dataset, we must exclude that, otherwise it will match every single key in the dictionary...
            try: #try looking for potential matches between the current terms and the possible desired labels.
                # print(f"{label}:{labels_pulmunary[label]}" )
                matching_keys = find_matching_keys(labels_pulmunary,label.strip())
                for k in matching_keys:
                    # Extend only with unique labels not already in the set
                    new_labels = [item for item in labels_pulmunary[k] if item not in labels_clean_set]
                    row['Labels_clean'].extend(new_labels)
                    labels_clean_set.update(new_labels)
            except KeyError:
                print("skip unrelevant label...")
        else:
            print("Skip empty label.")
    
    row['Labels_clean'] = list(labels_clean_set)
#%% 
# ora dovrò fare tante colonne quante le desired labels, con ogni cella riempita di 0 o 1 in base al valore            

# Remove rows where column labels clean has no findings
df = df[df['Labels_clean'].apply(lambda x: len(x) > 0)]

# Add new columns with default value of 0
for column in desired_labels:
    df[column] = 0

for index, row in df.iterrows():
    labels_list = row['Labels_clean'] 
    if len(labels_list)>0:
        for label in labels_list:
            df.at[index, label] = 1

df.fillna(value=0,inplace=True)
df.to_csv('Y:/raid/home/gianlucacarloni/causal_medimg/dataset_padchest/padchest_cleanGC_apr24.csv',index=False)

#%%
# labels_other_or_spurious = [
#     #related to soft tissues
#     'nipple shadow',
#     'pseudonodule', #----differente da nodule
#     'subcutaneous emphysema',
#     'soft tissue mass', #----non pulmonary/lung mass
#     'breast mass',
#     'gynecomastia',
#     'goiter',
#     
#     #related to procedures
#     'surgery lung',
#     'mastectomy',
#     'sternotomy',
#     'surgery breast',
#     'surgery heart',
#     'post radiotherapy changes',
#     #related to anatomy
#     'Chilaiditi sign',
#     'azygoesophageal recess shift',
#     'hiatal hernia',
#     'esophagic dilatation',
#     'hemidiaphragm elevation',
#     'diaphragmatic eventration',
#     'costophrenic angle blunting',
#     'volume loss',
#     'bronchiectasis',
#     'vascular hilar enlargement',
#     'hilar enlargement',
#     'hilar congestion',   
#     'azygos lobe',#anatomical variation, usually not pathological
#     'dextrocardia',    
#     'pulmonary artery enlargement',
#     #related to mediastinum
#     'superior mediastinal enlargement', #---widening
#     'pneumomediastinum',#Air in the mediastinum, not lung
#     #potentially relevant but need further investigation    
#     'end on vessel' #--differential diagnosis with nodules
#     'obesity',
#     'tracheal shift',
#     'flattened diaphragm',
#     'lung vascular paucity',
#     'pulmonary*hypertension',
#     'aortic atheromatosis',
#     'pericardial effusion',   
#     #other and unclear
#     'pneumoperitoneo', #Air in the abdominal cavity, not chest #---pneumoperitoneum; differential diagnosi with Chilaiditi sign
#     'adenopathy',
#     'exclude',    
#     'non axial articular degenerative changes',
#     'air bronchogram',##to difficult to say if normal or abnormal (suggesting consolidation, opacity, fluid...)
#     'respiratory distress',
#     'abnormal foreign body',
#     'prosthesis',#*----
#     'vascular redistribution',
#     #related to previous surgery
#     'surgery humeral',
#     'surgery neck',
#     ##Related to bones
#     'pectum excavatum', #Bone shape abnormalities
#     'pectum carinatum', #Bone shape abnormalities
#     'scoliosis', #Bone shape abnormalities
#     'kyphosis', #Bone shape abnormalities
#     'thoracic cage deformation',
#     'costochondral junction hypertrophy', #joint abnormalities
#     'sternoclavicular junction hypertrophy', #joint abnormalities    
#     'vertebral degenerative changes', # degenerative bone changes
#     'osteoporosis', # degenerative bone changes            
#     'lytic bone lesion', # bone density changes
#     'osteopenia', # bone density changes
#     'blastic bone lesion', # bone density changes    
#     'sclerotic bone lesion',
#     'vertebral anterior compression',    
#     'bone metastasis',
#     'subacromial space narrowing',
#     'cervical rib',
#     'axial hyperostosis'    
#     ]