import fastai
from fastai.vision.all import *
from tqdm import tqdm
from glob import glob
import os
import numpy as np
import pandas as pd

from utils.init_csv import CSV_Record

SEED = 85
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

labels_train_val = pd.read_csv('../../../../..//data/train_val_list.txt')
labels_train_val.columns = ['Image_Index']
labels_test = pd.read_csv('../../../../..//data/test_list.txt')
labels_test.columns = ['Image_Index']

disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']
# NIH Dataset Labels CSV File 
labels_df = pd.read_csv('../../../../..//data/Data_Entry_2017.csv')
labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']
# One hot encoding
for diseases in tqdm(disease_labels): 
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0)

labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

num_glob = glob('../../../../../data/images_all/*.png')
img_path = {os.path.basename(x): x for x in num_glob}

labels_df['Paths'] = labels_df['Image_Index'].map(img_path.get)
unique_patients = np.unique(labels_df['Patient_ID'])

train_val_df = labels_df[labels_df['Image_Index'].isin(labels_train_val['Image_Index'])]
test_df = labels_df[labels_df['Image_Index'].isin(labels_test['Image_Index'])]

print('train_val size', train_val_df.shape[0])
print('test size', test_df.shape[0])

item_transforms = [
    Resize((224, 224)),
]

batch_transforms = [
    Flip(),
    Rotate(),
    Normalize.from_stats(*imagenet_stats),
]

def get_x(row):
    return row['Paths']

def get_y(row):
    labels = row[disease_labels].tolist()
    return labels


csv_record = CSV_Record(model_name="syn")


dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=disease_labels)),
    get_x=get_x,
    get_y=get_y,
    item_tfms=item_transforms,
    batch_tfms=batch_transforms
)

dls = dblock.dataloaders(train_val_df, bs=32)

arch = 'coatnet_2_rw_224.sw_in12k_ft_in1k'

cbs = [
    SaveModelCallback(monitor='train_loss', min_delta=0.001, with_opt=True),
    EarlyStoppingCallback(monitor='train_loss', min_delta=0.001, patience=5),
    ShowGraphCallback()
]

learn = vision_learner(dls, arch, metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()], cbs=cbs, wd=0.001)

learn.model = torch.nn.DataParallel(learn.model)
lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
print('initial learning rate=', lrs.valley)
print()

learn.fine_tune(freeze_epochs=3, epochs=20, base_lr=lrs.valley)
learn.save('coatnet-70-10-20-split')

# Test set evaluation
test_dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=disease_labels)),
    get_x=get_x,
    get_y=get_y,
    item_tfms=item_transforms,
    batch_tfms=batch_transforms
)

test_dls = test_dblock.dataloaders(test_df, bs=32, shuffle=False)

# Evaluate on test set
test_results = learn.validate(dl=test_dls, metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()])
print('Test set evaluation results:', test_results)

