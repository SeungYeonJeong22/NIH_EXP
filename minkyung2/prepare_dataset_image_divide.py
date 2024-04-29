#%%
import os
import shutil

#%%
trainval_list = 'train_val_list.txt'
test_list = 'test_list.txt'

#%%
trainval_dir = 'images/trainval'
test_dir = 'images/test'

os.makedirs(trainval_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#%%
with open(trainval_list, 'r') as file :
    trainval_images = file.read().splitlines()
    for image_name in trainval_images :
        origin_path = os.path.join('nih_multi_data', image_name)
        new_path = os.path.join(trainval_dir, image_name)
        shutil.copy(origin_path, new_path)

#%%
with open(test_list, 'r') as file :
    test_images = file.read().splitlines()
    for image_name in test_images :
        origin_path2 = os.path.join('nih_multi_data', image_name)
        new_path2 = os.path.join(test_dir, image_name)
        shutil.copy(origin_path2, new_path2)

print("이미지 파일 복사 완료")
# %%
