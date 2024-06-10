import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
from torchvision import models, transforms
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle


# 재현성을 위해 시드 설정
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

# 데이터 불러오기
labels_train_val = pd.read_csv('../../../../..//data/train_val_list.txt')
labels_train_val.columns = ['Image_Index']
labels_test = pd.read_csv('../../../../..//data/test_list.txt')
labels_test.columns = ['Image_Index']

disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']

labels_df = pd.read_csv('../../../../..//data/Data_Entry_2017.csv')
labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                     'Patient_Age', 'Patient_Gender', 'View_Position',
                     'Original_Image_Width', 'Original_Image_Height',
                     'Original_Image_Pixel_Spacing_X',
                     'Original_Image_Pixel_Spacing_Y', 'dfd']

# One hot encoding
for disease in tqdm(disease_labels):
    labels_df[disease] = labels_df['Finding_Labels'].map(lambda result: 1 if disease in result else 0)

labels_df['Finding_Labels'] = labels_df['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

num_glob = glob('../../../../../data/images_all/*.png')
img_path = {os.path.basename(x): x for x in num_glob}

labels_df['Paths'] = labels_df['Image_Index'].map(img_path.get)

train_val_df = labels_df[labels_df['Image_Index'].isin(labels_train_val['Image_Index'])]
test_df = labels_df[labels_df['Image_Index'].isin(labels_test['Image_Index'])]

print('train_val size', train_val_df.shape[0])
print('test size', labels_df.shape[0] - train_val_df.shape[0])

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 사전학습된 모델 불러오기
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 마지막 FC 레이어 제거
model.eval()

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    return features.numpy().flatten()

def process_paths(paths, desc):
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_features, path): path for path in paths}
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())
        return results

if not os.path.exists("train_val_features.pkl") or not os.path.exists("test_features.pkl"):
    # 멀티프로세싱을 사용하여 이미지에서 특징 추출
    train_val_paths = train_val_df['Paths'].tolist()
    test_paths = test_df['Paths'].tolist()

    train_val_features = process_paths(train_val_paths, desc="Extracting features train_val_df")
    test_features = process_paths(test_paths, desc="Extracting features test_df")

    # train_val 데이터 저장
    with open('train_val_features.pkl', 'wb') as f:
        pickle.dump({'paths': train_val_paths, 'features': train_val_features}, f)

    # test 데이터 저장
    with open('test_features.pkl', 'wb') as f:
        pickle.dump({'paths': test_paths, 'features': test_features}, f)
        
else:
    # 피클 파일에서 데이터 로드
    with open('train_val_features.pkl', 'rb') as f:
        train_val_data = pickle.load(f)

    with open('test_features.pkl', 'rb') as f:
        test_data = pickle.load(f)
        
# # 특징과 레이블 분리
X_train = np.stack(train_val_data['features'])
y_train = train_val_df[disease_labels].values

X_test = np.stack(test_data['features'])
y_test = test_df[disease_labels].values

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import make_multilabel_classification
from tqdm.notebook import tqdm
import numpy as np

num_classes = 15

# XGBoost 모델 설정
xgb_model = xgb.XGBClassifier(
    # objective='binary:logistic',
    # eval_metric='logloss',
    # use_label_encoder=False
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=num_classes,
    use_label_encoder=False
)

epochs = 20
# OneVsRestClassifier로 멀티레이블 분류기 설정
multilabel_model = OneVsRestClassifier(xgb_model)

# tqdm 콜백 클래스 정의
class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total, desc):
        self.total = total
        self.pbar = tqdm(total=total, desc=desc)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if epoch == self.total - 1:
            self.pbar.close()
        return False

# OneVsRestClassifier로 멀티레이블 분류기 설정
epochs = 20
multilabel_model = OneVsRestClassifier(xgb_model)

# 모델 학습을 위한 맞춤 fit 함수 정의
def custom_fit(model, X_train, y_train, epochs):
    print("Model Fit Start")
    for i in range(y_train.shape[1]):
        print(f"Training for label {i}")
        model.estimator.fit(
            X_train, 
            y_train[:, i], 
            eval_set=[(X_train, y_train[:, i]), (X_test, y_test[:, i])],
            verbose=False, 
            callbacks=[TqdmCallback(epochs, desc=f'Training label {i}')]
        )

# 모델 학습
custom_fit(multilabel_model, X_train, y_train, epochs)

# 예측
y_pred = multilabel_model.predict(X_test)
y_pred_proba = multilabel_model.predict_proba(X_test)  # AUC-ROC 계산을 위해 필요

# 정확도 및 F1 스코어 계산
accuracy = accuracy_score(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')

# AUC-ROC 스코어 계산
roc_auc_micro = roc_auc_score(y_test, y_pred_proba, average='micro')
roc_auc_macro = roc_auc_score(y_test, y_pred_proba, average='macro')

print(f"Test Accuracy: {accuracy}")
# print(f"Test F1 Score (Micro): {f1_micro}")   # 라벨에 상관없이 전체적인 성능 평가 (==accuracy와 동일)
print(f"Test F1 Score (Macro): {f1_macro}")     # 모든 라벨이 유사한 중요도를 가져 단순 라벨들의 f1_score의 산술평균
# print(f"Test ROC AUC Score (Micro): {roc_auc_micro}") 
print(f"Test ROC AUC Score (Macro): {roc_auc_macro}")

# 각 레이블별 평가 결과 출력
for idx in range(y_test.shape[1]):
    f1_label = f1_score(y_test[:, idx], y_pred[:, idx])
    roc_auc_label = roc_auc_score(y_test[:, idx], y_pred_proba[:, idx])
    print(f"Class {idx} - F1 Score: {f1_label}, ROC AUC Score: {roc_auc_label}")
