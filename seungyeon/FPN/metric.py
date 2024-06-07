from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def compute_metrics(predictions, targets, threshold=0.5):
    # 시그모이드 함수를 적용하여 확률을 계산합니다.
    # predictions = softmax(predictions)
    
    # 확률을 이진 값으로 변환합니다.
    # predictions = (predictions > threshold).astype(np.int16)
    predictions = np.argmax(predictions, axis=1)
    
    # 각 메트릭을 계산합니다.
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='micro')
    recall = recall_score(targets, predictions, average='micro')
    f1 = f1_score(targets, predictions, average='micro')
    
    return accuracy, precision, recall, f1