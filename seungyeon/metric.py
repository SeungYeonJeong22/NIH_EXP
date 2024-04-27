from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def compute_metrics(predictions, targets, threshold=0.5):
    predictions = (predictions > threshold).astype(int)
    
    # print('Targets \t predictions')
    # for t, p in zip(targets, predictions):
    #     print(t, '\t', p)
    
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='micro')
    recall = recall_score(targets, predictions, average='micro')
    f1 = f1_score(targets, predictions, average='micro')
    
    return accuracy, precision, recall, f1
