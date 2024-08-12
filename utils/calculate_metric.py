# Cell 1: Import necessary libraries
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, coverage_error, label_ranking_loss, average_precision_score

# Assuming 'targets' and 'ensemble_preds' are already defined and have compatible shapes
# Convert ensemble probabilities to class predictions using a threshold (e.g., 0.5)


threshold = 0.5
# predicts = (ensemble_preds > threshold).to(int)

# Define the disease labels
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']

# Calculate accuracy for each class
def accuracy_each_class(predicts, targets):
    global disease_labels
    accuracy_class_list = []
    
    predicts = (predicts > threshold).to(int)
    print("Accuracy for each class:")
    for i, disease in enumerate(disease_labels):
        accuracy_class = accuracy_score(targets[:, i].cpu(), predicts[:, i])
        print(f'{disease} - Accuracy: {accuracy_class:.4f}')
        accuracy_class_list.append(f"{accuracy_class:.4f}")
        
    return accuracy_class_list


# 사실상 필요없는 건데 일단은 넣어둠
# Calculate overall accuracy
def accurcay_overall(predicts, targets):
    predicts = (predicts > threshold).to(int)
    overall_accuracy = accuracy_score(targets.cpu().reshape(-1), predicts.reshape(-1))
    print(f'Overall Accuracy: {overall_accuracy:.4f}')
    
    # Calculate overall accuracy before reshape (==subset accuracy)
    overall_accuracy_bf_reshape = accuracy_score(targets.cpu(), predicts)
    print(f'Overall Accuracy before reshape: {overall_accuracy_bf_reshape:.4f}')


# Cell 2: Calculate Hamming Loss
def Hamming_loss(predicts, targets):
    predicts = (predicts > threshold).to(int)
    hamming_loss_value = hamming_loss(targets.cpu(), predicts)
    print(f'Hamming Loss: {hamming_loss_value:.4f}')
    
    return f"{hamming_loss_value:.4f}"


# Cell 3: Calculate Ranking Loss
def Ranking_loss(predicts, targets):
    ranking_loss_value = label_ranking_loss(targets.cpu(), predicts)
    print(f'Ranking Loss: {ranking_loss_value:.4f}')
    
    return f"{ranking_loss_value:.4f}"


# Cell 4: Calculate Multilabel Accuracy
def Multilabel_Accuracy(predicts, targets):
    predicts = (predicts > threshold).to(int)
    accuracy_each_class = []

    # Calculate accuracy for each class
    for i, disease in enumerate(disease_labels):
        accuracy_each_class.append(accuracy_score(targets[:, i].cpu(), predicts[:, i]))

    multilabel_accuracy = np.average(accuracy_each_class)
    print(f'Multilabel Accuracy: {multilabel_accuracy:.4f}')
    
    return f"{multilabel_accuracy:.4f}"


# Cell 5: Calculate Multilabel Coverage
def Multilabel_Coverage(predicts, targets):
    multilabel_coverage = coverage_error(targets.cpu(), predicts)
    print(f'Multilabel Coverage: {multilabel_coverage:.4f}')
    
    return f"{multilabel_coverage:.4f}"


# Cell 6: Calculate One Error
def one_error(predicts, targets):
    def custom_one_error(y_true, y_pred):
        n_samples = y_true.shape[0]
        one_error_count = 0
        for i in range(n_samples):
            top_pred_idx = np.argmax(y_pred[i])
            if y_true[i, top_pred_idx] == 0:
                one_error_count += 1
                
        return one_error_count / n_samples                
            
    one_error_value = custom_one_error(targets.cpu().numpy(), predicts.cpu().numpy())
    print(f'One Error: {one_error_value:.4f}')            
            
    return f"{one_error_value:.4f}"
    

# Cell 7: Calculate Subset Accuracy
def Subset_Accuracy(predicts, targets):
    subset_accuracy = accuracy_score(targets.cpu(), predicts, normalize=True)
    print(f'Subset Accuracy: {subset_accuracy:.4f}')
    
    return f"{subset_accuracy:.4f}"


# Cell 8: Calculate Macro F1 Score
def Macro_F1_Score(predicts, targets):
    macro_f1_score = f1_score(targets.cpu(), predicts, average='macro')
    print(f'Macro F1 Score: {macro_f1_score:.4f}')
    
    return f"{macro_f1_score:.4f}"


# Cell 9: Calculate Micro F1 Score
def Micro_F1_Score(predicts, targets):
    micro_f1_score = f1_score(targets.cpu(), predicts, average='micro')
    print(f'Micro F1 Score: {micro_f1_score:.4f}')

    return f"{micro_f1_score:.4f}"


def results_low_score_image(df, predicts, targets, metric):
    predicts_binary = (predicts > threshold).to(int)
    
    df = df.loc[:, ['Image_Index', 'Finding_Labels', 'Paths']].reset_index()
    if metric.lower() == "hamming_loss".lower():
        each_row_score = {i: hamming_loss(targets[i].cpu(), predicts_binary[i]) for i, v in enumerate(targets)}
        
    elif metric.lower() == "Ranking_loss" .lower():
        each_row_score = {i: label_ranking_loss([targets[i].cpu()], [predicts[i]]) for i, v in enumerate(targets)}
        
    elif metric.lower() == "Multilabel_Coverage" .lower():
        each_row_score = {i: coverage_error([targets[i].cpu()], [predicts[i]]) for i, v in enumerate(targets)}
        
    # # Todo
    # elif metric.lower() == "Multilabel_Accuracy" .lower():
    #     each_row_score = {i: label_ranking_loss(targets[i].cpu(), predicts[i]) for i, v in enumerate(targets)}
        
    elif metric.lower() == "one_error".lower():
        each_row_score = {}
        for i in range(len(targets)):
            top_pred_idx = np.argmax(predicts[i])
            if targets[i, top_pred_idx] == 0:
                each_row_score[i] = 1
            else:
                each_row_score[i] = 0
    
    elif metric.lower() == "Subset_Accuracy" .lower():
        each_row_score = {i: accuracy_score(targets[i].cpu(), predicts_binary[i], normalize=True) for i, v in enumerate(targets)}
        
    elif metric.lower() == "Macro_F1_Score" .lower():
        each_row_score = {i: f1_score(targets[i].cpu(), predicts_binary[i], average="macro") for i, v in enumerate(targets)}
        
    elif metric.lower() == "Micro_F1_Score" .lower():
        each_row_score = {i: f1_score(targets[i].cpu(), predicts_binary[i], average="micro") for i, v in enumerate(targets)}
        
    scores = [each_row_score[i] for i in range(len(df))]
    df[f'{metric}_score'] = scores        
        
    sorted_dict = sorted(each_row_score.items(), key= lambda x : -x[1])
    sorted_indices = [i[0] for i in sorted_dict]
    return df.loc[sorted_indices, :]
    