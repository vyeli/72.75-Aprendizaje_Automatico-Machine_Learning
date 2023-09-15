import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        true_index = np.where(labels == true_label)[0][0]
        pred_index = np.where(labels == pred_label)[0][0]
        cm[true_index, pred_index] += 1
    
    return cm

