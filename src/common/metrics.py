"""Basic metrics: accuracy, macro F1, mAP placeholder."""
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def map_per_label(y_true, y_scores):
    # Placeholder that returns zeros; implement with sklearn/torchmetrics when needed
    return np.zeros(y_scores.shape[1])
