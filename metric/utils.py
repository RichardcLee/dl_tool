# richard lee
# 2020.10.22

import numpy as np


# 计算混淆矩阵 cal confusion matrix, pixel-wise or instance-wise
def gen_confusion_matrix_binary_class(predictions: np.ndarray, labels: np.ndarray, threshold=0.5):
    FP = np.float(np.sum((predictions >= threshold) & (labels < threshold)))
    FN = np.float(np.sum((predictions < threshold) & (labels >= threshold)))
    TP = np.float(np.sum((predictions >= threshold) & (labels >= threshold)))
    TN = np.float(np.sum((predictions < threshold) & (labels < threshold)))

    return FP, FN, TP, TN


# confusion maxtrix
def gen_confusion_matrix_multi_class(predict: np.ndarray, label: np.ndarray, num_class=2):
    """Computes scores:
        FP = False Positives
        FN = False Negatives
        TP = True Positives
        TN = True Negatives
        return: FP, FN, TP, TN
    """
    # remove classes from unlabeled pixels in gt image and predict
    mask = (label >= 0) & (label < num_class)
    label = num_class * label[mask] + predict[mask]
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix
