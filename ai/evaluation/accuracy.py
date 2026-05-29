import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy
    :param y_true: true label list
    :param y_pred: predict table list
    :return: accuracy score
    """
    return np.mean(np.array(y_true) == np.array(y_pred))