import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    """
    Construct num_classes x num_classes Confusion Matrix
    :param y_true: category index of true label
    :param y_pred: category index of predic label
    :param num_classes: total number of category
    :return: confusion matrix
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        matrix[yt, yp] += 1

    return matrix


def print_confusion_matrix(matrix):
    """
    Print the confusion matrix
    :param matrix: confusion matrix
    :return: None
    """
    print("\n--- Confusion Matrix ---")
    print(np.array(matrix))
    print("-------------------------\n")
