import numpy as np


def classification_report_stats(y_true, y_pred, num_classes):
    """
    Calculate Precision, Recall, and F1 for a multi-class classifier.

    :param y_true: category index of true label
    :param y_pred: category index of predic label
    :param num_classes: total number of category
    :return: dict, include list of precision, recall, f1 and macro average
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "class_precision": precision_list,
        "class_recall": recall_list,
        "class_f1": f1_list,
        "macro_precision": np.mean(precision_list),
        "macro_recall": np.mean(recall_list),
        "macro_f1": np.mean(f1_list),
    }
