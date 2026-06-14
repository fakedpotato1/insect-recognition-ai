from ai.utils.math import matmul
from ai.evaluation.accuracy import accuracy_score
from ai.evaluation.precision_recall import classification_report_stats
from ai.evaluation.confusion_matrix import confusion_matrix, print_confusion_matrix

A = [[1, 2], [3, 4]]
B = [[5], [6]]
print("Result of matrix multiplication:\n", matmul(A, B))

true_labels = [0, 1, 2, 0, 1, 2]
pred_lables = [0, 2, 2, 0, 1, 1]

print("\nCurrent accuracy: ", accuracy_score(true_labels, pred_lables))

stats = classification_report_stats(true_labels, pred_lables, num_classes=3)
print("Macro F1-Score: ", stats["macro_f1"])

cm = confusion_matrix(true_labels, pred_lables, num_classes=3)
print_confusion_matrix(cm)