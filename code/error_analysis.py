from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def error_analysis(pred):
    logits, labels = pred
    pred_labels = np.argmax(logits, axis=1)
    matrix = confusion_matrix(labels, pred_labels, normalize='true')
    print("")
    plot_confusion_matrix(matrix)

def plot_confusion_matrix(confusion_matrix):
    num_classes = confusion_matrix.shape[0]

    plt.figure(figsize=(12, 8))
    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(1, num_classes + 1), rotation=45)
    plt.yticks(tick_marks, range(1, num_classes + 1))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig('./images/confusion_matrix_heatmap.png')