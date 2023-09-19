import numpy as np
import matplotlib.pyplot as plt

class MetricsCalculator:

    def __init__(self):
        pass

    def accuracy(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
    
    def confusion_matrix(self, y_test, y_pred, desired_value):
        cm = np.zeros((2, 2), dtype=int)

        tp = np.sum((y_test == desired_value) & (y_pred == desired_value))
        tn = np.sum((y_test != desired_value) & (y_pred != desired_value))
        fp = np.sum((y_test != desired_value) & (y_pred == desired_value))
        fn = np.sum((y_test == desired_value) & (y_pred != desired_value))

        cm[0, 0] = tp
        cm[0, 1] = fp
        cm[1, 0] = fn
        cm[1, 1] = tn

        return cm
    
    def plot_confusion_matrix(self, cm, algorithm):
        labels = ["Returns ", "Not returns"]
        plt.imshow(cm, cmap="YlOrRd")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.annotate(cm[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', color = "white" if cm[i, j] > 100 else "black")

        if(algorithm == "decision_tree"):
            plt.title("Confusion matrix of decision tree classifier")
        else:
            plt.title("Confusion matrix of random forest classifier")
        plt.xlabel("Predicted result")
        plt.ylabel("Expected result")
        if(algorithm == "decision_tree"):
            plt.savefig("Output/ex1/decision_tree/confusion_matrix.png")
            plt.clf()
        elif(algorithm == "random_forest"):
            plt.savefig("Output/ex1/random_forest/confusion_matrix.png")
            plt.clf()
    
    def precision(self, y_test, y_pred, desired_value):
        tp = np.sum((y_test == desired_value) & (y_pred == desired_value))
        fp = np.sum((y_test != desired_value) & (y_pred == desired_value))
        return tp / (tp + fp)