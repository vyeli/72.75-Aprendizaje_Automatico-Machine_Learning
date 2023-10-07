import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from RandomForest import RandomForest

from ID3 import DecisionTree

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

        cm_norm = np.around(cm.astype('float') / np.sum(cm), decimals=4)

        for i, j in np.ndindex(cm.shape):
            count = cm[i, j]
            percent = cm_norm[i, j] * 100
            plt.text(j, i, f"{count:.0f}",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_norm[i, j] > 0.4 else "black")
            plt.text(j, i+0.3, f"{percent:.1f}%",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_norm[i, j] > 0.4 else "black")

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
    
    def plot_accuracy_over_testing_percentage(self, X, y, runs = 3):
        test_percentages = [10 * i for i in range(1, 10)]
        
        test_accuracies_dt_mean = []
        test_accuracies_rf_mean = []

        test_accuracies_dt_std = []
        test_accuracies_rf_std = []

        train_accuracies_dt_mean = []
        train_accuracies_rf_mean = []

        train_accuracies_dt_std = []
        train_accuracies_rf_std = []

        for test_percentage in test_percentages:
            test_accuracies_dt = []
            test_accuracies_rf = []

            train_accuracies_dt = []
            train_accuracies_rf = []

            for _ in range(runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage/100)
                decision_tree = DecisionTree()
                random_forest = RandomForest()
                decision_tree.fit(X_train, y_train)
                random_forest.fit(X_train, y_train)
                predictions_rf = random_forest.predict(X_test)
                predictions_dt = decision_tree.predict(X_test)

                test_accuracies_dt.append(self.accuracy(y_test, predictions_dt))
                test_accuracies_rf.append(self.accuracy(y_test, predictions_rf))

                train_accuracies_dt.append(self.accuracy(y_train, decision_tree.predict(X_train)))
                train_accuracies_rf.append(self.accuracy(y_train, random_forest.predict(X_train)))

            test_accuracies_dt_mean.append(np.mean(test_accuracies_dt))
            test_accuracies_dt_std.append(np.std(test_accuracies_dt))

            train_accuracies_dt_mean.append(np.mean(train_accuracies_dt))
            train_accuracies_dt_std.append(np.std(train_accuracies_dt))

            test_accuracies_rf_mean.append(np.mean(test_accuracies_rf))
            test_accuracies_rf_std.append(np.std(test_accuracies_rf))

            train_accuracies_rf_mean.append(np.mean(train_accuracies_rf))
            train_accuracies_rf_std.append(np.std(train_accuracies_rf))

        plt.errorbar(test_percentages, test_accuracies_dt_mean, yerr=test_accuracies_dt_std, label="Test", marker='o', linestyle="dashed")
        plt.errorbar(test_percentages, train_accuracies_dt_mean, yerr=train_accuracies_dt_std, label="Train", marker='o', linestyle="dashed")
        plt.legend()
        plt.xlabel("Testing percentage (%)")
        plt.ylabel("Precision")
        plt.savefig("Output/ex1/decision_tree/test_train_split.png")
        plt.clf()

        plt.errorbar(test_percentages, test_accuracies_rf_mean, yerr=test_accuracies_rf_std, label="Test", marker='o', linestyle="dashed")
        plt.errorbar(test_percentages, train_accuracies_rf_mean, yerr=train_accuracies_rf_std, label="Train", marker='o', linestyle="dashed")
        plt.legend()
        plt.xlabel("Testing percentage (%)")
        plt.ylabel("Precision")
        plt.savefig("Output/ex1/random_forest/test_train_split.png")
        plt.clf()


    def precision(self, y_test, y_pred, desired_value):
        tp = np.sum((y_test == desired_value) & (y_pred == desired_value))
        fp = np.sum((y_test != desired_value) & (y_pred == desired_value))
        return tp / (tp + fp)