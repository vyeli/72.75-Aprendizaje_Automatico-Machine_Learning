import numpy as np
import matplotlib.pyplot as plt
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
        train_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        accuracy_df= []
        accuracy_rf = []

        for train_percentage in train_percentages:
            accuracy_average_dt = 0
            accuracy_average_rf = 0
            for _ in range(runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= train_percentage/100, random_state=42)
                decision_tree = DecisionTree()
                random_forest = RandomForest()
                decision_tree.fit(X_train, y_train)
                random_forest.fit(X_train, y_train)
                predictions_rf = random_forest.predict(X_test)
                predictions_dt = decision_tree.predict(X_test)
                accuracy_average_dt += self.accuracy(y_test, predictions_dt)
                accuracy_average_rf += self.accuracy(y_test, predictions_rf)

            accuracy_df.append(accuracy_average_dt/runs)
            accuracy_rf.append(accuracy_average_rf/runs)

        plt.plot(train_percentages, accuracy_df, label="Decision tree")
        plt.plot(train_percentages, accuracy_rf, label="Random forest")
        plt.legend()
        plt.xlabel("Training percentage (%)")
        plt.ylabel("Accuracy")
        plt.savefig("Output/ex1/decision_tree/accuracy_over_training_percentage.png")
        plt.clf()  
        


    def precision(self, y_test, y_pred, desired_value):
        tp = np.sum((y_test == desired_value) & (y_pred == desired_value))
        fp = np.sum((y_test != desired_value) & (y_pred == desired_value))
        return tp / (tp + fp)