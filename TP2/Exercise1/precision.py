import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from ID3 import DecisionTree
from RandomForest import RandomForest
from metrics import MetricsCalculator
from sklearn.model_selection import train_test_split

# Create the output folder if it doesn't exist
if not os.path.exists("Output/ex1"):
    os.makedirs("Output/ex1")

# Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
df = pd.read_csv("Data/german_credit.csv", delimiter=",")
# split the data into features and target variable where the target column is Creditability
X = df.drop("Creditability", axis=1).to_numpy()
y = df["Creditability"].to_numpy()

heights = [2+i for i in range(20)]

train_accuracies_mean = []
train_accuracies_std = []

test_accuracies_mean = []
test_accuracies_std = []

rf_train_accuracies_mean = []
rf_train_accuracies_std = []

rf_test_accuracies_mean = []
rf_test_accuracies_std = []

nodes_mean = []
nodes_std = []

rf_nodes_mean = []
rf_nodes_std = []

for height in heights:
    train_accuracies = []
    test_accuracies = []

    rf_train_accuracies = []
    rf_test_accuracies = []

    nodes = []
    rf_nodes = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        #decision tree
        decision_tree = DecisionTree(max_depth=height)
        decision_tree.fit(X_train, y_train)
        predictions = decision_tree.predict(X_test)
        nodes.append(decision_tree.get_Total_nodes())

        random_forest = RandomForest(max_depth=height)
        random_forest.fit(X_train, y_train)
        predictions_rf = random_forest.predict(X_test)
        rf_nodes.append(random_forest.get_mean_nodes_per_tree())

        metrics = MetricsCalculator()
        train_accuracies.append(metrics.accuracy(y_train, decision_tree.predict(X_train)))
        test_accuracies.append(metrics.accuracy(y_test, predictions))
        rf_train_accuracies.append(metrics.accuracy(y_train, random_forest.predict(X_train)))
        rf_test_accuracies.append(metrics.accuracy(y_test, predictions_rf))
    
    train_accuracies_mean.append(np.mean(train_accuracies))
    train_accuracies_std.append(np.std(train_accuracies))

    test_accuracies_mean.append(np.mean(test_accuracies))
    test_accuracies_std.append(np.std(test_accuracies))

    nodes_mean.append(np.mean(nodes))
    nodes_std.append(np.std(nodes))

    rf_train_accuracies_mean.append(np.mean(rf_train_accuracies))
    rf_train_accuracies_std.append(np.std(rf_train_accuracies))

    rf_test_accuracies_mean.append(np.mean(rf_test_accuracies))
    rf_test_accuracies_std.append(np.std(rf_test_accuracies))

    rf_nodes_mean.append(np.mean(rf_nodes))
    rf_nodes_std.append(np.std(rf_nodes))

fig, ax = plt.subplots()

sc = ax.scatter(nodes_mean, heights, c=train_accuracies_mean, cmap=plt.cm.plasma, vmin=np.min(train_accuracies_mean), vmax=np.max(train_accuracies_mean))
plt.colorbar(sc, label="Train accuracy")

ax.errorbar(nodes_mean, heights, xerr=nodes_std, fmt='none', color="gray", alpha=0.5)

plt.ylabel("Max tree height")
plt.xlabel("Nodes")
plt.savefig("Output/ex1/precision_vs_train_nodes.png")
plt.clf()

fig, ax = plt.subplots()
sc = ax.scatter(nodes_mean, heights, c=test_accuracies_mean, cmap=plt.cm.plasma, vmin=np.min(test_accuracies_mean), vmax=np.max(test_accuracies_mean))
plt.colorbar(sc, label="Test accuracy")

ax.errorbar(nodes_mean, heights, xerr=nodes_std, fmt='none', color="gray", alpha=0.5)

plt.ylabel("Max tree height")
plt.xlabel("Nodes")
plt.savefig("Output/ex1/precision_vs_test_nodes.png")
plt.clf()

plt.plot(nodes_mean, train_accuracies_mean, marker='o', linestyle="-", label="Train")
plt.plot(nodes_mean, test_accuracies_mean, marker='o', linestyle="-", label="Test")
plt.ylabel("Precision")
plt.xlabel("Nodes")
plt.legend()
plt.savefig("Output/ex1/decision_tree/precision_vs_nodes.png")
plt.clf()

plt.plot(rf_nodes_mean, rf_train_accuracies_mean, marker='o', linestyle="-", label="Train")
plt.plot(rf_nodes_mean, rf_test_accuracies_mean, marker='o', linestyle="-", label="Test")
plt.ylabel("Precision")
plt.xlabel("Nodes (mean per tree)")
plt.legend()
plt.savefig("Output/ex1/random_forest/precision_vs_nodes.png")