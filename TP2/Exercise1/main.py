import pandas as pd
from ID3 import DecisionTree
from RandomForest import RandomForest
from sklearn import datasets
from metrics import MetricsCalculator
from sklearn.model_selection import train_test_split

# Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
df = pd.read_csv("Data/german_credit.csv", delimiter=",")
# split the data into features and target variable where the target column is Creditability
X = df.drop("Creditability", axis=1).to_numpy()
y = df["Creditability"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#decision tree
decision_tree = DecisionTree()
decision_tree.fit(X_train, y_train)
predictions = decision_tree.predict(X_test)
metrics = MetricsCalculator()
cm = metrics.confusion_matrix(y_test, predictions, 1)
metrics.plot_confusion_matrix(cm, "decision_tree")

#random forest
random_forest = RandomForest()
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)
cm = metrics.confusion_matrix(y_test, predictions, 1)
metrics.plot_confusion_matrix(cm, "random_forest")

