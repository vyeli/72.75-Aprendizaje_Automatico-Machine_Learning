import pandas as pd
from ID3 import DecisionTree
from metrics import MetricsCalculator
from sklearn.model_selection import train_test_split

# Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
df = pd.read_csv("Data/german_credit.csv", delimiter=";")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

decision_tree = DecisionTree()
decision_tree.fit(X_train, y_train)

predictions = decision_tree.predict(X_test)
metrics = MetricsCalculator()
acc = metrics.accuracy(y_test, predictions)