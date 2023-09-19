import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix

# Create the output folder if it doesn't exist
if not os.path.exists("Output/ex2"):
    os.makedirs("Output/ex2")



# A 
# Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
df = pd.read_csv("Data/reviews_sentiment.csv", delimiter=";")

# Remove the rows with missing values
df = df.dropna()
# Remove the columns of Review Title, Review Text and textSentiment (we don't need use them for this exercise)
df = df.drop(columns=["Review Title", "Review Text", "textSentiment"])
# Replace the values in the "titleSentiment" column with 0 or 1
df["titleSentiment"] = df["titleSentiment"].map({"negative": 0, "positive": 1})

one_star_comments = df[df["Star Rating"] == 1]
avg_word_count = one_star_comments["wordcount"].mean()
print(f"The average word count of comments with a 1-star rating is {avg_word_count:.2f}. in the total of {len(one_star_comments)} comments.")

# plot the histogram
one_star_comments["wordcount"].plot.hist(bins=15, title="Word count of 1-star comments", color="red", alpha=0.5)
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.savefig("Output/ex2/wordcount_histogram.png")

# plot bar chart
fig = plt.figure()
one_star_comments["wordcount"].value_counts().sort_index().plot.bar(title="Word count of 1-star comments", color="red", alpha=0.5)
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.savefig("Output/ex2/wordcount_barchart.png")


# normalize the data
def standar_normalize(data):
    return (data - data.mean()) / data.std()

def feature_scaling(data, a=0, b=1):
    return (data - data.min()) * (b - a) / (data.max() - data.min()) + a

columns_to_normalize = ["wordcount", "titleSentiment", "sentimentValue"]

normalized_df = df.copy()

normalized_df[columns_to_normalize] = normalized_df[columns_to_normalize].apply(lambda x: standar_normalize(x))

print(normalized_df)

# B split the data into test and train, and normalize the data
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (80% train, 20% test), random_state=42 to get the same results
train_set, test_set = train_test_split(normalized_df, test_size=0.2, random_state=42)

# extract the labels from the train and test sets
y_train = train_set.pop("Star Rating")
y_test = test_set.pop("Star Rating")

# Convert the DataFrames to numpy arrays
train_set = train_set.values
test_set = test_set.values

y_train = y_train.values
y_test = y_test.values

print("y_train:", y_train)

# C knn classifier
from knn import KNN

no_weight_knn = KNN(k=5, weighted=False)

# D calculate the accuracy of the classifier and plot the confusion matrix

# Predict the labels of the test set using the no-weighted knn classifier
y_pred = no_weight_knn.predict(test_set, train_set, y_train)

# Calculate the accuracy of the classifier
accuracy = np.sum(y_pred == y_test) / len(y_test)

print(f"The accuracy of the no-weighted knn classifier is {accuracy:.2f}")

# Calculate the confusion matrix for the KNN classifier

labels = np.unique(np.concatenate((y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
def plot_confusion_matrix(cm, labels, title="Confusion matrix", cmap=plt.cm.Blues, weight=False):
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
        
        plt.title(title + (" (weighted)" if weight else "no weighted"))
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig("Output/ex2/confusion_matrix" + ("_weighted" if weight else "_no_weighted") + ".png")

fig = plt.figure()            
plot_confusion_matrix(cm, labels)


# Predict the labels of the test set using the weighted knn classifier
weighted_knn = KNN(k=5, weighted=True)
y_pred = weighted_knn.predict(test_set, train_set, y_train)

print("y_pred:", y_pred)

# Calculate the accuracy of the classifier
accuracy = np.sum(y_pred == y_test) / len(y_test)

print(f"The accuracy of the weighted knn classifier is {accuracy:.2f}")

# Calculate the confusion matrix for the KNN classifier
labels = np.unique(y_test)


cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
fig = plt.figure()
plot_confusion_matrix(cm, labels, weight=True)


