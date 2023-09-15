import pandas as pd
import matplotlib.pyplot as plt
import os

# Create the output folder if it doesn't exist
if not os.path.exists("Output/ex2"):
    os.makedirs("Output/ex2")



# A 
# Read the CSV file into a DataFrame
df = pd.read_csv("Data/reviews_sentiment.csv", delimiter=";")

# Filter the rows that have a 1-star rating
one_star_comments = df[df["Star Rating"] == 1]

# Calculate the average word count of the comments in those rows
avg_word_count = one_star_comments["wordcount"].mean()

print(f"The average word count of comments with a 1-star rating is {avg_word_count:.2f}. in the total of {len(one_star_comments)} comments.")

# plot the histogram
one_star_comments["wordcount"].plot.hist(bins=15, title="Word count of 1-star comments", color="red", alpha=0.5)
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.savefig("Output/ex2/wordcount_histogram.png")

# plot bar chart
one_star_comments["wordcount"].value_counts().sort_index().plot.bar(title="Word count of 1-star comments", color="red", alpha=0.5)
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.savefig("Output/ex2/wordcount_barchart.png")


# B split the data into test and train
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (80% train, 20% test), random_state=42 to get the same results
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# C knn classifier
