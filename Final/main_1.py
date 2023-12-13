import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# create a copy of the dataframe
matches = pd.read_csv('teams_players.csv')

# drop rows with missing values
matches.dropna(inplace=True)

# create a new column with the match outcome
# 0 - home team wins
# 1 - draw
# 2 - away team wins
matches['match_outcome'] = np.where(matches['home_team_goal'] > matches['away_team_goal'], 0, 2)
matches['match_outcome'] = np.where(matches['home_team_goal'] == matches['away_team_goal'], 1, matches['match_outcome'])

# drop columns that are not needed
matches.drop(['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal'], axis=1, inplace=True)
# matches.drop(['id', 'country_name', 'league_name', 'date', 'home_team', 'away_team', 'home_team_goal', 'away_team_goal'], axis=1, inplace=True)

# encode categorical features
le = LabelEncoder()
# matches['match_outcome'] = le.fit_transform(matches['match_outcome'])

# Saco los que coincidan en la característica
# match_outcome = matches['match_outcome']
# matches = matches.select_dtypes(exclude=np.number)
# matches['match_outcome'] = match_outcome
attributes = matches.select_dtypes(exclude=np.number)

for attribute in attributes.columns.to_list():
    print(attribute)
    matches[attribute] = le.fit_transform(matches[attribute])
    # matches['home_' + attribute] = le.fit_transform(matches['home_' + attribute])
    # matches['away_' + attribute] = le.fit_transform(matches['away_' + attribute])

matches = matches[abs(matches['home_rating'] - matches['away_rating']) > 4]
print(f'Hay {len(matches)} muestras')

# matches['season'] = le.fit_transform(matches['season'])
# matches['home_team'] = le.fit_transform(matches['home_team'])
# matches['away_team'] = le.fit_transform(matches['away_team'])

# split data into training and testing sets
X = matches.drop(['match_outcome'], axis=1)
y = matches['match_outcome']
mean_cm = np.zeros((3, 3))

iterations = 5
method = '__'

for i in range(iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # sss = StratifiedShuffleSplit(test_size=0.1)

    # for train_index, test_index in sss.split(X, y):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    gb_clf = XGBClassifier(n_estimators=500, learning_rate=0.001, max_depth=3, objective='multi:softmax')
    gb_clf.fit(X_train, y_train)

    # make predictions
    y_pred = gb_clf.predict(X_test)

    # evaluate model performance
    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))
    print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))

    # plot confusion matrix
    mean_cm += confusion_matrix(y_test, y_pred)

    not_exists = False

    if not os.path.exists('predictions.csv'):
        not_exists = True

    # with open('predictions.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     if not_exists:
    #         writer.writerow(['method', 'accuracy', 'recall', 'f1-score'])
    #     writer.writerow([method, accuracy_score(y_test, y_pred), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')])

mean_cm /= iterations
sns.heatmap(mean_cm, annot=True, fmt='.1f', xticklabels=['Home', 'Draw', 'Away'], yticklabels=['Home', 'Draw', 'Away'])

if not os.path.exists('output'):
    os.makedirs('output')
# plt.savefig(f'output/{method}_confusion_matrix.png')
# plot confusion matrix
# ConfusionMatrixDisplay.from_estimator(gb_clf, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()