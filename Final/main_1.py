import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os

# Database connection
database = 'database.sqlite'
conn = sqlite3.connect(database)

detailed_matches = pd.read_sql("""SELECT Match.id,
                                        match_api_id,
                                        Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season, 
                                        stage, 
                                        date,
                                        HT.team_long_name AS  home_team,
                                        AT.team_long_name AS away_team,
                                        home_team_goal, 
                                        away_team_goal,
                                        B365H as home_win_odds,
                                        B365D as draw_odds,
                                        B365A as away_win_odds                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                ORDER by date;""", conn).dropna()

print(detailed_matches.head())

# write pandas dataframe to csv file
# detailed_matches.to_csv('matches.csv', index=False)

# countries = pd.read_sql("""SELECT *
#                         FROM Country;""", conn)
# print(countries)

# try to predict match outcome using gradient boosting
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# create a copy of the dataframe
matches = detailed_matches.copy()

# drop columns that are not needed
matches.drop(['id', 'country_name', 'league_name', 'date'], axis=1, inplace=True)

# drop rows with missing values
matches.dropna(inplace=True)

# create a new column with the match outcome
# 1 - home team wins
# 0 - draw
# -1 - away team wins
matches['match_outcome'] = np.where(matches['home_team_goal'] > matches['away_team_goal'], 1, 0)
matches['match_outcome'] = np.where(matches['home_team_goal'] < matches['away_team_goal'], -1, matches['match_outcome'])

# drop columns that are not needed
matches.drop(['home_team_goal', 'away_team_goal'], axis=1, inplace=True)

# encode categorical features
le = LabelEncoder()

matches['season'] = le.fit_transform(matches['season'])
matches['home_team'] = le.fit_transform(matches['home_team'])
matches['away_team'] = le.fit_transform(matches['away_team'])

# split data into training and testing sets
X = matches.drop(['match_outcome'], axis=1)
y = matches['match_outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
gb_clf.fit(X_train, y_train)

# make predictions
y_pred = gb_clf.predict(X_test)

# evaluate model performance
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Classification report: \n', classification_report(y_test, y_pred))
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))

# plot confusion matrix
ConfusionMatrixDisplay.from_estimator(gb_clf, X_test, y_test, cmap=plt.cm.Blues)
plt.show()