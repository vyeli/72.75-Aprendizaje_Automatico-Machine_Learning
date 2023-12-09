import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
import xgboost as xgb

# Database connection
database = 'database.sqlite'
conn = sqlite3.connect(database)

# detailed_matches = pd.read_sql("""SELECT Match.id,
#                                         match_api_id,
#                                         Country.name AS country_name, 
#                                         League.name AS league_name, 
#                                         season, 
#                                         stage, 
#                                         date,
#                                         HT.team_long_name AS  home_team,
#                                         AT.team_long_name AS away_team,
#                                         home_team_goal, 
#                                         away_team_goal,
#                                         B365H as home_win_odds,
#                                         B365D as draw_odds,
#                                         B365A as away_win_odds                                        
#                                 FROM Match
#                                 JOIN Country on Country.id = Match.country_id
#                                 JOIN League on League.id = Match.league_id
#                                 LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
#                                 LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
#                                 ORDER by date;""", conn).dropna()

# detailed_matches = pd.read_sql("""SELECT Match.id,
#                                         Country.name AS country_name, 
#                                         League.name AS league_name, 
#                                         season,
#                                         date,
#                                         HT.team_long_name AS  home_team,
#                                         AT.team_long_name AS away_team,
#                                         home_team_goal, 
#                                         away_team_goal,
#                                         B365H as home_win_odds_1,
#                                         B365D as draw_odds_1,
#                                         B365A as away_win_odds_1                                   
#                                 FROM Match
#                                 JOIN Country on Country.id = Match.country_id
#                                 JOIN League on League.id = Match.league_id
#                                 LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
#                                 LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
#                                 ORDER by date;""", conn).dropna()

# players = pd.read_sql("""SELECT * from player_attributes;""", conn).dropna()
# print(players.head())

# # output to csv
# detailed_matches.to_csv('matches_bets.csv', index=False)

# countries = pd.read_sql("""SELECT *
#                         FROM Country;""", conn)
# print(countries)

# try to predict match outcome using gradient boosting
# import libraries
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# create a copy of the dataframe
matches = pd.read_csv('teams.csv')

# drop rows with missing values
matches.dropna(inplace=True)

# create a new column with the match outcome
# 0 - home team wins
# 1 - draw
# 2 - away team wins
matches['match_outcome'] = np.where(matches['home_team_goal'] > matches['away_team_goal'], 0, 1)
matches['match_outcome'] = np.where(matches['home_team_goal'] < matches['away_team_goal'], 2, matches['match_outcome'])

# drop columns that are not needed
matches.drop(['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal'], axis=1, inplace=True)

# encode categorical features
le = LabelEncoder()
# matches['match_outcome'] = le.fit_transform(matches['match_outcome'])

# Saco los que coincidan en la característica
attributes = []

for attribute in attributes:
    matches['home_' + attribute] = le.fit_transform(matches['home_' + attribute])
    matches['away_' + attribute] = le.fit_transform(matches['away_' + attribute])

print(matches.head())

# matches['season'] = le.fit_transform(matches['season'])
# matches['home_team'] = le.fit_transform(matches['home_team'])
# matches['away_team'] = le.fit_transform(matches['away_team'])

# split data into training and testing sets
X = matches.drop(['match_outcome'], axis=1)
y = matches['match_outcome']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sss = StratifiedShuffleSplit(test_size=0.1)

for train_index, test_index in sss.split(X, y):
    # print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

gb_clf = XGBClassifier(n_estimators=500, learning_rate=0.001, max_depth=5, objective='multi:softmax')
gb_clf.fit(X_train, y_train)

# make predictions
y_pred = gb_clf.predict(X_test)

# evaluate model performance
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Classification report: \n', classification_report(y_test, y_pred))
print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))

# plot confusion matrix
# ConfusionMatrixDisplay.from_estimator(gb_clf, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()