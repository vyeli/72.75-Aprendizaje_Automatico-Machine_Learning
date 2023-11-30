import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

# load data
conn = sqlite3.connect('./Data/database.sqlite')
with sqlite3.connect('./Data/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    player = pd.read_sql_query("SELECT * from Player", con)
    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes", con)
    team_attributes = pd.read_sql_query("SELECT * from Team_Attributes", con)

# preprarate the data
# select the top 5 leagues

selected_countries = ['England','France','Germany','Italy','Spain']
countries = countries[countries.name.isin(selected_countries)]
leagues = leagues[leagues.country_id.isin(countries.id)]

# create a new column for the match table to indicate the winner (1 for home team, 2 for away team, 0 for draw)
matches["winner"] = np.where(matches["home_team_goal"] > matches["away_team_goal"], 1, np.where(matches["home_team_goal"] < matches["away_team_goal"], 2, 0))
