import numpy as np
import pandas as pd
import sqlite3

database = 'database.sqlite'
conn = sqlite3.connect(database)

# teams = pd.read_sql("""SELECT Match.id, Match.home_team_api_id as home_team_id, home_buildup_play_speed, home_buildup_play_dribbling, home_build_play_passing, home_buildup_chance_creation_passing, home_buildup_chance_creation_crossing, home_buildup_chance_creation_shooting, home_buildup_defence_pressure, home_defence_aggression, home_defence_team_width
#                     FROM Match JOIN (
#                         SELECT
#                     team_api_id as home_team_api_id,
#                     buildUpPlaySpeed as home_buildup_play_speed,
#                     buildUpPlayDribbling as home_buildup_play_dribbling,
#                     buildUpPlayPassing as home_build_play_passing,
#                     chanceCreationPassing as home_buildup_chance_creation_passing,
#                     chanceCreationCrossing as home_buildup_chance_creation_crossing,
#                     chanceCreationShooting as home_buildup_chance_creation_shooting,
#                     defencePressure as home_buildup_defence_pressure,
#                     defenceAggression as home_defence_aggression,
#                     defenceTeamWidth as home_defence_team_width
#                     FROM Team_Attributes
#                     WHERE date LIKE '2015%') as team_attr
#                     ON Match.home_team_api_id = team_attr.home_team_api_id;""", conn).dropna()

matches = pd.read_sql("""SELECT id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal FROM Match WHERE season LIKE '2015/2016'""", conn)

teams = pd.read_sql("""SELECT team_api_id, buildUpPlaySpeed, buildUpPlayDribbling, buildUpPlayPassing FROM Team_attributes WHERE date LIKE '2015%';""", conn)
# teams.drop(['id', 'team_fifa_api_id', 'date'], axis=1, inplace=True)

home_teams = teams.add_prefix('home_')
away_teams = teams.add_prefix('away_')

matches_teams = pd.merge(matches, home_teams, on='home_team_api_id')
matches_teams = pd.merge(matches_teams, away_teams, on='away_team_api_id')

matches_teams = matches_teams.drop(columns=matches_teams.select_dtypes(exclude='number').columns.tolist())

print(matches_teams.head())

matches_teams.to_csv('teams.csv', index=False)
