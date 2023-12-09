import numpy as np
import pandas as pd
import sqlite3

database = '../database.sqlite'
conn = sqlite3.connect(database)

matches = pd.read_sql("""SELECT * FROM Match""", conn)
matches = matches[['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'date', 'country_id', 'season']]

teams = pd.read_sql("""SELECT * FROM Team""", conn)
teams = teams[['team_api_id', 'team_long_name']]

countries = pd.read_sql("""SELECT * FROM Country""", conn)

home_teams = teams.add_prefix('home_')
away_teams = teams.add_prefix('away_')

matches_teams  = pd.merge(matches, home_teams, left_on='home_team_api_id', right_on='home_team_api_id')
matches_teams = pd.merge(matches_teams, away_teams, left_on='away_team_api_id', right_on='away_team_api_id')
matches_teams = pd.merge(matches_teams, countries, left_on='country_id', right_on='id')

matches_teams.drop(['home_team_api_id', 'away_team_api_id', 'country_id', 'id'], axis=1, inplace=True)
matches_teams = matches_teams.rename(columns={'home_team_goal': 'FTHG', 'away_team_goal': 'FTAG', 'home_team_long_name': 'HomeTeam', 'away_team_long_name': 'AwayTeam', 'date': 'Date'})

# Cambiar acá el país analizado
matches_teams = matches_teams[matches_teams["name"] == "England"]
matches_teams = matches_teams.drop(["name"], axis=1)

seasons = ['2008/2009', '2009/2010', '2010/2011', '2011/2012', '2012/2013', '2013/2014', '2014/2015', '2015/2016']

for season in seasons:
    season_matches = matches_teams[matches_teams['season'] == season]
    season_matches.drop(['season'], axis=1)
    season_matches = season_matches.sort_values(by=['Date'])
    season_matches['Date'] = pd.to_datetime(season_matches['Date']).dt.strftime('%d-%m-%Y')
    season_matches['FTR'] = np.where(season_matches['FTHG'] > season_matches['FTAG'], 'H', np.where(season_matches['FTHG'] < season_matches['FTAG'], 'A', 'D'))
    pd.DataFrame(season_matches).to_csv(f'input/{season[0:4]}-{season[7:9]}.csv', index=False)

print(f'Hay {len(matches_teams)} registros')
print(matches_teams.head())