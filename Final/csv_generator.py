import numpy as np
import pandas as pd
import sqlite3

def get_seasons(date_array):
    seasons = []
    for date in date_array:
        year = date[0:4]
        match year:
            case '2010':
                seasons.append('2009/2010')
            case '2011':
                seasons.append('2010/2011')
            case '2012':
                seasons.append('2011/2012')
            case '2013':
                seasons.append('2013/2014')
            case '2014':
                seasons.append('2014/2015')
            case '2015':
                seasons.append('2015/2016')
    return seasons


database = 'database.sqlite'
conn = sqlite3.connect(database)

# attributes = ['buildUpPlaySpeed', 'buildUpPlayPassing']
matches = pd.read_sql("""SELECT * FROM Match""", conn)
matches = matches[['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']]
matches.dropna(inplace=True)
# for attribute in attributes:
#     teams[attribute] = (teams[attribute] - teams[attribute].min()) / (teams[attribute].max() - teams[attribute].min())

# home_teams = teams.add_prefix('home_')
# home_teams.rename(columns={'home_season': 'season'}, inplace=True)

# away_teams = teams.add_prefix('away_')
# away_teams.rename(columns={'away_season': 'season'}, inplace=True)

# matches_teams = pd.merge(matches, home_teams, on=['season', 'home_team_api_id'])
# matches_teams = pd.merge(matches_teams, away_teams, on=['season', 'away_team_api_id'])

# condition = matches_teams['home_' + attribute] == matches_teams['away_' + attribute]
# matches_teams = matches_teams.loc[~condition]

# matches_teams.drop(['home_id', 'away_id'], axis=1, inplace=True)

matches.to_csv('matches_bets.csv', index=False)

####################################################### PLAYERS ##############################################################

# matches = pd.read_sql("""SELECT home_team_api_id, away_team_api_id, home_team_goal, away_team_goal,
#                       home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
#                       home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11,
#                       away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
#                       away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11
#                       FROM Match WHERE season LIKE '2015/2016'""", conn).dropna()

# players = pd.read_sql("""SELECT player_api_id, date, overall_rating FROM Player_Attributes""", conn)

# players = players.sort_values(by=['player_api_id', 'date'], ascending=[True, False])
# players = players.loc[players.groupby('player_api_id')['date'].idxmax()]
# players = players[players['date'] > '2015-05-31'].dropna()

# print(f'Hay {len(players)} jugadores')

# matches_players = matches.copy()[['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal']]

# for i in range(1, 12):
#     for state in ['home_', 'away_']:
#         player_match = pd.merge(matches, players, left_on=state + 'player_' + str(i), right_on='player_api_id')
#         matches_players[state + 'player_' + str(i) + '_rating'] = player_match['overall_rating']

# matches_players['home_rating'] = (matches_players['home_player_1_rating'] + matches_players['home_player_2_rating'] + matches_players['home_player_3_rating'] + matches_players['home_player_4_rating'] + matches_players['home_player_5_rating'] + matches_players['home_player_6_rating'] + matches_players['home_player_7_rating'] + matches_players['home_player_8_rating'] + matches_players['home_player_9_rating'] + matches_players['home_player_10_rating'] + matches_players['home_player_11_rating']) / 11
# matches_players['away_rating'] = (matches_players['away_player_1_rating'] + matches_players['away_player_2_rating'] + matches_players['away_player_3_rating'] + matches_players['away_player_4_rating'] + matches_players['away_player_5_rating'] + matches_players['away_player_6_rating'] + matches_players['away_player_7_rating'] + matches_players['away_player_8_rating'] + matches_players['away_player_9_rating'] + matches_players['away_player_10_rating'] + matches_players['away_player_11_rating']) / 11

# matches_players = matches_players[['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'home_rating', 'away_rating']]
# # matches_players = matches_players[(matches_players['home_rating'] - matches_players['away_rating']).abs() >= 3]

# print(matches_players.head())
# print(f'Hay {len(matches_players)} registros')
# matches_players.to_csv('teams.csv', index=False)
