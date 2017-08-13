'''
this is quicker than the dixon coles method but not as efficient
because the simulation uses a bivariate poisson distribution
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

now = datetime.datetime.now()

def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

def promoted_teams_attack(old_param):
    parameter = 0.53683956207924333 * old_param + 0.049598077388848139
    return parameter

def promoted_teams_defense(old_param):
    parameter = -0.88138228873841129 * old_param + 2.2188585603647821
    return parameter

data = (pd.read_csv(
    '/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/16-17 with xG exponential.csv',
    index_col=0
))

data['HomeMeasure'] = 0.2*data['FTHG'] + 0.8*data['xGH']
data['AwayMeasure'] = 0.2*data['FTAG'] + 0.8*data['xGA']

teams = set(list(data['HomeTeam']))
away_defense_rating = dict.fromkeys(teams, 0)
home_defense_rating = dict.fromkeys(teams, 0)
away_attack_rating = dict.fromkeys(teams, 0)
home_attack_rating = dict.fromkeys(teams, 0)

# calculate initial estimates for home_attack and away_defense ratings
denom = sum(list(data['HomeMeasure']))
for team in teams:
    nume = data.loc[data['AwayTeam'] == team, 'HomeMeasure'].sum()
    away_defense_rating[team] = nume / np.sqrt(denom)
for team in teams:
    nume = data.loc[data['HomeTeam'] == team, 'HomeMeasure'].sum()
    home_attack_rating[team] = nume / np.sqrt(denom)

# calculate initial estimates for away_attack and home_defense ratings
denom = sum(list(data['AwayMeasure']))
for team in teams:
    nume = data.loc[data['AwayTeam'] == team, 'AwayMeasure'].sum()
    away_attack_rating[team] = nume / np.sqrt(denom)
for team in teams:
    nume = data.loc[data['HomeTeam'] == team, 'AwayMeasure'].sum()
    home_defense_rating[team] = nume / np.sqrt(denom)


# get estimates better and better for home_attack and away_defense ratings
for i in range(0, 1000):

    for team in teams:
        away_defense_rating[team] = data.loc[data['AwayTeam'] == team, 'HomeMeasure'].sum() / (
        sum(home_attack_rating.values()) - home_attack_rating[team])

    for team in teams:
        home_attack_rating[team] = data.loc[data['HomeTeam'] == team, 'HomeMeasure'].sum() / (
        sum(away_defense_rating.values()) - away_defense_rating[team])

# get estimates better and better for away_attack and home_defense ratings
for i in range(0, 1000):

    for team in teams:
        home_defense_rating[team] = data.loc[data['HomeTeam'] == team, 'AwayMeasure'].sum() / (
        sum(away_attack_rating.values()) - away_attack_rating[team])

    for team in teams:
        away_attack_rating[team] = data.loc[data['AwayTeam'] == team, 'AwayMeasure'].sum() / (
        sum(home_defense_rating.values()) - home_defense_rating[team])

ratings = pd.DataFrame({'HomeAttack': home_attack_rating, 'HomeDefense': home_defense_rating,
                        'AwayAttack': away_attack_rating, 'AwayDefense': away_defense_rating})

ratings.to_csv('teamratings_16-17.csv')

print(datetime.datetime.now()-now)