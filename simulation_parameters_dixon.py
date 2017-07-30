import numpy as np
import pandas as pd
from math import factorial as fact
from scipy.optimize import minimize
import datetime


def create_team_map(teams):
    unique_items = np.unique(teams)
    return {item: id_ for id_, item in enumerate(unique_items, start=0)}


def tau(x, y, lambdaa, mu, rho):
    if x == 0 and y == 0:
        result = 1 - (lambdaa*mu*rho)
    elif x == 0 and y == 1:
        result = 1 + (lambdaa*rho)
    elif x == 1 and y == 0:
        result = 1 + (mu*rho)
    elif x == 1 and y == 1:
        result = 1 - rho
    else:
        result = 1
    return result


def log_likelihood_function(theta, data):
    rho=0.15
    result=0
    most_recent_match = max(data['Date'])
    for index, row in data.iterrows():
        days = (most_recent_match - row['Date']).days
        result += np.exp(-0.002*days) * (np.log(tau(row['HomeMeasure'], row['AwayMeasure'], theta[row['HomeTeamNumber']]*theta[row['AwayTeamNumber']+60], theta[row['AwayTeamNumber']+40]*theta[row['HomeTeamNumber']+20], rho))\
                  - (theta[row['HomeTeamNumber']]*theta[row['AwayTeamNumber']+60]) + row['HomeMeasure']*np.log(theta[row['HomeTeamNumber']]*theta[row['AwayTeamNumber']+60])\
                  - np.log(fact(np.round(row['HomeMeasure'],0))) - (theta[row['HomeTeamNumber']+20]*theta[row['AwayTeamNumber']+40]) + row['AwayMeasure']*np.log(theta[row['HomeTeamNumber']+20]*theta[row['AwayTeamNumber']+40])\
                  - np.log(fact(np.round(row['AwayMeasure'],0))))
    return result


def read_in_data():
    data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/16-17 with xG exponential.csv')

    team_map = create_team_map(data['HomeTeam'])

    data['HomeTeamNumber'] = data['HomeTeam']
    data['AwayTeamNumber'] = data['AwayTeam']
    data['HomeTeamNumber'].replace(team_map, inplace=True)
    data['AwayTeamNumber'].replace(team_map, inplace=True)
    data['HomeMeasure'] = 0.2 * data['FTHG'] + 0.8 * data['xGH']
    data['AwayMeasure'] = 0.2 * data['FTAG'] + 0.8 * data['xGA']

    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')

    return data, team_map


def initial_parameter_estimates(data):

    home_attack_rating = [0] * 20
    home_defense_rating = [0] * 20
    away_attack_rating = [0] * 20
    away_defense_rating = [0] * 20

    # calculate initial estimates for home_attack and away_defense ratings
    denom = sum(list(data['HomeMeasure']))
    for team in team_map.values():
        nume = data.loc[data['AwayTeamNumber'] == team, 'HomeMeasure'].sum()
        away_defense_rating[team] = nume / np.sqrt(denom)
    for team in team_map.values():
        nume = data.loc[data['HomeTeamNumber'] == team, 'HomeMeasure'].sum()
        home_attack_rating[team] = nume / np.sqrt(denom)

    # calculate initial estimates for away_attack and home_defense ratings
    denom = sum(list(data['AwayMeasure']))
    for team in team_map.values():
        nume = data.loc[data['AwayTeamNumber'] == team, 'AwayMeasure'].sum()
        away_attack_rating[team] = nume / np.sqrt(denom)
    for team in team_map.values():
        nume = data.loc[data['HomeTeamNumber'] == team, 'AwayMeasure'].sum()
        home_defense_rating[team] = nume / np.sqrt(denom)

    return home_attack_rating, home_defense_rating, away_attack_rating, away_defense_rating


def get_parameters(data, team_map, home_attack_rating, home_defense_rating, away_attack_rating, away_defense_rating):

    theta = []
    theta.extend(home_attack_rating)
    theta.extend(home_defense_rating)
    theta.extend(away_attack_rating)
    theta.extend(away_defense_rating)

    nll = lambda *args: -log_likelihood_function(*args) # multiple likelihood by -1
    result = minimize(nll, theta, args=(data), tol=0.01)
    results = result["x"]

    home_attack, home_defense, away_attack, away_defense = [results[x:x + 20] for x in range(0, len(results), 20)]

    dataframe = pd.DataFrame({'HomeAttack': home_attack, 'HomeDefense': home_defense, 'AwayAttack': away_attack,
                              'AwayDefense': away_defense}, index=list(team_map.keys()))

    return result['message'], dataframe

now = datetime.datetime.now()

data, team_map = read_in_data()

# print(data)
# print(team_map)

home_attack_rating, home_defense_rating, \
away_attack_rating, away_defense_rating = initial_parameter_estimates(data)



message, results = get_parameters(data, team_map, home_attack_rating,home_defense_rating, away_attack_rating, away_defense_rating)

print(datetime.datetime.now()-now)

results.to_csv('./Team ratings/E0/teamratings_16-17_dixoncoles-moretol.csv')



print(message)