import pandas as pd
import numpy as np
import time
import math

def promoted_teams_attack(old_param):
    parameter = 0.53683956207924333 * old_param + 0.049598077388848139
    return parameter

def promoted_teams_defense(old_param):
    parameter = -0.88138228873841129 * old_param + 2.2188585603647821
    return parameter

def pois(x, lambdaa):
    result = (np.power(lambdaa, x) * np.exp(-lambdaa)) /math.factorial(x)
    return result

def tau(x, y, lambdaa, mu, rho):
    if x == 0 and y == 0:
        result = 1 - (lambdaa*mu*rho)
    elif x == 0 and y == 1:
        result = 1 + (lambdaa*rho)
    elif  x == 1 and y == 0:
        result = 1 + (mu*rho)
    elif x == 1 and y == 1:
        result = 1 - rho
    else:
        result = 1
    return result

def bivpois2(lambdaa, mu, rho):
    max_goals=6
    weights = []
    population = []
    for i in range(0, max_goals+1):
        for j in range(0, max_goals+1):
            population.append([i,j])
            weights.append(tau(i, j, lambdaa, mu, rho)*pois(i, lambdaa)*pois(j, mu))
    return population, weights

def predictor2(population, weights):
    new = random.choices(population, weights)
    return new

def calculate_win_perc(population, weights):
    homeperc, drawperc, awayperc = 0, 0, 0
    for i in range(0, len(population)):
        if population[i][0] > population[i][1]:
            homeperc += weights[i]
        elif population[i][0] == population[i][1]:
            drawperc += weights[i]
        elif population[i][0] < population[i][1]:
            awayperc += weights[i]
    return [homeperc, drawperc, awayperc]

def calculate_goal_line(population, weights):
    result = 0
    for i in range(0, len(population)):
        if population[i][0] + population[i][1] > 2:
            result += weights[i]

    return result, 1-result

def read_in_team_ratings():
    team_ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17.csv',
                           index_col=0)

    championship_team_ratings = pd.read_csv('./Team ratings/E1/teamratings_16-17.csv',
                                        index_col=0)

    for column in championship_team_ratings:
        if 'Attack' in column:
            championship_team_ratings[column] = promoted_teams_attack(championship_team_ratings[column])
        else:
            championship_team_ratings[column] = promoted_teams_defense(championship_team_ratings[column])

    team_ratings = team_ratings.append(championship_team_ratings)

    ''' the following code updates the team ratings based on this seasons current results'''
    new_data = calculate_this_seasons_ratings()

    new_data = combine_ratings(new_data, team_ratings)

    new_data.to_csv('./Team ratings/E0/teamratings_17-18.csv')
    return new_data

def combine_ratings(new_data, team_ratings, exp_factor=0.08):
    '''
    :param new_data: new_data is the current seasons ratings
    :param team_ratings: team_ratings is last seasons ratings
    :param exp_factor:
    :return:
    '''
    for index, row in new_data.iterrows():
        team_ratings.set_value(index, 'HomeAttack',
                               team_ratings.ix[index, 'HomeAttack'] * np.exp(-exp_factor * row['HomeCount']))
        team_ratings.set_value(index, 'AwayAttack',
                               team_ratings.ix[index, 'AwayAttack'] * np.exp(-exp_factor * row['AwayCount']))
        team_ratings.set_value(index, 'HomeDefense',
                               team_ratings.ix[index, 'HomeDefense'] * np.exp(-exp_factor * row['HomeCount']))
        team_ratings.set_value(index, 'AwayDefense',
                               team_ratings.ix[index, 'AwayDefense'] * np.exp(-exp_factor * row['AwayCount']))

    for index, row in new_data.iterrows():
        new_data.set_value(index, 'HomeAttack', team_ratings.ix[index, 'HomeAttack'] + row['HomeAttack'])
        new_data.set_value(index, 'AwayAttack', team_ratings.ix[index, 'AwayAttack'] + row['AwayAttack'])
        new_data.set_value(index, 'HomeDefense', team_ratings.ix[index, 'HomeDefense'] + row['HomeDefense'])
        new_data.set_value(index, 'AwayDefense', team_ratings.ix[index, 'AwayDefense'] + row['AwayDefense'])

    new_data.drop(['HomeCount',
                   'AwayCount',
                   ], axis=1 ,inplace=True)
    for index, row in team_ratings.iterrows():
        if index not in new_data.index.values:
            new_data.loc[index] = row

    return new_data

def calculate_this_seasons_ratings(exp_factor=0.08):
    data = pd.read_csv(
        './Football-data.co.uk/E0/17-18.csv'
    )
    teams = set(list(data['HomeTeam']) + list(data['AwayTeam']))
    list_of_ratings = ['HomeAttack', 'HomeDefense', 'AwayAttack', 'AwayDefense']
    new_ratings = pd.DataFrame(data=np.zeros(20*6).reshape(20, 6), index=teams, columns=list_of_ratings + ['HomeCount', 'AwayCount'])
    for rating in list_of_ratings:
        temp_data = data.groupby(by=[rating[0:4] + 'Team'], as_index=True)[['xGH', 'xGA',
                                                               'FTHG', 'FTAG',
                                                               'HST', 'AST',
                                                               'HS', 'AS']].mean()

        temp_data['xGH FTHG'] = 0.8 * temp_data['xGH'] + 0.2 * temp_data['FTHG']
        temp_data['xGA FTAG'] = 0.8 * temp_data['xGA'] + 0.2 * temp_data['FTAG']

        X = temp_data.as_matrix(columns=['xGH FTHG',
                                    'xGA FTAG'])


        theta = np.load('./Ratings regression/Parameters/theta_' + rating + '.npy')
        X = np.c_[np.ones(X.shape[0]), X]
        temp_ratings = np.dot(X, theta)
        temp_teams = temp_data.index.values
        temp_dict = dict.fromkeys(temp_teams, 0)
        for i in range(0, len(temp_teams)):
            temp_dict[temp_teams[i]] = temp_ratings[i]
        for key, value in temp_dict.items():
            for index, row in new_ratings.iterrows():
                if index == key:
                    new_ratings.set_value(index, rating, temp_dict[index])


    # need to apply exponential function to these
    home_games_played = data.groupby(by=['HomeTeam'], as_index=True)[['xGH']].count()
    home_games_played = home_games_played.to_dict()['xGH']
    away_games_played = data.groupby(by=['AwayTeam'], as_index=True)[['xGA']].count()
    away_games_played = away_games_played.to_dict()['xGA']
    for key, value in home_games_played.items():
        for index, row in new_ratings.iterrows():
            if index == key:
                new_ratings.set_value(index, 'HomeCount', home_games_played[index])
    for key, value in away_games_played.items():
        for index, row in new_ratings.iterrows():
            if index == key:
                new_ratings.set_value(index, 'AwayCount', away_games_played[index])

    new_ratings['HomeAttack'] = (1 - np.exp(-exp_factor * new_ratings['HomeCount'])) * new_ratings['HomeAttack']
    new_ratings['HomeDefense'] = (1 - np.exp(-exp_factor * new_ratings['HomeCount'])) * new_ratings['HomeDefense']
    new_ratings['AwayAttack'] = (1 - np.exp(-exp_factor * new_ratings['AwayCount'])) * new_ratings['AwayAttack']
    new_ratings['AwayDefense'] = (1 - np.exp(-exp_factor * new_ratings['AwayCount'])) * new_ratings['AwayDefense']

    return new_ratings

team_ratings = read_in_team_ratings()

home_team = input('Enter the home team: ')
time.sleep(0.5)
away_team = input('Enter the away team: ')

home_attack = team_ratings.loc[home_team]['HomeAttack']
home_defense = team_ratings.loc[home_team]['HomeDefense']
away_attack = team_ratings.loc[away_team]['AwayAttack']
away_defense = team_ratings.loc[away_team]['AwayDefense']

population, weights = bivpois2(home_attack*away_defense, away_attack*home_defense, 0.15)

percentages = calculate_win_perc(population, weights)
overs, unders = calculate_goal_line(population, weights)

print(home_team, 1 - percentages[0])
print('Draw', 1/percentages[1])
print(away_team, 1/percentages[2])
print('O 2.5', 1/overs)
print('U 2.5', 1/unders)


