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

    return result



team_ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17_dixoncoles-moretol.csv',
                           index_col=0)

championship_team_ratings = pd.read_csv('./Team ratings/E1/championship_teamratings_16-17.csv',
                                        index_col=0)

for column in championship_team_ratings:
    if 'Attack' in column:
        championship_team_ratings[column] = promoted_teams_attack(championship_team_ratings[column])
    else:
        championship_team_ratings[column] = promoted_teams_defense(championship_team_ratings[column])

team_ratings = team_ratings.append(championship_team_ratings)

home_team = input('Enter the home team: ')
time.sleep(0.5)
away_team = input('Enter the away team: ')

home_attack = team_ratings.loc[home_team]['HomeAttack']
home_defense = team_ratings.loc[home_team]['HomeDefense']
away_attack = team_ratings.loc[away_team]['AwayAttack']
away_defense = team_ratings.loc[away_team]['AwayDefense']

population, weights = bivpois2(home_attack*away_defense, away_attack*home_defense, 0.15)

percentages = calculate_win_perc(population, weights)
goal_line = calculate_goal_line(population, weights)

print(home_team, 1/percentages[0])
print('Draw', 1/percentages[1])
print(away_team, 1/percentages[2])
print('O 2.5', 1/goal_line)


