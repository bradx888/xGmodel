import pandas as pd
import numpy as np
import math
import random
import datetime

def promoted_teams_attack(old_param):
    parameter = 0.53683956207924333 * old_param + 0.049598077388848139
    return parameter

def promoted_teams_defense(old_param):
    parameter = -0.88138228873841129 * old_param + 2.2188585603647821
    return parameter

def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

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

def bivpois(max_goals, lambdaa, mu, rho):
        probability_matrix = [[0 for i in range(max_goals + 1)] for j in range(max_goals + 1)]
        for i in range(0, max_goals+1):
            for j in range(0, max_goals+1):
                probability_matrix[i][j] = tau(i, j, lambdaa, mu, rho)*pois(i, lambdaa)*pois(j, mu)
        return np.array(probability_matrix)

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

def predictor(probability_matrix):
    x = random.random()
    running_total = 0
    home_goals = 10
    away_goals = 10
    score = []
    for i in range(0, 5):
        for j in range(0, 5):
            # print(str(probability_matrix[i][j]) + ', ', end =" ")
            running_total += probability_matrix[i][j]
            if running_total > x:
                home_goals = i
                away_goals = j
                break
        if home_goals != 10 and away_goals != 10:
            break
    if home_goals == 10 and away_goals == 10:
        home_goals = 0
        away_goals = 0
    score = [home_goals, away_goals]
    return score

def read_in_fixtures():
     fixtures = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Fixtures/E0/17-18 Fixtures.csv',
                           index_col=0)

     return fixtures

def read_in_team_ratings():
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

    return team_ratings

def iterator(fixtures, team_ratings, mc_iterations):

    teams = set(list(fixtures['HomeTeam']))

    total_points = dict.fromkeys(teams, 0)
    goals = dict.fromkeys(teams, 0)
    winnercount = dict.fromkeys(teams, 0)
    relegationcount = dict.fromkeys(teams, 0)
    top4count = dict.fromkeys(teams, 0)
    wincount = dict.fromkeys(teams, 0)
    drawcount = dict.fromkeys(teams, 0)
    losscount = dict.fromkeys(teams, 0)


    population = dict()
    weights = dict()

    for index, row in fixtures.iterrows():
        home_attack = team_ratings.loc[row['HomeTeam']]['HomeAttack']
        home_defense = team_ratings.loc[row['HomeTeam']]['HomeDefense']
        away_attack = team_ratings.loc[row['AwayTeam']]['AwayAttack']
        away_defense = team_ratings.loc[row['AwayTeam']]['AwayDefense']
        population[index], weights[index] = bivpois2(home_attack * away_defense, away_attack * home_defense, 0.15)

    for mc_iteration in range(0, mc_iterations):
        points = dict.fromkeys(teams, 0)
        for index, row in fixtures.iterrows():
            score = predictor2(population[index], weights[index])[0]
            # print(score)
            if score[0] > score[1]:
                points[row['HomeTeam']] += 3
                wincount[row['HomeTeam']] += 1
                losscount[row['AwayTeam']] += 1
            elif score[0] == score[1]:
                points[row['HomeTeam']] += 1
                points[row['AwayTeam']] += 1
                drawcount[row['HomeTeam']] += 1
                drawcount[row['AwayTeam']] += 1
            elif score[0] < score[1]:
                points[row['AwayTeam']] += 3
                wincount[row['AwayTeam']] += 1
                losscount[row['HomeTeam']] += 1
            goals[row['HomeTeam']] += score[0]
            goals[row['HomeTeam']] -= score[1]
            goals[row['AwayTeam']] += score[1]
            goals[row['AwayTeam']] -= score[0]
            # print(row['HomeTeam'], row['AwayTeam'], score)
        for team in teams:
            total_points[team] += points[team]
        points = pd.Series(points)
        winnercount[points.idxmax()] += 1
        top4count[points.idxmax()] += 1
        points.drop(points.idxmax(), axis=0, inplace=True)
        for i in range(0, 3):
            relegationcount[points.idxmin()] += 1
            top4count[points.idxmax()] += 1
            points.drop([points.idxmin(), points.idxmax()], axis=0, inplace=True)

    results = pd.DataFrame(
        {'W': wincount, 'D': drawcount, 'L': losscount, 'Pts': total_points, 'GD': goals, '%Title': winnercount, '%Top4': top4count, '%Releg': relegationcount})
    for column in results:
        if '%' in column:
            results[column] = np.round((results[column] / mc_iterations) * 100, decimals=2)
        else:
            results[column] = np.round((results[column] / mc_iterations), decimals=2)

    results.sort_values('Pts', ascending=False, inplace=True)
    cols = ['W', 'D', 'L', 'Pts', 'GD', '%Title', '%Top4', '%Releg']
    results = results[cols]
    return results

now = datetime.datetime.now() # for measuring the time taken

fixtures = read_in_fixtures()
team_ratings = read_in_team_ratings()

results = iterator(fixtures, team_ratings, 10000)

results.to_csv('./Table Predictions/E0/' + datetime.datetime.today().strftime("%Y-%m-%d") + '.csv')

print(datetime.datetime.now()-now) # for measuring the time taken