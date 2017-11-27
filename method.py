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

def read_in_fixtures():
     fixtures = pd.read_csv('./Fixtures/E0/Remaining 17-18 Fixtures.csv',
                           index_col=0)

     return fixtures

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

def iterator(fixtures, team_ratings, current_table, mc_iterations, num_cores):

    teams = set(list(fixtures['HomeTeam']))

    total_points = dict.fromkeys(teams, 0)
    goals = dict.fromkeys(teams, 0)
    winnercount = dict.fromkeys(teams, 0)
    relegationcount = dict.fromkeys(teams, 0)
    top4count = dict.fromkeys(teams, 0)
    wincount = dict.fromkeys(teams, 0)
    drawcount = dict.fromkeys(teams, 0)
    losscount = dict.fromkeys(teams, 0)

    for index, row in current_table.iterrows():
        goals[index] += row['GD']*int(mc_iterations/num_cores)
        wincount[index] += row['Wins']*int(mc_iterations/num_cores)
        drawcount[index] += row['Draws']*int(mc_iterations/num_cores)
        losscount[index] += row['Losses']*int(mc_iterations/num_cores)

    population = dict()
    weights = dict()

    for index, row in fixtures.iterrows():
        home_attack = team_ratings.loc[row['HomeTeam']]['HomeAttack']
        home_defense = team_ratings.loc[row['HomeTeam']]['HomeDefense']
        away_attack = team_ratings.loc[row['AwayTeam']]['AwayAttack']
        away_defense = team_ratings.loc[row['AwayTeam']]['AwayDefense']
        population[index], weights[index] = bivpois2(home_attack * away_defense, away_attack * home_defense, 0.15)

    r = Parallel(n_jobs=num_cores, verbose=100)(delayed(future_table)(current_table=current_table, drawcount=drawcount, fixtures=fixtures, goals=goals, losscount=losscount,
                                             mc_iterations=int(mc_iterations/4), population=population, relegationcount=relegationcount,
                 teams=teams, top4count=top4count, total_points=total_points, weights=weights, wincount=wincount, winnercount=winnercount)  for i in range(4))

    for _ in r:
        print(_)
    result =[]
    for i in range(8):
        result.append({k: r[0][i].get(k, 0) + r[1][i].get(k, 0) + r[2][i].get(k, 0) + r[3][i].get(k, 0) for k in set(r[0][i])})
    wincount, drawcount, losscount, total_points, goals, winnercount, top4count, relegationcount = result

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


def future_table(current_table, drawcount, fixtures, goals, losscount, mc_iterations, population, relegationcount,
                 teams, top4count, total_points, weights, wincount, winnercount):
    for i in range(int(mc_iterations)):
        points = dict.fromkeys(teams, 0)
        for index, row in current_table.iterrows():
            points[index] += row['Points']
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

    return wincount, drawcount, losscount, total_points, goals, winnercount, top4count, relegationcount


def calculate_current_table(fixtures):
    data = pd.read_csv('./Football-data.co.uk/E0/17-18.csv')
    teams = set(list(fixtures['HomeTeam']))
    points = dict.fromkeys(teams, 0)
    goals = dict.fromkeys(teams, 0)
    wincount = dict.fromkeys(teams, 0)
    drawcount = dict.fromkeys(teams, 0)
    losscount = dict.fromkeys(teams, 0)

    for index, row in data.iterrows():
        if row['FTHG'] > row['FTAG']:
            points[row['HomeTeam']] += 3
            wincount[row['HomeTeam']] += 1
            losscount[row['AwayTeam']] += 1
        elif row['FTHG'] == row['FTAG']:
            points[row['HomeTeam']] += 1
            points[row['AwayTeam']] += 1
            drawcount[row['HomeTeam']] += 1
            drawcount[row['AwayTeam']] += 1
        elif row['FTHG'] < row['FTAG']:
            points[row['AwayTeam']] += 3
            wincount[row['AwayTeam']] += 1
            losscount[row['HomeTeam']] += 1
        goals[row['HomeTeam']] += row['FTHG']
        goals[row['HomeTeam']] -= row['FTAG']
        goals[row['AwayTeam']] += row['FTAG']
        goals[row['AwayTeam']] -= row['FTHG']

    results = pd.DataFrame({'Points': points, 'Wins': wincount, 'Draws': drawcount, 'Losses': losscount, 'GD': goals})
    return results

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