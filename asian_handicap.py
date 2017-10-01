import click
import pandas as pd
import numpy as np
import time
import math

import method

def asian_0(population, weights):
    homeperc, drawperc, awayperc = 0, 0, 0
    for i in range(0, len(population)):
        if population[i][0] > population[i][1]:
            homeperc += weights[i]
        elif population[i][0] == population[i][1]:
            drawperc += weights[i]
        elif population[i][0] < population[i][1]:
            awayperc += weights[i]
    home = (1 - drawperc) * 1 / homeperc
    away = (1 - drawperc) * 1 / awayperc
    return home, away



def asian(populations, weights, handicap):
    homeperc, drawperc, awayperc = 0, 0, 0
    for i in range(0, len(populations)):
        populations[i] = [populations[i][0] + handicap, populations[i][1]]
    for i in range(0, len(populations)):
        if populations[i][0] > populations[i][1]:
            homeperc += weights[i]
        elif populations[i][0] == populations[i][1]:
            drawperc += weights[i]
        elif populations[i][0] < populations[i][1]:
            awayperc += weights[i]
    home = (1 - drawperc) * 1 / homeperc
    away = (1 - drawperc) * 1 / awayperc
    return [handicap, home, away]


if __name__ == '__main__':
    team_ratings = method.read_in_team_ratings()

    home_team = input('Enter home team: ')
    away_team = input('Enter away team: ')
    home_team = home_team.title()
    away_team = away_team.title()


    home_attack = team_ratings.loc[home_team]['HomeAttack']
    home_defense = team_ratings.loc[home_team]['HomeDefense']
    away_attack = team_ratings.loc[away_team]['AwayAttack']
    away_defense = team_ratings.loc[away_team]['AwayDefense']

    handicaps = np.arange(-2, 2.5, 0.5)

    results = []
    for handicap in handicaps:
        population, weights = method.bivpois2(home_attack * away_defense, away_attack * home_defense, 0.15)
        results.append(asian(population, weights, handicap))

    df = pd.DataFrame(columns=['Handicap', home_team, away_team])
    for i in range(0, len(results)):
        df.loc[i] = results[i]
    df.set_index('Handicap', inplace=True)
    df['Diff'] = np.abs(df[home_team] - df[away_team])
    df.sort_values(by='Diff', ascending=True, inplace=True)
    df.drop('Diff', axis=1, inplace=True)

    print(df)


