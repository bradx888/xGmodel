'''
this script is what is used to actually 'make' an xG model
the myprob function is the current formula used to assign a probability to each shot
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def promoted_teams_attack(old_param):
    parameter = 0.53683956207924333 * old_param + 0.049598077388848139
    return parameter

def promoted_teams_defense(old_param):
    parameter = -0.88138228873841129 * old_param + 2.2188585603647821
    return parameter

    return homeattack, homedefense, awayattack, awaydefense

def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

# Read in both datasets and join them together for a better sample size
shot_data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 16-17/shots.csv')
shot_data2 = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 15-16/shots.csv')

shot_data['Season'] = '16/17'
shot_data2['Season'] = '15/16'

shot_data = shot_data.append(shot_data2, ignore_index=True)

del(shot_data2)

# edit the y coordinate so that y=0 is in the centre of the goal
#shot_data['y'] = shot_data['y'] - 366/2

# calculate the angle and the distance from the goal
shot_data['Angle'] = np.arctan((np.absolute(shot_data['y'])/shot_data['x']))
shot_data['Distance'] = np.sqrt(shot_data['y']*shot_data['y'] + shot_data['x']*shot_data['x'])

# assign colours and numbers based on whether the shots were scored or missed
for index, row in shot_data.iterrows():
    if row['Scored'] == 'Scored':
        shot_data.set_value(index, 'Colour', 'b')
        shot_data.set_value(index, 'ScoredBinary', 1)
    else:
        shot_data.set_value(index, 'Colour', 'r')
        shot_data.set_value(index, 'ScoredBinary', 0)

# # xG model for angle from the goal
# angle_cuts = (shot_data.groupby(pd.cut(shot_data['Angle'], 6, right=False)).mean())
# new_index = []
# for item in angle_cuts.index.values:
#     new_index.append(item.mid)
#
# angle_cuts.index = new_index
# p_angle = np.poly1d(np.polyfit(angle_cuts.index.values, angle_cuts['ScoredBinary'], deg=3))
#
# # xG model for dist from the goal
# distance_cuts = (shot_data.groupby(pd.cut(shot_data['Distance'], 10, right=False)).mean())
# new_index = []
# for item in distance_cuts.index.values:
#     new_index.append(item.mid)
#
# distance_cuts.index = new_index
# popt, pcov = curve_fit(p_distance, distance_cuts.index.values, distance_cuts['ScoredBinary'], p0 = (1, 1, -0.4))
#
# print(popt)
#
# xx = np.linspace(0, 300)
#
# plt.scatter(xx, p_distance(xx, *popt))
# plt.show()
#
# shot_data['Proba'] = 0.0
#
# for index, row in shot_data.iterrows():
#     val = 0.8 * p_distance(row['Distance'], popt[0], popt[1], popt[2]) + 0.2 * p_angle(row['Angle'])
#     shot_data.set_value(index, 'Proba', val)
#     # print(p_angle(row['Angle']))

xGmodel = LogisticRegression()

X, y = shot_data[['Distance', 'Angle']], shot_data['ScoredBinary']

xGmodel.fit(X, y)

# shot_data = shot_data[shot_data['Season']=='16/17']

probabilities = xGmodel.predict_proba(shot_data[['Distance', 'Angle']])[:,1]

shot_data['Proba'] = probabilities

shot_data.to_csv('shots with proba - xy.csv')

xG = []
G = []

# teams = set(list(shot_data['Team']))
#
# for team in teams:
#     xG.append(sum(shot_data[shot_data['Against']==team]['Proba']))
#     G.append(sum(shot_data[shot_data['Against']==team]['ScoredBinary']))
#
# G = np.array(G)
# xG = np.array(xG)
#
# plt.scatter(G, xG)
#
# regr = LinearRegression()
# regr.fit(G[:, None], xG)
#
# print(regr.score(G[:, None], xG))
#
# plt.show()