'''
this script sums the xG for each team in a match and then assigns it to
the corresponding match in the football-data.co.uk file

THIS ALSO NOW USES THE NEURAL NETWORK
'''

import numpy as np
import pandas as pd

np.seterr(over='ignore', under='ignore')

def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

def sigmoid(x):
    return 1/(1+np.exp(-x))

def nn_prob(shot_data):
    theta1 = np.load('./Neural Net/xG_theta1.npy')
    theta2 = np.load('./Neural Net/xG_theta2.npy')
    X_cv = shot_data.as_matrix(columns=['x', 'y', 'Header',
                                        'Distance', 'Angle'])

    X_cv = np.c_[np.ones(X_cv.shape[0]), X_cv]

    m = X_cv.shape[0]

    a_1 = X_cv.T
    z_2 = np.dot(theta1, a_1)
    a_2 = sigmoid(z_2)
    # WILL ADD A BIAS UNIT SOON
    z_3 = np.dot(theta2, a_2)
    a_3 = sigmoid(z_3)
    h_theta = a_3
    return h_theta.T


def tidy_and_format_data(shot_data):
    '''
    takes shot data, adds distance, angle, colour and a probability
    :param shot_data:
    :return:
    '''
    for index, row in shot_data.iterrows():
        if row['x'] > 240:
            shot_data.set_value(index, 'x', 480 - row['x'])

    shot_data['y'] = shot_data['y'] - 366/2
    shot_data['y'] = shot_data['y'] * -1
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

    shot_data['Proba_exp'] = nn_prob(shot_data)

    return shot_data

def add_xG_to_fd(raw_shots, football_data):
    '''
    add xG values to the football data file so that it can be used for team ratings etc
    :param raw_shots: shot data exactly how it is scraped from squawka
    :param football_data: football-data file from football-data.co.uk
    :return: saves the new file in the respective folder
    '''
    raw_shots = raw_shots.groupby(by=['Team', 'Match No'], as_index=False)['Proba_exp'].sum()
    raw_shots.sort_values(by='Match No', axis=0, inplace=True)
    for index, row in football_data.iterrows():
        for index1, row1 in raw_shots.iterrows():
            if index == row1['Match No'] and row1['Team'] == row['AwayTeam']:
                football_data.set_value(index, 'xGA', row1['Proba_exp'])
            if index == row1['Match No'] and row1['Team'] == row['HomeTeam']:
                football_data.set_value(index, 'xGH', row1['Proba_exp'])

    football_data.to_csv('./Football-data.co.uk/E0/17-18.csv')

raw_shots = pd.read_csv('./All shots from 17-18/E0/shots.csv', index_col=0)
football_data = pd.read_csv('./Football-data.co.uk/E0/17-18.csv')
mappings = pd.read_csv('./All shots from 17-18/E0/mappings.csv', index_col=1, header=None)
raw_shots.replace(mappings[0], inplace=True)

raw_shots = tidy_and_format_data(raw_shots)

add_xG_to_fd(raw_shots, football_data)

