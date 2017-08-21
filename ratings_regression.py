import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import inv

def format_data(what_rating, home_or_away):
    '''
    reads in all data regarding the ratings
    X contains the features with each feature vector as a row
    Y contains the corresponding ratings
    :return: X and Y
    '''
    data = pd.read_csv('./Football-data.co.uk/E0/15-16 with xG exponential.csv', index_col=0)
    data = data.groupby(by=[home_or_away], as_index=True)[['xGH', 'xGA',
                                                               'FTHG', 'FTAG',
                                                               'HST', 'AST',
                                                               'HS', 'AS']].mean()
    ratings = pd.read_csv('./Team ratings/E0/teamratings_15-16.csv', index_col=0)
    data = pd.concat([data, ratings], axis=1)
    X1 = data.as_matrix(columns=['xGH', 'xGA',
                                    'FTHG', 'FTAG',
                                    'HST', 'AST',
                                    'HS', 'AS'])
    Y1 = data.as_matrix(columns=[what_rating])

    data = pd.read_csv('./Football-data.co.uk/E0/16-17 with xG exponential.csv', index_col=0)
    data = data.groupby(by=[home_or_away], as_index=True)[['xGH', 'xGA',
                                                           'FTHG', 'FTAG',
                                                           'HST', 'AST',
                                                           'HS', 'AS']].mean()
    ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17.csv', index_col=0)
    data = pd.concat([data, ratings], axis=1)
    X2 = data.as_matrix(columns=['xGH', 'xGA',
                                 'FTHG', 'FTAG',
                                 'HST', 'AST',
                                 'HS', 'AS'])
    Y2 = data.as_matrix(columns=[what_rating])

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    X = np.c_[np.ones(X.shape[0]), X]
    return X, Y


def find_theta(X, Y):
    '''
    find theta using the normal equation
    :param X: rows of feature vector
    :param Y:
    :return:
    '''
    theta = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return theta


X, Y = format_data('AwayDefense', 'AwayTeam')
theta = find_theta(X, Y)


Y_rec = np.dot(X, theta)

print(np.sum(np.abs(np.subtract(Y, Y_rec))))

np.save('./Simulation Regression/theta_AwayDefense', theta)