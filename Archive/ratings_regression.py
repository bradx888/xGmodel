import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import inv
import time
from random import randint

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
                                                           'HS', 'AS',
                                                           ]].mean()
    data['HSToverHS'] = data['HST'] / data['HS']
    data['ASToverAS'] = data['AST'] / data['AS']
    data['xGHoverFTHG'] = data['xGH'] / data['FTHG']
    data['xGAoverFTAG'] = data['xGA'] / data['FTAG']
    ratings = pd.read_csv('./Team ratings/E0/teamratings_15-16.csv', index_col=0)
    data = pd.concat([data, ratings], axis=1)
    X = data.as_matrix(columns=['xGH', 'xGA',
                                'FTHG', 'FTAG',
                                'HST', 'AST',
                                'HS', 'AS',
                                'HSToverHS', 'ASToverAS',
                                'xGHoverFTHG', 'xGAoverFTAG'])
    Y = data.as_matrix(columns=[what_rating])

    data = pd.read_csv('./Football-data.co.uk/E0/16-17 with xG exponential.csv', index_col=0)
    data = data.groupby(by=[home_or_away], as_index=True)[['xGH', 'xGA',
                                                           'FTHG', 'FTAG',
                                                           'HST', 'AST',
                                                           'HS', 'AS',
                                                           ]].mean()
    data['HSToverHS'] = data['HST'] / data['HS']
    data['ASToverAS'] = data['AST'] / data['AS']
    data['xGHoverFTHG'] = data['xGH'] / data['FTHG']
    data['xGAoverFTAG'] = data['xGA'] / data['FTAG']
    ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17.csv', index_col=0)
    data = pd.concat([data, ratings], axis=1)
    X_test = data.as_matrix(columns=['xGH', 'xGA',
                                    'FTHG', 'FTAG',
                                    'HST', 'AST',
                                    'HS', 'AS',
                                    'HSToverHS', 'ASToverAS',
                                    'xGHoverFTHG', 'xGAoverFTAG'])
    Y_test = data.as_matrix(columns=[what_rating])

    # X = np.concatenate((X1, X2), axis=0)
    # Y = np.concatenate((Y1, Y2), axis=0)

    X = np.c_[np.ones(X.shape[0]), X]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    col_names = ['bias', 'xGH', 'xGA',
                'FTHG', 'FTAG',
                'HST', 'AST',
                'HS', 'AS',
                'HSToverHS', 'ASToverAS',
                'xGHoverFTHG', 'xGAoverFTAG']

    return X, Y, X_test, Y_test, col_names


def find_theta(X, Y):
    '''
    find theta using the normal equation
    :param X: rows of feature vector
    :param Y:
    :return:
    '''
    theta = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return theta


X, Y, X_test, Y_test, col_names = format_data('AwayAttack', 'AwayTeam')

lowest_error = 100
best_cols = []

for i in range(0, 10000):
    list_of_ints = [0]
    how_many = randint(2, (X.shape[1]-1))
    for j in range(0, how_many):
        x = randint(2, (X.shape[1])) - 1
        list_of_ints.append(x)
    list_of_ints = list(set(list_of_ints))
    list_of_ints.sort()
    temp_col_names = [col_names[i] for i in list_of_ints]
    # print(temp_col_names)
    theta = find_theta(X[:, list_of_ints], Y)
    Y_rec = np.dot(X_test[:, list_of_ints], theta)
    new_error = np.subtract(Y_test, Y_rec)
    new_error = np.power(new_error, 2)
    new_error = np.sum((new_error))
    if new_error < lowest_error:
        best_cols = temp_col_names
        lowest_error = new_error
        best_theta = theta

print(best_cols)
print(lowest_error)

np.save('./Simulation Regression/theta_AwayAttack_new', best_theta)
np.save('./Simulation Regression/colnames_AwayAttack_new', best_cols)