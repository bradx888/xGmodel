from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import fmin_cg
import functools
import collections

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

    return shot_data

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))

def neural_net(X, Y, alpha, Lambda, num_of_iters, X_cv, Y_cv):
    '''

    :param X: must be fed in with each row corresponding to a training example, without a bias
    :param Y: must be fed in with each row corresponding to a training example (so a column vector)
    :param alpha: learning rate
    :param num_of_iters: number of iterations to train the NN
    :return:
    '''
    m = X.shape[0] # number of training examples
    X = np.c_[np.ones(X.shape[0]), X]
    X_cv = np.c_[np.ones(X_cv.shape[0]), X_cv]

    Y = Y.T
    Y_cv = Y_cv.T

    theta1 = 2 * np.random.random((X.shape[0], X.shape[1])) - 1
    theta2 = 2 * np.random.random((1, X.shape[0])) - 1

    for j in range(0, num_of_iters):
        #print(j)
        '''
        a_i corresponds to layer i of the neural net
        '''

        a_1 = X.T
        z_2 = np.dot(theta1, a_1)
        a_2 = sigmoid(z_2)

        # WILL ADD A BIAS UNIT SOON

        z_3 = np.dot(theta2, a_2)
        a_3 = sigmoid(z_3)

        # NOW START BACKPROPAGATION

        delta_3 = np.subtract(a_3, Y)
        err = np.power(delta_3, 2)
        err = (1/m)*np.sum(err)
        test_err = np.subtract(sigmoid(np.dot(theta2, sigmoid(np.dot(theta1, X_cv.T)))), Y_cv)
        test_err = np.power(test_err, 2)
        test_err = (1 / m) * np.sum(test_err)
        if (j % 100 == 0):
            print(err, test_err)

        #print(delta_3.shape, theta2.shape)
        #print(sigmoid_prime(z_2).shape)

        delta_2 = np.multiply(np.dot(theta2.T, delta_3), sigmoid_prime(z_2))

        reg_theta2 = theta2
        reg_theta2[:, 0] = 0
        reg_theta1 = theta1
        reg_theta1[:, 0] = 0

        theta2 -= alpha * (1 / m) * np.dot(delta_3, a_2.T) + Lambda*reg_theta2
        theta1 -= alpha * (1 / m) * np.dot(delta_2, a_1.T) + Lambda *reg_theta1

    return theta1, theta2


def nnCostFunction(x, *args):
    '''
    :param theta_flattened:
    :param X: args
    :param Y: args
    :param Lambda: args
    :param input_layer_size: args
    :param hidden_layer_size: args
    :return:
    '''
    X, Y, Lambda, input_layer_size, hidden_layer_size = args
    J = 0
    m = X.shape[0] # number of training examples
    X = np.c_[np.ones(X.shape[0]), X]
    n = X.shape[1] # number of features including a bias unit

    Y = Y.T
    theta_flattened = x
    theta1 = theta_flattened[0:(hidden_layer_size) * (n)].reshape(hidden_layer_size, n)
    theta2 = theta_flattened[hidden_layer_size * n:].reshape(1, hidden_layer_size)

    a_1 = X.T
    z_2 = np.dot(theta1, a_1)
    a_2 = sigmoid(z_2)
    # WILL ADD A BIAS UNIT SOON
    z_3 = np.dot(theta2, a_2)
    a_3 = sigmoid(z_3)
    h_theta = a_3

    J = J + (-1/m)*np.sum(np.multiply(Y, np.log(h_theta)) + np.multiply((np.ones(Y.shape) - Y), np.log(np.ones(h_theta.shape) - h_theta)))

    # add regularization
    J = J +(Lambda/(2*m))*(np.sum(np.power(theta1, 2)) + np.sum(np.power(theta2, 2)))

    return J


def nnGradFunction(x, *args):
    '''
    :param theta_flattened:
    :param X: args
    :param Y: args
    :param Lambda: args
    :param input_layer_size: args
    :param hidden_layer_size: args
    :return:
    '''
    X, Y, Lambda, input_layer_size, hidden_layer_size = args
    m = X.shape[0] # number of training examples
    X = np.c_[np.ones(X.shape[0]), X]
    n = X.shape[1] # number of features including a bias unit

    Y = Y.T
    theta_flattened = x
    theta1 = theta_flattened[0:(hidden_layer_size)*(n)].reshape(hidden_layer_size,n)
    theta2 = theta_flattened[hidden_layer_size*n:].reshape(1, hidden_layer_size)

    a_1 = X.T
    z_2 = np.dot(theta1, a_1)
    a_2 = sigmoid(z_2)
    # WILL ADD A BIAS UNIT SOON
    z_3 = np.dot(theta2, a_2)
    a_3 = sigmoid(z_3)
    h_theta = a_3

    # NOW START BACKPROPAGATION

    delta_3 = np.subtract(a_3, Y)
    delta_2 = np.multiply(np.dot(theta2.T, delta_3), sigmoid_prime(z_2))

    reg_theta2 = theta2
    reg_theta2[:, 0] = 0
    reg_theta1 = theta1
    reg_theta1[:, 0] = 0

    theta2_grad = (1/m) * np.dot(delta_3, a_2.T) + Lambda * reg_theta2
    theta1_grad = (1 / m) * np.dot(delta_2, a_1.T) + Lambda * reg_theta1

    theta_grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))

    return theta_grad


data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 15-16/E0/shots_with_headers_formatted.csv')
data2 = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 16-17/E0/shots_with_headers_formatted.csv')

data = pd.concat([data, data2])

# data = tidy_and_format_data(data)

train_data = data
test_data = data.iloc[len(data)-5:]

del data

print(len(train_data), len(test_data))

X = train_data.as_matrix(columns = ['x', 'y', 'Header',
                                    'Distance', 'Angle'])

Y = train_data.as_matrix(columns = ['ScoredBinary'])

lamdda_vals = [0, 0, 0, 0, 0, 0]

best_err = 1

for Lambda in lamdda_vals:
    input_layer_size = X.shape[1] + 1
    hidden_layer_size = input_layer_size
    args = X, Y, Lambda, input_layer_size, hidden_layer_size
    initial_guesses = 2*np.random.random(hidden_layer_size*(X.shape[1]+1)+hidden_layer_size) - 1

    theta, err, na1, na2, na3 = fmin_cg(nnCostFunction, x0 = initial_guesses, fprime=nnGradFunction, args=(args), full_output=True)

    X_cv = test_data.as_matrix(columns = ['x', 'y', 'Header',
                                          'Distance', 'Angle'])

    X_cv = np.c_[np.ones(X_cv.shape[0]), X_cv]

    Y_cv = test_data.as_matrix(columns = ['ScoredBinary'])

    Y_cv = np.vstack(([Y_cv, [1]]))
    Y_cv = np.vstack(([Y_cv, [1]]))
    X_cv = np.vstack(([X_cv, [1, 0, 0, 0, 0, 0]]))
    X_cv = np.vstack(([X_cv, [1, 0, 0, 1, 0, 0]]))
    print(X_cv.shape, Y_cv.shape)

    m = X_cv.shape[0]

    theta1 = theta[0:hidden_layer_size*(X.shape[1]+1)].reshape(hidden_layer_size,(X.shape[1]+1))
    theta2 = theta[hidden_layer_size*(X.shape[1]+1):].reshape(1, hidden_layer_size)

    a_1 = X_cv.T
    z_2 = np.dot(theta1, a_1)
    a_2 = sigmoid(z_2)
    # WILL ADD A BIAS UNIT SOON
    z_3 = np.dot(theta2, a_2)
    a_3 = sigmoid(z_3)
    h_theta = a_3

    J = (-1/m)*np.sum(np.multiply(Y_cv, np.log(h_theta)) + np.multiply((np.ones(Y_cv.shape) - Y_cv), np.log(np.ones(h_theta.shape) - h_theta)))

    # Y_cv = Y_cv.flatten()
    #
    # print(Y_cv.shape, X_cv.shape)
    # h_theta = h_theta.flatten()

    theta = np.concatenate([theta1.flatten(), theta2.flatten()])
    if err < best_err:
        best_err = err
        theta1_best = theta1
        theta2_best = theta2
    print(err, J)
    # for i in range(0, len(Y_cv)):
    print(X_cv[X_cv.shape[0] - 2:, :], h_theta[:, -2:], Y_cv[Y_cv.shape[0] - 2:])

np.save('xG_theta1_head2', theta1_best)
np.save('xG_theta2_head2', theta2_best)