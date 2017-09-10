import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import fmin_cg

'''

The relationship will be between 0.8*xG plus 0.2*Goals. This is what I used 
for the main ratings in the first place. Plus they all fit a straight line fairly well

'''

def plot_relationships():

    xG = pd.read_csv('./Football-data.co.uk/E0/16-17 with xG.csv')
    ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17.csv')

    xG_home = xG.groupby(by='HomeTeam').mean()
    xG_away = xG.groupby(by='AwayTeam').mean()

    plt.scatter(0.8*xG_home['xGH']+0.2*xG_home['FTHG'], ratings['HomeAttack'])
    for label, x, y in zip(xG_home.index.values, xG_home['xGH'], ratings['HomeAttack']):
        plt.annotate(
            label,
            xy=(x, y), xytext=(x, y),
            textcoords='offset points')
    plt.savefig('./Ratings regression/Plots/xGH vs. HomeAttack 16-17.png')
    plt.clf()

    plt.scatter(0.8*xG_home['xGA']+0.2*xG_home['FTAG'], ratings['HomeDefense'])
    for label, x, y in zip(xG_home.index.values, xG_home['xGA'], ratings['HomeDefense']):
        plt.annotate(
            label,
            xy=(x, y), xytext=(x, y),
            textcoords='offset points')
    plt.savefig('./Ratings regression/Plots/xGH vs. HomeDefense 16-17.png')
    plt.clf()

    plt.scatter(0.8*xG_away['xGH'] + 0.2*xG_away['FTHG'], ratings['AwayDefense'])
    for label, x, y in zip(xG_away.index.values, xG_away['xGH'], ratings['AwayDefense']):
        plt.annotate(
            label,
            xy=(x, y), xytext=(x, y),
            textcoords='offset points')
    plt.savefig('./Ratings regression/Plots/xGH vs. AwayDefense 16-17.png')
    plt.clf()

    plt.scatter(0.8*xG_away['xGA']+0.2*xG_away['FTAG'], ratings['AwayAttack'])
    for label, x, y in zip(xG_away.index.values, xG_away['xGA'], ratings['AwayAttack']):
        plt.annotate(
            label,
            xy=(x, y), xytext=(x, y),
            textcoords='offset points')
    plt.savefig('./Ratings regression/Plots/xGH vs. AwayAttack 16-17.png')
    plt.clf()

def calculate_parameters():
    xG = pd.read_csv('./Football-data.co.uk/E0/16-17 with xG.csv')
    ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17.csv')

    xG_home = xG.groupby(by='HomeTeam').mean()
    xG_away = xG.groupby(by='AwayTeam').mean()

    X = np.matrix(0.8*xG_away['xGH'] + 0.2*xG_away['FTHG']).T
    X = np.c_[np.ones(X.shape[0]), X]
    Y = ratings.as_matrix(columns=['AwayDefense'])

    return find_theta(X, Y)

def find_theta(X, Y):
    '''
    I've commented out the normal equation and am now minimizing the cost function
    :param X: rows of feature vector
    :param Y:
    :return:
    '''
    # NORMAL EQUATION ## theta = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))
    theta0 = np.random.rand(2, 1)  # intial estimates
    for i in range(0, 100000):
        args = (X, Y)
        J = lin_cost_function(theta0, X, Y)
        theta0 = np.subtract(theta0, 0.01*lin_grad(theta0, X, Y))
    return theta0


def lin_cost_function(theta, X, Y):
    #X, Y = args
    m = X.shape[0]
    J = (1/(2*m)) * np.power(np.subtract(np.dot(X, theta), Y), 2)
    J = np.sum(J)
    return J

def lin_grad(theta, X, Y):
    #X, Y = args
    m = X.shape[0]
    Jprime = (1/m)*np.dot(X.T, np.subtract(np.dot(X, theta), Y));
    return Jprime


theta = calculate_parameters()
theta = np.insert(theta, 2, 0, axis=0)
print(theta)
np.save('./Ratings regression/Parameters/theta_AwayDefense', theta)

