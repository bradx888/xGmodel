import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.misc import imread
from datetime import datetime

np.seterr(over='ignore', under='ignore')

def sigmoid(x):
    return 1/(1+np.exp(-x))

def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

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


data = pd.read_csv('./All shots from 17-18/E0/shots.csv', index_col=0)
football_data = pd.read_csv('./Football-data.co.uk/E0/17-18.csv')
mappings = pd.read_csv('./All shots from 17-18/E0/mappings.csv', index_col=1, header=None)
data.replace(mappings[0], inplace=True)
img = imread('./xG Plots/background.jpg')
img = np.swapaxes(img, 0, 1)

data = tidy_and_format_data(data) # add xG values to each shot and a colour etc

print(list(data.groupby(['PlayerID'])['Proba_exp'].sum().sort_values(ascending=False).head(20).index.values))

player_id = input('Input player ID: ')
player_id = int(player_id)

print(data[data.PlayerID == player_id]['PlayerName'].head(1))

player_name = input('Enter player name: ')

data = data[data["PlayerID"] == player_id]

team = data.iloc[0]['Team']

try:
    goals = data['Scored'].value_counts()['Scored']
except KeyError:
    goals = 0

xG = np.round(sum(data['Proba_exp']), 2)


data['Colour'] = data['Colour'].replace({'r': 'lightcoral', 'b':'royalblue'})

data['y'] = -1 * data['y']

data.sort_values(by=['Scored'], ascending=True, inplace=True)

fig, ax = plt.subplots()

ax.scatter(data[data.Header == 0]['y'], data[data.Header == 0]['x'], marker="o", s=data[data.Header == 0]['Proba_exp']*800,
            facecolors=data[data.Header == 0]['Colour'],
            edgecolors='black', linewidth=0.4)
ax.scatter(data[data.Header == 1]['y'], data[data.Header == 1]['x'], marker="^", s=data[data.Header == 1]['Proba_exp']*800,
            facecolors=data[data.Header == 1]['Colour'],
            edgecolors='black', linewidth=0.4)
plt.xlim(-366/2, 366/2)
plt.ylim(-10, 250)
plt.imshow(img, zorder=0, extent=[-366/2, 366/2, -10, 490])
text = plt.text(160, 180, player_name + '\n' + team + '\n' + 'Goals: ' + str(goals) + '\n' + 'xG: ' + str(xG),
                horizontalalignment='right',
         color='gold', fontsize=14, fontweight='bold')
text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])

plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                wspace=None, hspace=None)
plt.axis('off')
plt.show()
plt.savefig('./xG Plots/17-18 Players/' + player_name + ' ' + datetime.today().strftime("%Y-%m-%d") + '.png',
            bbox_inches=0, pad_inches=0)


