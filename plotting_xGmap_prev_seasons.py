'''
This is slightly different from the 17-18 version of the script purely because
the shot files for previous seasons were all formatted before writing the script.
The new script works from raw shot data scraped from squawka
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.misc import imread

data = pd.read_csv('./All shots from 16-17/E0/shots with proba.csv', index_col=0)
football_data = pd.read_csv('./Football-data.co.uk/E0/16-17 with xG exponential.csv', index_col=0)
mappings = pd.read_csv('./All shots from 16-17/E0/mappings.csv', index_col=1, header=None)
data.replace(mappings[0], inplace=True)
img = imread('./xG Plots/background.jpg')

home_team = input('Input home team: ')
away_team = input('Input away team: ')

football_data['Date'] = pd.to_datetime(football_data['Date'], format='%d/%m/%y')

for index, row in football_data.iterrows():
    if row['HomeTeam'] == home_team and row['AwayTeam'] == away_team:
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        date = row['Date']
        match_no = index

data = data[data["Match No"] == match_no]

data['Colour'] = data['Colour'].replace({'r': 'lightcoral', 'b':'royalblue'})

data['y'] = -1 * data['y']

xG_home = data.loc[data['Team'] == home_team, 'Proba_exp'].sum()
xG_away = data.loc[data['Team'] == away_team, 'Proba_exp'].sum()

for index, row in data.iterrows():
    if row['Team'] == away_team:
        data.set_value(index, 'x', 480 -row['x'])

plt.scatter(data['x'], data['y'], s=data['Proba_exp']*400, facecolors=data['Colour'],
            edgecolors='black', linewidth=0.4)
plt.ylim(-366/2, 366/2)
plt.xlim(-10, 490)
plt.imshow(img, zorder=0, extent=[-10, 490, -366/2, 366/2])
text = plt.text(240, 120, home_team + ' vs. ' + away_team + '\n'
         + 'Score: ' + str(home_goals) + '  -  ' + str(away_goals) + '\n'
         + 'xG: ' + str(np.round(xG_home,2)) + '  -  ' + str(np.round(xG_away,2)), horizontalalignment='center',
         color='gold', fontsize=10, fontweight='bold')
# plt.text(240, 130, 'Score: ' + str(home_goals) + '  -  ' + str(away_goals), horizontalalignment='center',
#          color='gold', fontsize=10, fontweight='bold')
# plt.text(240, 110, 'xG: ' + str(np.round(xG_home,2)) + '  -  ' + str(np.round(xG_away,2)), horizontalalignment='center',
#          color='gold', fontsize=10, fontweight='bold')
text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])

plt.axis('off')
plt.savefig('./xG Plots/' + home_team + '-' + away_team + ' ' + date.strftime("%Y-%m-%d") + '.png',
            bbox_inches='tight')
