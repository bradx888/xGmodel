import numpy as np
import pandas as pd


def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result


def tidy_data(shot_data):
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

    shot_data['Proba_exp'] = myprob(shot_data['Distance'], shot_data['Angle'])

    return shot_data


raw_shots = pd.read_csv('./All shots from 17-18/E0/shots.csv', index_col=0)
football_data = pd.read_csv('./Football-data.co.uk/E0/17-18.csv')
mappings = pd.read_csv('./All shots from 17-18/E0/mappings.csv', index_col=1, header=None)
raw_shots.replace(mappings[0], inplace=True)

raw_shots = tidy_data(raw_shots)

raw_shots = raw_shots.groupby(by = ['Team', 'Match No'], as_index=False)['Proba_exp'].sum()
raw_shots.sort_values(by='Match No',axis=0, inplace=True)
for index, row in football_data.iterrows():
    for index1, row1 in raw_shots.iterrows():
        if index == row1['Match No'] and row1['Team'] == row['AwayTeam']:
            football_data.set_value(index, 'xGA', row1['Proba_exp'])
        if index == row1['Match No'] and row1['Team'] == row['HomeTeam']:
            football_data.set_value(index, 'xGH', row1['Proba_exp'])

football_data.to_csv('./Football-data.co.uk/E0/17-18.csv')