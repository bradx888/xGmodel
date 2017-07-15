'''i plotted the probability gained from a logistic regression
against a combination of distance and angle. The below was the best
fit for the exponential graph i saw

Fit an exponential by using the np.polyfit function with deg=1 and the y
data being logged
'''

# function for calculating the probability based on an exponential function
def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

# group shots by match No and team and then match up with Football Data creating an xGH and xGA column

shots = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 15-16/shots with proba.csv', index_col =0)
shots = shots.groupby(by = ['Team', 'Match No'], as_index=False)['Proba_exp'].sum()
shots.sort_values(by='Match No',axis=0, inplace=True)
data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/15-16.csv')
mappings = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 15-16/mappings.csv', index_col=1, header=None)
shots.replace(mappings[0], inplace=True)
for index, row in data.iterrows():
    for index1, row1 in shots.iterrows():
        if index == row1['Match No'] and row1['Team'] == row['AwayTeam']:
            data.set_value(index, 'xGA', row1['Proba_exp'])
        if index == row1['Match No'] and row1['Team'] == row['HomeTeam']:
            data.set_value(index, 'xGH', row1['Proba_exp'])
data.to_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/15-16 with xG exponential.csv')