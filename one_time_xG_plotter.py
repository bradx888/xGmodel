import numpy as np
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import lxml
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import datetime
import time

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

def get_shots(home_team, away_team, date, league):

    mappings = pd.read_csv('./Team mappings/all_leagues.csv', index_col=0, header=None)

    home_team = mappings[1][home_team]
    away_team = mappings[1][away_team]

    if league == 'BPL':
        league = 'english-premier-league'
        league_prefix = 'epl'
    elif league == 'La Liga':
        league = 'spanish-la-liga'
        league_prefix = 'la-liga'
    elif league == 'Bundesliga':
        league = 'german-bundesliga'
        league_prefix = 'b-liga'
    elif league == 'Serie A':
        league = 'italian-serie-a'
        league_prefix = 'serie-a'
    elif league == 'Ligue 1':
        league = 'french-ligue-1'
        league_prefix = 'ligue-1'
    elif league == 'Champions League':
        league = 'champions-league'
        league_prefix = 'champions-league'
    elif league == 'Championship':
        league = 'english-football-league-championship'
        league_prefix = 'championship'

    temp_shots_data = []

    while len(temp_shots_data) == 0:
        chromedriver = "./Chromedriver/chromedriver"
        driver = webdriver.Chrome(chromedriver)
        driver.get(
            'http://' + league_prefix + '.squawka.com/' + league + '/' + date + '/' + home_team + '-vs-' +
            away_team + '/matches')
        try:

            driver.find_element_by_xpath('//*[(@id = "mc-stat-shot")]').click()
            driver.find_element_by_xpath('//*[(@id = "team2-select")]').click()
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]/span').click()

            soup = BeautifulSoup(driver.page_source, 'lxml')

            shots = soup.find_all('g')

            for shot in shots:
                if shot.circle['r'] == '6.5':
                    if shot.circle.next_sibling['fill'] == '#333333':
                        scored = 'Scored'
                    else:
                        scored = 'Missed'
                    if float(shot.circle['cx']) > 240:
                        temp_shots_data.append({'x': 480 - float(shot.circle['cx']),
                                                'y': shot.circle['cy'],
                                                'Team': home_team,
                                                'Against': away_team,
                                                'Scored': scored,
                                                'Date': date,
                                                'PlayerID': shot['class'][0],
                                                'PlayerName': shot['class'][0],
                                                'Header': 1})
                    else:
                        temp_shots_data.append({'x': shot.circle['cx'],
                                                'y': shot.circle['cy'],
                                                'Team': away_team,
                                                'Against': home_team,
                                                'Scored': scored,
                                                'Date': date,
                                                'PlayerID': shot['class'][0],
                                                'PlayerName': shot['class'][0],
                                                'Header': 1})

            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]/span').click()
            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[6]/span').click()

            soup = BeautifulSoup(driver.page_source, 'lxml')

            shots = soup.find_all('g')

            for shot in shots:
                if shot.circle['r'] == '6.5':
                    if shot.circle.next_sibling['fill'] == '#333333':
                        scored = 'Scored'
                    else:
                        scored = 'Missed'
                    if float(shot.circle['cx']) > 240:
                        temp_shots_data.append({'x': 480 - float(shot.circle['cx']),
                                                'y': shot.circle['cy'],
                                                'Team': home_team,
                                                'Against': away_team,
                                                'Scored': scored,
                                                'Date': date,
                                                'PlayerID': shot['class'][0],
                                                'PlayerName': shot['class'][0],
                                                'Header': 0})
                    else:
                        temp_shots_data.append({'x': shot.circle['cx'],
                                                'y': shot.circle['cy'],
                                                'Team': away_team,
                                                'Against': home_team,
                                                'Scored': scored,
                                                'Date': date,
                                                'PlayerID': shot['class'][0],
                                                'PlayerName': shot['class'][0],
                                                'Header': 0})
        except Exception as e:
            print(e)
            temp_shots_data = []
            pass

        driver.quit()
        time.sleep(10)

    shots_data_df = pd.DataFrame(temp_shots_data)

    mappings = pd.read_csv('./Team mappings/all_leagues.csv', index_col=1, header=None)
    shots_data_df.replace(mappings[0], inplace=True)

    return shots_data_df

def calc_goals_scored(shot_data, home_team, away_team):

    shot_data_team = shot_data[shot_data['Team']==home_team]
    try:
        home_goals = shot_data_team['Scored'].value_counts()['Scored']
    except KeyError:
        home_goals = 0

    shot_data_team = shot_data[shot_data['Team'] == away_team]
    try:
        away_goals = shot_data_team['Scored'].value_counts()['Scored']
    except KeyError:
        away_goals = 0

    return home_goals, away_goals

def plot_data(shot_data, home_team, away_team, match_date):
    shot_data = tidy_and_format_data(shot_data)  # add xG values to each shot and a colour etc
    img = imread('./xG Plots/background3.jpg')
    shot_data['Colour'] = shot_data['Colour'].replace({'r': 'white', 'b': 'magenta'})
    shot_data['y'] = -1 * shot_data['y']
    xG_home = shot_data.loc[shot_data['Team'] == home_team, 'Proba_exp'].sum()
    xG_away = shot_data.loc[shot_data['Team'] == away_team, 'Proba_exp'].sum()

    home_goals, away_goals = calc_goals_scored(shot_data, home_team, away_team)

    for index, row in shot_data.iterrows():
        if row['Team'] == away_team:
            shot_data.set_value(index, 'x', 480 - row['x'])

    shot_data.sort_values(by=['Scored'], ascending=True, inplace=True)

    plt.scatter(shot_data[shot_data['Scored']=='Missed']['x'], shot_data[shot_data['Scored']=='Missed']['y'],
                s=shot_data[shot_data['Scored']=='Missed']['Proba_exp'] * 400, facecolors=shot_data[shot_data['Scored']=='Missed']['Colour'],
                alpha=0.6, edgecolors='black', linewidth=0.4)
    plt.scatter(shot_data[shot_data['Scored'] == 'Scored']['x'], shot_data[shot_data['Scored'] == 'Scored']['y'],
                s=shot_data[shot_data['Scored'] == 'Scored']['Proba_exp'] * 400,
                facecolors=shot_data[shot_data['Scored'] == 'Scored']['Colour'],
                alpha=1.0, edgecolors='black', linewidth=0.4)
    plt.ylim(-366 / 2, 366 / 2)
    plt.xlim(-10, 490)
    plt.imshow(img, zorder=0, extent=[0, 480, -366 / 2, 366 / 2])
    text = plt.text(240, 120, home_team + ' vs. ' + away_team + '\n'
                    + 'Score: ' + str(home_goals) + '  -  ' + str(away_goals) + '\n'
                    + 'xG: ' + str(np.round(xG_home, 2)) + '  -  ' + str(np.round(xG_away, 2)),
                    horizontalalignment='center',
                    color='white', alpha=0.8, fontsize=10, fontweight='bold')
    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                           path_effects.Normal()])

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                        wspace=None, hspace=None)
    plt.axis('off')
    plt.savefig('./xG Plots/Other Plots/' + home_team + '-' + away_team + ' ' + match_date + '.png',
                bbox_inches=0, pad_inches=0)

def print_teams_associated_with_league(league):
    if league == 'BPL':
        file_path = './Football-data.co.uk/E0/17-18.csv'
    elif league == 'La Liga':
        file_path = './Football-data.co.uk/SP1/SP1.csv'
    elif league == 'Bundesliga':
        file_path = './Football-data.co.uk/D1/D1.csv'
    elif league == 'Serie A':
        file_path = './Football-data.co.uk/I1/I1.csv'
    elif league == 'Ligue 1':
        file_path = './Football-data.co.uk/F1/F1.csv'
    elif league == 'Champions League':
        file_path = './Football-data.co.uk/SP1/SP1.csv'
    elif league == 'Championship':
        file_path = './Football-data.co.uk/E1/17-18.csv'

    data = pd.read_csv(file_path)
    teams = list(set(data.iloc[:, 2:4].values.T.ravel()))
    teams.sort()
    for i in range(0, len(teams)):
        print(teams[i])
    print('\n')

def main():
    print('Options for league...\n BPL  Champions League  La Liga  Bundesliga  Ligue 1  Championship  Serie A  Europa')
    time.sleep(1)
    league = input('Enter league: ')

    print_teams_associated_with_league(league)
    time.sleep(1)

    home_team = input('Enter home team: ')
    away_team = input('Enter away team: ')

    print('Enter the date as dd-mm-yyyy or Today if the match was today.')
    time.sleep(1)
    match_date = input('Enter date: ')
    if match_date == 'Today':
        match_date = datetime.datetime.today().strftime('%d-%m-%Y')

    shot_data = get_shots(home_team, away_team, date=match_date, league=league)
    shot_data.to_csv('./xG Plots/Other Plots/temp.csv')
    shot_data = pd.read_csv('./xG Plots/Other Plots/temp.csv', index_col=0)
    plot_data(shot_data, home_team, away_team, match_date)

if __name__ == '__main__':
    main()
    