from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import numpy as np
import time
import datetime
import os
import glob


def download_and_update_football_data():
    chromedriver = "/Users/bgrantham/Documents/Personal/xGmodel/Chromedriver/chromedriver"
    driver = webdriver.Chrome(chromedriver)
    driver.get("http://www.football-data.co.uk/mmz4281/1718/E0.csv")
    time.sleep(6)
    driver.quit()
    most_recent_download = max(glob.iglob('/Users/bgrantham/Downloads/*.csv'), key=os.path.getctime)
    data = pd.read_csv(most_recent_download)
    data.to_csv(
        '/Users/bgrantham/Documents/Personal/xGmodel/Football-data.co.uk/E0/17-18.csv'
    )


def remove_numbers(text):
    return ''.join([x for x in text if not x.isdigit()])


def get_player_ids(page_source):
    soup = BeautifulSoup(page_source, 'lxml')
    team_lineups = soup.find_all('ul', {'class': 'team-lineup'})
    player_ids = []
    player_names = []
    for team_lineup in team_lineups:
        players = team_lineup.find_all('li', {'class': 'mc-option'})
        for player in players:
            player_ids.append(player['data-pid'])
            player_names.append(remove_numbers(player.text))

    return player_ids, player_names

def get_new_match_nos():
    start_match_no = pd.read_csv(
        'shots.csv',
    usecols=['Match No'])
    start_match_no = list(set(start_match_no['Match No']))
    if len(start_match_no)==0:
        start_match_no = 0
    else:
        start_match_no = max(start_match_no) + 1

    end_match_no = pd.read_csv(
        '/Users/bgrantham/Documents/Personal/xGmodel/Football-data.co.uk/E0/17-18.csv')
    end_match_no = len(end_match_no['HomeTeam'])
    return list(range(start_match_no, end_match_no))

def combine_all():
    direc = './Raw/'
    files = os.listdir(direc)

    for i in range(0, len(files)):
        if i == 0:
            data = pd.read_csv(direc + files[i], index_col=0)
        else:
            new_data = pd.read_csv(direc + files[i], index_col=0)
            data = pd.concat([data, new_data], ignore_index=True)

    data.to_csv(
        'shots.csv'
    )


'''

Start of main program is here

'''

download_and_update_football_data()


matches =[]

data = pd.read_csv('/Users/bgrantham/Documents/Personal/xGmodel/Football-data.co.uk/E0/17-18.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')

mappings = pd.read_csv('mappings.csv', index_col=0, header=None)
data.replace(mappings[1], inplace=True)
for index, row in data.iterrows():
    date = datetime
    matches.append([row['HomeTeam'], row['AwayTeam'], row['Date'].strftime('%d-%m-%Y')])

chromedriver = "./Chromedriver/chromedriver"
driver = webdriver.Chrome(chromedriver)

shots_data = []
list_of_matchnos = get_new_match_nos()
missed_matches = []
player_ids = []
player_names = []


while len(list_of_matchnos) != 0:
    list_of_matchnos = [x for x in list_of_matchnos if x not in missed_matches]
    # list_of_matchnos = new
    chromedriver = "/Users/bgrantham/Documents/Personal/xGmodel/Chromedriver/chromedriver"
    driver = webdriver.Chrome(chromedriver)
    for i in list_of_matchnos:
        temp_shots_data = []
        driver.get(
            'http://epl.squawka.com/english-premier-league/' + matches[i][2] + '/' + matches[i][0] + '-vs-' +
            matches[i][1] + '/matches')
        try:

            driver.find_element_by_xpath('//*[(@id = "mc-stat-shot")]').click()
            driver.find_element_by_xpath('//*[(@id = "team2-select")]').click()
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]/span').click()

            new_player_ids, new_player_names = get_player_ids(driver.page_source)
            player_ids = player_ids + new_player_ids
            player_names = player_names + new_player_names

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
                                   'Team': matches[i][0],
                                   'Against': matches[i][1],
                                   'Scored': scored,
                                   'Match No': str(i),
                                   'Date': matches[i][2],
                                    'PlayerID': shot['class'][0],
                                           'PlayerName': shot['class'][0],
                                           'Header': 1})
                    else:
                        temp_shots_data.append({'x': shot.circle['cx'],
                                      'y': shot.circle['cy'],
                                      'Team': matches[i][1],
                                      'Against': matches[i][0],
                                      'Scored': scored,
                                      'Match No': str(i),
                                      'Date': matches[i][2],
                                           'PlayerID': shot['class'][0],
                                           'PlayerName': shot['class'][0],
                                           'Header':1})

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
                                           'Team': matches[i][0],
                                           'Against': matches[i][1],
                                           'Scored': scored,
                                           'Match No': str(i),
                                           'Date': matches[i][2],
                                           'PlayerID': shot['class'][0],
                                           'PlayerName': shot['class'][0],
                                           'Header': 0})
                    else:
                        temp_shots_data.append({'x': shot.circle['cx'],
                                           'y': shot.circle['cy'],
                                           'Team': matches[i][1],
                                           'Against': matches[i][0],
                                           'Scored': scored,
                                           'Match No': str(i),
                                           'Date': matches[i][2],
                                           'PlayerID': shot['class'][0],
                                           'PlayerName': shot['class'][0],
                                           'Header': 0})
        except Exception:
            temp_shots_data = []
            pass

        shots_data.extend(temp_shots_data)

    shots_data_df = pd.DataFrame(shots_data)
    try:
        missed_matches = list(set(shots_data_df['Match No']))
        missed_matches = [int(x) for x in missed_matches]
        print(missed_matches)
    except KeyError:
        pass

    driver.quit()
    time.sleep(60)

player_ids = pd.Series(data=player_names, index=player_ids)
shots_data_df['PlayerName'].replace(player_ids, inplace=True)

shots_data_df.to_csv('./Raw/shots_' + datetime.datetime.today().strftime('%d-%m-%Y') + '.csv')

combine_all()