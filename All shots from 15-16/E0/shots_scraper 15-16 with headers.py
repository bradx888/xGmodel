from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import numpy as np
import time
import datetime

def remove_numbers(text):
    return ''.join([x for x in text if not x.isdigit()])


def get_player_ids(page_source, current_player_ids):
    soup = BeautifulSoup(page_source, 'lxml')
    team_lineups = soup.find_all('ul', {'class': 'team-lineup'})
    player_ids = []
    player_names = []
    for team_lineup in team_lineups:
        players = team_lineup.find_all('li', {'class': 'mc-option'})
        for player in players:
            if player['data-pid'] not in current_player_ids:
                player_ids.append(player['data-pid'])
                player_names.append(remove_numbers(player.text))

    return player_ids, player_names


matches =[]

data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/15-16.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')

mappings = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 15-16/E0/mappings.csv', index_col=0, header=None)
data.replace(mappings[1], inplace=True)
for index, row in data.iterrows():
    date = datetime
    matches.append([row['HomeTeam'], row['AwayTeam'], row['Date'].strftime('%d-%m-%Y')])

shots_data = []
list_of_matchnos = [i for i in range(0, len(matches))]
missed_matches = []


while len(list_of_matchnos) != 0:
    chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
    driver = webdriver.Chrome(chromedriver)
    list_of_matchnos = [x for x in list_of_matchnos if x not in missed_matches]
    # list_of_matchnos = new
    for i in list_of_matchnos:
        temp_shots_data = []
        driver.get(
            'http://epl.squawka.com/english-barclays-premier-league/' + matches[i][2] + '/' + matches[i][0] + '-vs-' +
            matches[i][1] + '/matches')
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
        print(len(missed_matches))
    except KeyError:
        pass

    driver.quit()
    time.sleep(60)


    shots_data_df.to_csv('./Raw with headers/shots_' + datetime.datetime.today().strftime('%d-%m-%Y') + '.csv')