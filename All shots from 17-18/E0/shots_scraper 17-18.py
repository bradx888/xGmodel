from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import numpy as np
import time

matches =[]

data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/17-18.csv')

mappings = pd.read_csv('mappings.csv', index_col=0, header=None)
data.replace(mappings[1], inplace=True)
for index, row in data.iterrows():
    matches.append([row['HomeTeam'], row['AwayTeam'], row['Date'].replace('/', '-')])


chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
driver = webdriver.Chrome(chromedriver)

shots_data = []
list_of_matchnos = [i for i in range(0, len(matches))]
missed_matches = []

while len(list_of_matchnos) != 0:
    list_of_matchnos = [x for x in list_of_matchnos if x not in missed_matches]
    # list_of_matchnos = new
    for i in list_of_matchnos:
        driver.get(
            'http://epl.squawka.com/english-premier-league/' + matches[i][2] + '/' + matches[i][0] + '-vs-' +
            matches[i][1] + '/matches')
        try:

            driver.find_element_by_xpath('//*[(@id = "mc-stat-shot")]').click()
            driver.find_element_by_xpath('//*[(@id = "team2-select")]').click()
            soup = BeautifulSoup(driver.page_source, 'lxml')

            shots = soup.find_all('g')

            for shot in shots:
                if shot.circle['r'] == '6.5':
                    if shot.circle.next_sibling['fill'] == '#333333':
                        scored = 'Scored'
                    else:
                        scored = 'Missed'
                    if float(shot.circle['cx']) > 240:
                        shots_data.append({'x': 480 - float(shot.circle['cx']),
                                   'y': shot.circle['cy'],
                                   'Team': matches[i][0],
                                   'Against': matches[i][1],
                                   'Scored': scored,
                                   'Match No': str(i),
                                   'Date': matches[i][2]})
                    else:
                        shots_data.append({'x': shot.circle['cx'],
                                      'y': shot.circle['cy'],
                                      'Team': matches[i][1],
                                      'Against': matches[i][0],
                                      'Scored': scored,
                                      'Match No': str(i),
                                      'Date': matches[i][2]})
        except Exception:
            pass


    shots_data_df = pd.DataFrame(shots_data)
    missed_matches = list(set(shots_data_df['Match No']))
    missed_matches = [int(x) for x in missed_matches]

driver.quit()

shots_data_df.to_csv('shots.csv')