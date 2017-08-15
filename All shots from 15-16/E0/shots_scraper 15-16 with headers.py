from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import numpy as np
import time
import datetime

matches =[]

data = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/Football-data.co.uk/E0/15-16.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')

mappings = pd.read_csv('/Users/BradleyGrantham/Documents/Python/FootballPredictions/xG model/All shots from 15-16/E0/mappings.csv', index_col=0, header=None)
data.replace(mappings[1], inplace=True)
for index, row in data.iterrows():
    date = datetime
    matches.append([row['HomeTeam'], row['AwayTeam'], row['Date'].strftime('%d-%m-%Y')])

chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
driver = webdriver.Chrome(chromedriver)

shots_data = []
list_of_matchnos = [i for i in range(0, len(matches))]
missed_matches = []

while len(list_of_matchnos) != 0:
    list_of_matchnos = [x for x in list_of_matchnos if x not in missed_matches]
    # list_of_matchnos = new
    for i in range(0, 3):
        driver.get(
            'http://epl.squawka.com/english-barclays-premier-league/' + matches[i][2] + '/' + matches[i][0] + '-vs-' +
            matches[i][1] + '/matches')
        driver.find_element_by_xpath('//*[(@id = "mc-stat-shot")]').click()
        driver.find_element_by_xpath('//*[(@id = "team2-select")]').click()
        time.sleep(2)
        driver.find_element_by_xpath("/html[@class=' pw-locale-en ra1-pw-desktop']/body/div[@id='sq-overflow-container']/div[@id='sq-mc-outer']/div[@id='sq-mc-outer2']/div[@id='sq-mc-container']/div[@id='mc-content-wrap']/div[@id='mc-main-container']/div[@id='fullstats-conatiner']/div[@id='mc-pitch-container']/div[@id='mc-pitch-legend']/ul[@class='pitch-legend-list shot-leg stat-selected']/li[@class='legend6 show'][3]").click()
        try:

            driver.find_element_by_xpath('//*[(@id = "mc-stat-shot")]').click()
            driver.find_element_by_xpath('//*[(@id = "team2-select")]').click()
            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]').click()
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
                                           'Date': matches[i][2],
                                           'Headed': 'Y'})
                    else:
                        shots_data.append({'x': shot.circle['cx'],
                                           'y': shot.circle['cy'],
                                           'Team': matches[i][1],
                                           'Against': matches[i][0],
                                           'Scored': scored,
                                           'Match No': str(i),
                                           'Date': matches[i][2],
                                           'Headed': 'Y'})
        except Exception:
            pass

        try:
            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]').click()
            driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[6]').click()
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
                                   'Date': matches[i][2],
                                           'Headed': 'N'})
                    else:
                        shots_data.append({'x': shot.circle['cx'],
                                      'y': shot.circle['cy'],
                                      'Team': matches[i][1],
                                      'Against': matches[i][0],
                                      'Scored': scored,
                                      'Match No': str(i),
                                      'Date': matches[i][2],
                                           'Headed': 'N'})
        except Exception:
            pass


    shots_data_df = pd.DataFrame(shots_data)
    missed_matches = list(set(shots_data_df['Match No']))
    missed_matches = [int(x) for x in missed_matches]

driver.quit()

shots_data_df.to_csv('shots_with_headers.csv')

