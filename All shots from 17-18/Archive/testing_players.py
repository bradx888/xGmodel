from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import numpy as np
import time
import datetime

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

chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
driver = webdriver.Chrome(chromedriver)

driver.get('http://epl.squawka.com/english-premier-league/13-08-2017/man-utd-vs-west-ham/matches')

print(get_player_ids(driver.page_source))
