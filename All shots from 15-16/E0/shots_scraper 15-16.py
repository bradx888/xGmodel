from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd

# missed_data = pd.read_csv('matches_not_scraped.csv')
#
# # missed_data = [200, 217, 359]
#
# missed_data = list(missed_data["MatchNo"])
matches =[]



with open('E0.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		matches.append([row['HomeTeam'], row['AwayTeam'], row['Date']])


chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
driver = webdriver.Chrome(chromedriver)

with open('shots_missedInBigScrape.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'Team', 'Against', 'Scored', 'Match No', 'Date'])
    for i in missed_data:
        driver.get(
            'http://epl.squawka.com/english-barclays-premier-league/' + matches[i][2] + '/' + matches[i][0] + '-vs-' +
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
                        writer.writerow([480 - float(shot.circle['cx']), shot.circle['cy'], matches[i][0], matches[i][1], scored, str(i), matches[i][2]])
                    else:
                        writer.writerow([shot.circle['cx'], shot.circle['cy'], matches[i][1], matches[i][0], scored, str(i), matches[i][2]])
        except Exception:
            pass
driver.quit()