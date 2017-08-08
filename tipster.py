import numpy as np
import pandas as pd
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import math
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import schedule

def promoted_teams_attack(old_param):
    parameter = 0.53683956207924333 * old_param + 0.049598077388848139
    return parameter

def promoted_teams_defense(old_param):
    parameter = -0.88138228873841129 * old_param + 2.2188585603647821
    return parameter

def pois(x, lambdaa):
    result = (np.power(lambdaa, x) * np.exp(-lambdaa)) /math.factorial(x)
    return result

def tau(x, y, lambdaa, mu, rho):
    if x == 0 and y == 0:
        result = 1 - (lambdaa*mu*rho)
    elif x == 0 and y == 1:
        result = 1 + (lambdaa*rho)
    elif  x == 1 and y == 0:
        result = 1 + (mu*rho)
    elif x == 1 and y == 1:
        result = 1 - rho
    else:
        result = 1
    return result

def bivpois2(lambdaa, mu, rho):
    max_goals=6
    weights = []
    population = []
    for i in range(0, max_goals+1):
        for j in range(0, max_goals+1):
            population.append([i,j])
            weights.append(tau(i, j, lambdaa, mu, rho)*pois(i, lambdaa)*pois(j, mu))
    return population, weights

def predictor2(population, weights):
    new = random.choices(population, weights)
    return new

def calculate_win_perc(population, weights):
    homeperc, drawperc, awayperc = 0, 0, 0
    for i in range(0, len(population)):
        if population[i][0] > population[i][1]:
            homeperc += weights[i]
        elif population[i][0] == population[i][1]:
            drawperc += weights[i]
        elif population[i][0] < population[i][1]:
            awayperc += weights[i]
    return [homeperc, drawperc, awayperc]

def read_in_team_ratings():
    team_ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17_dixoncoles-moretol.csv',
                           index_col=0)

    championship_team_ratings = pd.read_csv('./Team ratings/E1/championship_teamratings_16-17.csv',
                                        index_col=0)

    for column in championship_team_ratings:
        if 'Attack' in column:
            championship_team_ratings[column] = promoted_teams_attack(championship_team_ratings[column])
        else:
            championship_team_ratings[column] = promoted_teams_defense(championship_team_ratings[column])

    team_ratings = team_ratings.append(championship_team_ratings)

    return team_ratings

def get_flashscores_schedule():
    my_url = 'http://www.flashscores.co.uk/football/'

    # initialise chromedriver
    chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
    driver = webdriver.Chrome(chromedriver)
    driver.get(my_url)
    # time.sleep(1)
    htmlSource = driver.page_source
    driver.quit()

    soup = BeautifulSoup(htmlSource, 'lxml')
    # find all table headers
    t_heads = soup.find_all('thead')
    what_league = 'Premier League'

    # check what tournament the thead corresponds to and check it matches the one we want
    for t_head in t_heads:
        if t_head.find_all('span', {'class', 'tournament_part'})[0].text == what_league:
            matches = t_head.next_sibling

    # use a try because there may not be any games today
    try:
        # find all the times and teams on that day in that particular league
        times = matches.find_all('td', {'class': 'cell_ad time '})
        home_team = matches.find_all('span', {'class': 'padr'})
        away_team = matches.find_all('span', {'class': 'padl'})

        # take just the text of each part
        for i in range(0, len(times)):
            times[i] = times[i].text
            home_team[i] = home_team[i].text
            away_team[i] = away_team[i].text

        # convert to numpy array
        times = np.array(times)
        home_team = np.array(home_team)
        away_team = np.array(away_team)

        # put into a dataframe and write to a csv
        schedule = pd.DataFrame({'KO': times, 'HomeTeam': home_team, 'AwayTeam': away_team})

        # convert FlashScores team names to Football Data team names
        mappings = pd.read_csv('./Team mappings/E0/flashscores_to_footballdata.csv',
                               index_col=0, header=None)
        schedule.replace(mappings[1], inplace=True)
    except NameError:
        schedule = None
    return schedule

def get_corresponding_odds(schedule, team_ratings):
    '''
    Need to find a way to also collect which
    bookie is offering the best price
    '''
    if schedule is not None:
        chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
        driver = webdriver.Chrome(chromedriver)
        # convert FootballData team names to oddschecker names
        schedule['AwayTeam_OddsName'] = schedule['AwayTeam']
        schedule['HomeTeam_OddsName'] = schedule['HomeTeam']
        mappings = pd.read_csv('./Team mappings/E0/footballdata_to_oddschecker.csv',
                               index_col=0, header=None)
        schedule['AwayTeam_OddsName'].replace(mappings[1], inplace=True)
        schedule['HomeTeam_OddsName'].replace(mappings[1], inplace=True)
        schedule.replace('Man United', 'Man Utd', inplace=True)
        for index, row in schedule.iterrows():
            my_url = 'https://www.oddschecker.com/football/english/premier-league/' + row['HomeTeam_OddsName'] + '-v-' + row[
                'AwayTeam_OddsName'] + '/winner'
            driver.get(my_url)
            try:
                driver.find_element_by_xpath('//*[@id="promo-modal"]/div[1]/div/span').click()
            except:
                pass
            html_source = driver.page_source
            soup = BeautifulSoup(html_source, 'lxml')
            home_win = soup.find('tr', {'data-bname': row['HomeTeam']})
            schedule.set_value(index, 'Best_Bookie_H', home_win['data-best-bks'])
            draw = soup.find('tr', {'data-bname': 'Draw'})
            schedule.set_value(index, 'Best_Bookie_D', draw['data-best-bks'])
            away_win = soup.find('tr', {'data-bname': row['AwayTeam']})
            schedule.set_value(index, 'Best_Bookie_A', away_win['data-best-bks'])
            schedule.set_value(index, 'Bookies_H', 1 / float(home_win['data-best-dig']))
            schedule.set_value(index, 'Bookies_A', 1 / float(away_win['data-best-dig']))
            schedule.set_value(index, 'Bookies_D', 1 / float(draw['data-best-dig']))
            if row['HomeTeam'] == 'Man Utd' or row['AwayTeam'] == 'Man Utd':
                home_attack = team_ratings.loc['Man United']['HomeAttack']
                home_defense = team_ratings.loc['Man United']['HomeDefense']
                away_attack = team_ratings.loc[row['AwayTeam']]['AwayAttack']
                away_defense = team_ratings.loc[row['AwayTeam']]['AwayDefense']
            else:
                home_attack = team_ratings.loc[row['HomeTeam']]['HomeAttack']
                home_defense = team_ratings.loc[row['HomeTeam']]['HomeDefense']
                away_attack = team_ratings.loc[row['AwayTeam']]['AwayAttack']
                away_defense = team_ratings.loc[row['AwayTeam']]['AwayDefense']

            population, weights = bivpois2(home_attack * away_defense, away_attack * home_defense, 0.15)

            percentages = calculate_win_perc(population, weights)
            schedule.set_value(index, 'Brad_H', percentages[0])
            schedule.set_value(index, 'Brad_A', percentages[2])
            schedule.set_value(index, 'Brad_D', percentages[1])

        driver.quit()
        # convert oddschecker team names back to FD team names
        mappings = pd.read_csv(
            './Team mappings/E0/footballdata_to_oddschecker.csv',
            index_col=1, header=None)
        #schedule.replace(mappings[0], inplace=True)
        return schedule
    else:
        return None

def get_tips(schedule):
    if schedule is not None:
        schedule.to_csv('./Tips/'+ 'ALL__' + datetime.today().strftime("%Y-%m-%d") + '.csv')
        selection, my_odds, b365_odds, home_team, away_team, bookie = [], [], [], [], [], []
        for index, row in schedule.iterrows():
            if row['Brad_H'] > (row['Bookies_H']) and 1 / row['Brad_H'] < 2.8:
                selection.append('Home Winner')
                my_odds.append(1 / row['Brad_H'])
                b365_odds.append(1 / row['Bookies_H'])
                home_team.append(row['HomeTeam'])
                away_team.append(row['AwayTeam'])
                bookie.append(row['Best_Bookie_H'])
            if row['Brad_D'] > (row['Bookies_D']) and 1 / row['Brad_D'] < 2.8:
                selection.append('Draw')
                my_odds.append(1 / row['Brad_D'])
                b365_odds.append(1 / row['Bookies_D'])
                home_team.append(row['HomeTeam'])
                away_team.append(row['AwayTeam'])
                bookie.append(row['Best_Bookie_D'])
            if row['Brad_A'] > (row['Bookies_A']) and 1 / row['Brad_A'] < 2.8:
                selection.append('Away Winner')
                my_odds.append(1 / row['Brad_A'])
                b365_odds.append(1 / row['Bookies_A'])
                home_team.append(row['HomeTeam'])
                away_team.append(row['AwayTeam'])
                bookie.append(row['Best_Bookie_A'])

        tips = pd.DataFrame({'Selection': selection, 'MyOdds': my_odds,
                             'Bookies Odds': b365_odds, 'HomeTeam': home_team,
                             'AwayTeam': away_team, 'Bookie': bookie}).reindex_axis(['HomeTeam', 'AwayTeam', 'Selection',
                                                                   'MyOdds', 'Bookies Odds', 'Bookie'], axis=1)
        tips['MyOdds'] = np.round(tips['MyOdds'], decimals=2)
        tips.to_csv('./Tips/'+ datetime.today().strftime("%Y-%m-%d") + '.csv')
        return tips
    else:
        return None

def send_tips():
    gmail_username = "automatedtransactionsbg@gmail.com"
    gmail_password = "Z1s$ASuo#2NE"
    toaddr = 'bgrantham343@gmail.com'
    msg = MIMEMultipart()

    # set the to and from address as well as the subject
    msg['From'] = gmail_username
    msg['To'] = toaddr
    msg['Subject'] = 'Premier League Tips for ' + datetime.today().strftime("%Y-%m-%d") + '\n\n'

    # write the number of days
    body = 'Bet wisely, young one'
    msg.attach(MIMEText(body, 'plain'))

    # attach a csv of the transactions

    filename = './Tips/'+ datetime.today().strftime("%Y-%m-%d") + '.csv'
    f = open(filename, 'r')
    attachment = MIMEText(f.read())
    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(attachment)

    # set up the server, send the email and then quit
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()

    server.login(gmail_username, gmail_password)

    server.sendmail(gmail_username, toaddr, msg.as_string())
    server.quit()

def send_all():
    gmail_username = "automatedtransactionsbg@gmail.com"
    gmail_password = "Z1s$ASuo#2NE"
    toaddr = 'bgrantham343@gmail.com'
    msg = MIMEMultipart()

    # set the to and from address as well as the subject
    msg['From'] = gmail_username
    msg['To'] = toaddr
    msg['Subject'] = 'Premier League Tips for ' + datetime.today().strftime("%Y-%m-%d") + '\n\n'

    # write the number of days
    body = 'Bet wisely, young one'
    msg.attach(MIMEText(body, 'plain'))

    # attach a csv of the transactions

    filename = './Tips/' + 'ALL__' + datetime.today().strftime("%Y-%m-%d") + '.csv'
    f = open(filename, 'r')
    attachment = MIMEText(f.read())
    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(attachment)

    # set up the server, send the email and then quit
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()

    server.login(gmail_username, gmail_password)

    server.sendmail(gmail_username, toaddr, msg.as_string())
    server.quit()

def main():
    schedule = pd.read_csv('schedule.csv', index_col=0)
    team_ratings = read_in_team_ratings()
    #schedule = get_flashscores_schedule()
    if schedule is not None:
        schedule = get_corresponding_odds(schedule, team_ratings)
        get_tips(schedule)
        send_tips()
        send_all()
        schedule.to_csv('testing.csv')
    else:
        pass

# schedule.every().day.at("19:55").do(main)
#
# while True:
#     schedule.run_pending()
#     time.sleep(1)

main()