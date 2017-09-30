'''
as long as the files that this script is dependant on are up to date
this should run automatically on a raspberry pi and email me tips every morning
'''

import numpy as np
import pandas as pd
from selenium import webdriver
from pyvirtualdisplay import Display
import time
from bs4 import BeautifulSoup
import math
from datetime import datetime
from datetime import timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import schedule

def myprob(distance, angle):
    x = distance*np.power((angle+1), 0.5)
    result = 1.0646383882981121*np.exp(-0.0247111*x)
    return result

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
    team_ratings = pd.read_csv('./Team ratings/E0/teamratings_16-17.csv',
                           index_col=0)

    championship_team_ratings = pd.read_csv('./Team ratings/E1/teamratings_16-17.csv',
                                        index_col=0)

    for column in championship_team_ratings:
        if 'Attack' in column:
            championship_team_ratings[column] = promoted_teams_attack(championship_team_ratings[column])
        else:
            championship_team_ratings[column] = promoted_teams_defense(championship_team_ratings[column])

    team_ratings = team_ratings.append(championship_team_ratings)

    ''' the following code updates the team ratings based on this seasons current results'''
    new_data = calculate_this_seasons_ratings()

    new_data = combine_ratings(new_data, team_ratings)

    new_data.to_csv('./Team ratings/E0/teamratings_17-18.csv')
    return new_data

def combine_ratings(new_data, team_ratings, exp_factor=0.08):
    '''
    :param new_data: new_data is the current seasons ratings
    :param team_ratings: team_ratings is last seasons ratings
    :param exp_factor:
    :return:
    '''
    for index, row in new_data.iterrows():
        team_ratings.set_value(index, 'HomeAttack',
                               team_ratings.ix[index, 'HomeAttack'] * np.exp(-exp_factor * row['HomeCount']))
        team_ratings.set_value(index, 'AwayAttack',
                               team_ratings.ix[index, 'AwayAttack'] * np.exp(-exp_factor * row['AwayCount']))
        team_ratings.set_value(index, 'HomeDefense',
                               team_ratings.ix[index, 'HomeDefense'] * np.exp(-exp_factor * row['HomeCount']))
        team_ratings.set_value(index, 'AwayDefense',
                               team_ratings.ix[index, 'AwayDefense'] * np.exp(-exp_factor * row['AwayCount']))

    for index, row in new_data.iterrows():
        new_data.set_value(index, 'HomeAttack', team_ratings.ix[index, 'HomeAttack'] + row['HomeAttack'])
        new_data.set_value(index, 'AwayAttack', team_ratings.ix[index, 'AwayAttack'] + row['AwayAttack'])
        new_data.set_value(index, 'HomeDefense', team_ratings.ix[index, 'HomeDefense'] + row['HomeDefense'])
        new_data.set_value(index, 'AwayDefense', team_ratings.ix[index, 'AwayDefense'] + row['AwayDefense'])

    new_data.drop(['HomeCount',
                   'AwayCount',
                   ], axis=1 ,inplace=True)
    for index, row in team_ratings.iterrows():
        if index not in new_data.index.values:
            new_data.loc[index] = row

    return new_data

def calculate_this_seasons_ratings(exp_factor=0.08):
    data = pd.read_csv(
        './Football-data.co.uk/E0/17-18.csv'
    )
    teams = set(list(data['HomeTeam']) + list(data['AwayTeam']))
    list_of_ratings = ['HomeAttack', 'HomeDefense', 'AwayAttack', 'AwayDefense']
    new_ratings = pd.DataFrame(data=np.zeros(20*6).reshape(20, 6), index=teams, columns=list_of_ratings + ['HomeCount', 'AwayCount'])
    for rating in list_of_ratings:
        temp_data = data.groupby(by=[rating[0:4] + 'Team'], as_index=True)[['xGH', 'xGA',
                                                               'FTHG', 'FTAG',
                                                               'HST', 'AST',
                                                               'HS', 'AS']].mean()

        temp_data['xGH FTHG'] = 0.8 * temp_data['xGH'] + 0.2 * temp_data['FTHG']
        temp_data['xGA FTAG'] = 0.8 * temp_data['xGA'] + 0.2 * temp_data['FTAG']

        X = temp_data.as_matrix(columns=['xGH FTHG',
                                    'xGA FTAG'])


        theta = np.load('./Ratings regression/Parameters/theta_' + rating + '.npy')
        X = np.c_[np.ones(X.shape[0]), X]
        temp_ratings = np.dot(X, theta)
        temp_teams = temp_data.index.values
        temp_dict = dict.fromkeys(temp_teams, 0)
        for i in range(0, len(temp_teams)):
            temp_dict[temp_teams[i]] = temp_ratings[i]
        for key, value in temp_dict.items():
            for index, row in new_ratings.iterrows():
                if index == key:
                    new_ratings.set_value(index, rating, temp_dict[index])


    # need to apply exponential function to these
    home_games_played = data.groupby(by=['HomeTeam'], as_index=True)[['xGH']].count()
    home_games_played = home_games_played.to_dict()['xGH']
    away_games_played = data.groupby(by=['AwayTeam'], as_index=True)[['xGA']].count()
    away_games_played = away_games_played.to_dict()['xGA']
    for key, value in home_games_played.items():
        for index, row in new_ratings.iterrows():
            if index == key:
                new_ratings.set_value(index, 'HomeCount', home_games_played[index])
    for key, value in away_games_played.items():
        for index, row in new_ratings.iterrows():
            if index == key:
                new_ratings.set_value(index, 'AwayCount', away_games_played[index])

    new_ratings['HomeAttack'] = (1 - np.exp(-exp_factor * new_ratings['HomeCount'])) * new_ratings['HomeAttack']
    new_ratings['HomeDefense'] = (1 - np.exp(-exp_factor * new_ratings['HomeCount'])) * new_ratings['HomeDefense']
    new_ratings['AwayAttack'] = (1 - np.exp(-exp_factor * new_ratings['AwayCount'])) * new_ratings['AwayAttack']
    new_ratings['AwayDefense'] = (1 - np.exp(-exp_factor * new_ratings['AwayCount'])) * new_ratings['AwayDefense']

    return new_ratings

def read_in_fixtures(todays_schedule):
    if todays_schedule is not None:
        data = pd.read_csv(
            './Fixtures/E0/Remaining 17-18 Fixtures.csv',
            index_col=0)
        to_drop = []
        for index, row in todays_schedule.iterrows():
            for index1, row1 in data.iterrows():
                if row['HomeTeam'] == row1['HomeTeam'] and row['AwayTeam'] == row1['AwayTeam']:
                    to_drop.append(index1)
        data.drop(to_drop, axis=0, inplace=True)
        data.to_csv('./Fixtures/E0/Remaining 17-18 Fixtures.csv')
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
        data = data[data.Date < datetime.today() + timedelta(days=7)]
        return data
    else:
        data = pd.read_csv(
            './Fixtures/E0/Remaining 17-18 Fixtures.csv',
            index_col=0)
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
        data = data[data.Date < datetime.today() + timedelta(days=7)]
        # data.to_csv('./Fixtures/E0/Remaining 17-18 Fixtures.csv')
        return data

def get_flashscores_schedule():
    my_url = 'http://www.flashscores.co.uk/football/'

    # initialise chromedriver
    chromedriver = "./Chromedriver/chromedriver"
    driver = webdriver.Chrome(chromedriver)
    # display = Display(visible=0, size=(1024,768))
    # display.start()
    # driver = webdriver.Firefox()
    driver.get(my_url)
    # time.sleep(1)
    htmlSource = driver.page_source
    driver.quit()
    # display.stop()

    soup = BeautifulSoup(htmlSource, 'lxml')
    # find all table headers
    t_heads = soup.find_all('thead')

    # check what tournament the thead corresponds to and check it matches the one we want
    for t_head in t_heads:
        if t_head.find_all('span', {'class', 'tournament_part'})[0].text == 'Premier League' \
                and t_head.find_all('span', {'class', 'country_part'})[0].text == 'ENGLAND: ':
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
        chromedriver = "./Chromedriver/chromedriver"
        driver = webdriver.Chrome(chromedriver)
        # display = Display(visible=0, size=(1024,768))
        # display.start()
        # driver = webdriver.Firefox()
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
            # print(my_url)
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
            if row['HomeTeam'] == 'Man Utd':
                home_attack = team_ratings.loc['Man United']['HomeAttack']
                home_defense = team_ratings.loc['Man United']['HomeDefense']
                away_attack = team_ratings.loc[row['AwayTeam']]['AwayAttack']
                away_defense = team_ratings.loc[row['AwayTeam']]['AwayDefense']
            elif row['AwayTeam'] == 'Man Utd':
                home_attack = team_ratings.loc[row['HomeTeam']]['HomeAttack']
                home_defense = team_ratings.loc[row['HomeTeam']]['HomeDefense']
                away_attack = team_ratings.loc['Man United']['AwayAttack']
                away_defense = team_ratings.loc['Man United']['AwayDefense']
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
        # display.stop()
        # convert oddschecker team names back to FD team names
        mappings = pd.read_csv(
            './Team mappings/E0/footballdata_to_oddschecker.csv',
            index_col=1, header=None)
        #schedule.replace(mappings[0], inplace=True)
        return schedule
    else:
        return None

def get_tips(schedule, boolean):
    if schedule is not None:
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
        if boolean == '7days':
            tips.to_csv('./Tips/7 Day Forecast/' + datetime.today().strftime("%Y-%m-%d") + '.csv')
        else:
            tips.to_csv('./Tips/Daily/'+ datetime.today().strftime("%Y-%m-%d") + '.csv')
        #return tips
    else:
        return None

def send_tips():
    gmail_username = "automatedtransactionsbg@gmail.com"
    gmail_password = "Z1s$ASuo#2NE"
    toaddr = 'bradley.grantham@bath.edu'
    msg = MIMEMultipart()

    # set the to and from address as well as the subject
    msg['From'] = gmail_username
    msg['To'] = toaddr
    msg['Subject'] = 'Todays PL Tips for ' + datetime.today().strftime("%Y-%m-%d") + '\n\n'

    # write the number of days
    body = 'Bet wisely, young one'
    msg.attach(MIMEText(body, 'plain'))

    # attach a csv of the transactions

    filename = './Tips/Daily/'+ datetime.today().strftime("%Y-%m-%d") + '.csv'
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
    toaddr = 'bradley.grantham@bath.edu'
    msg = MIMEMultipart()

    # set the to and from address as well as the subject
    msg['From'] = gmail_username
    msg['To'] = toaddr
    msg['Subject'] = 'PL Tips for next 7 days' + '\n\n'

    # write the number of days
    body = 'Bet wisely, young one'
    msg.attach(MIMEText(body, 'plain'))

    # attach a csv of the transactions

    filename = './Tips/7 Day Forecast/' + datetime.today().strftime("%Y-%m-%d") + '.csv'
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
    todays_schedule = None #get_flashscores_schedule()
    next7days_schedule = read_in_fixtures(todays_schedule)
    team_ratings = read_in_team_ratings()
    next7days_schedule = get_corresponding_odds(next7days_schedule, team_ratings)
    get_tips(next7days_schedule, '7days')
    send_all()
    if todays_schedule is not None:
        todays_schedule = get_corresponding_odds(todays_schedule, team_ratings)
        get_tips(todays_schedule, 'Today')
        send_tips()
    else:
        pass


schedule.every().day.at("06:00").do(main)
# schedule.every(15).minutes.do(main)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

main()

