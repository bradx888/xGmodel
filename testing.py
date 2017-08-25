from selenium import webdriver
import time

chromedriver = "/Users/BradleyGrantham/Documents/Chromedriver/chromedriver"
driver = webdriver.Chrome(chromedriver)

driver.get('http://epl.squawka.com/english-premier-league/20-08-2017/spurs-vs-chelsea/matches')

driver.find_element_by_xpath('//*[(@id = "mc-stat-shot")]').click()
driver.find_element_by_xpath('//*[(@id = "team2-select")]').click()
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]/span').click()
time.sleep(5)
driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[7]/span').click()
driver.find_element_by_xpath('//*[@id="mc-pitch-legend"]/ul[2]/li[6]/span').click()
