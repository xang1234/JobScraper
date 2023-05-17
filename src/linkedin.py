from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium.webdriver.chrome.options import Options
from datetime import date

import sys

for i in range(1, len(sys.argv)):
    print('argument:', i, 'value:', sys.argv[i])

today = date.today()

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')

job_name = "Data Scientist"
country_name = "Singapore"

job_url ="";
for item in job_name.split(" "):
    if item != job_name.split(" ")[-1]:
        job_url = job_url + item + "%20"
    else:

        job_url = job_url + item

country_url ="";
for item in country_name.split(" "):
    if item != country_name.split(" ")[-1]:
        country_url = country_url + item + "%20"
    else:
        country_url = country_url + item

JOB=sys.argv[i]

dict_keyword={

"DS":"https://www.linkedin.com/jobs/search?keywords=Data%2BScientist&location=Singapore&geoId=102454443&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0",
"DE":"https://www.linkedin.com/jobs/search?keywords=Data%2BEngineer&location=Singapore&geoId=102454443&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0",
"ML":"https://www.linkedin.com/jobs/search?keywords=Machine%2BLearning&location=Singapore&geoId=102454443&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0",
"Q":"https://www.linkedin.com/jobs/search?keywords=Quant&location=Singapore&geoId=102454443&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0",
"CRYPTO":"https://www.linkedin.com/jobs/search?keywords=Crypto&location=Singapore&geoId=102454443&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0"

}

#url.format(job_url,country_url)


# Creating a webdriver instance
driver = webdriver.Chrome("ChromeDriver_Path/chromedriver")
# Opening the url we have just defined in our browser
driver.get(dict_keyword[JOB])


#We find how many jobs are offered.
jobs_num = driver.find_element(By.CSS_SELECTOR,"h1>span").get_attribute("innerText")
if len(jobs_num.split(',')) > 1:
    jobs_num = int(jobs_num.split(',')[0])*1000
else:
    jobs_num = int(jobs_num)

jobs_num   = int(jobs_num)
print(jobs_num)
#We create a while loop to browse all jobs. 
i = 2
#while i <= int(jobs_num/2)+1:
while i <= 50:
    #We keep scrollind down to the end of the view.
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    i = i + 1
    print("Current at: ", i, "Percentage at: ", ((i+1)/(int(jobs_num/2)+1))*100, "%",end="\r")
    try:
        #We try to click on the load more results buttons in case it is already displayed.
        infinite_scroller_button = driver.find_element(By.XPATH, ".//button[@aria-label='Load more results']")
        infinite_scroller_button.click()
        time.sleep(2)
    except:
        #If there is no button, there will be an error, so we keep scrolling down.
        time.sleep(8)
        pass


print('done scrolling')

#We get a list containing all jobs that we have found.
job_lists = driver.find_element(By.CLASS_NAME,"jobs-search__results-list")
jobs = job_lists.find_elements(By.TAG_NAME,"li") # return a list

#We declare void list to keep track of all obtaind data.
job_title_list = []
company_name_list = []
location_list = []
date_list = []
job_link_list = []

#We loof over every job and obtain all the wanted info.
for job in jobs:
    #job_title
    job_title = job.find_element(By.CSS_SELECTOR,"h3").get_attribute("innerText")
    job_title_list.append(job_title)
    
    #company_name
    company_name = job.find_element(By.CSS_SELECTOR,"h4").get_attribute("innerText")
    company_name_list.append(company_name)
    
    #location
    location = job.find_element(By.CSS_SELECTOR,"div>div>span").get_attribute("innerText")
    location_list.append(location)
    
    #date
    date = job.find_element(By.CSS_SELECTOR,"div>div>time").get_attribute("datetime")
    date_list.append(date)
    
    #job_link
    job_link = job.find_element(By.CSS_SELECTOR,"a").get_attribute("href")
    job_link_list.append(job_link)


jd = [] #job_description
seniority = []
emp_type = []
job_func = []
job_ind = []
time.sleep(4)
for item in range(len(jobs)):
    
    job_func0=[]
    industries0=[]
    # clicking job to view job details
    
    #__________________________________________________________________________ JOB Link
    
    try: 
        job_click_path = f'/html/body/div/div/main/section/ul/li[{item+1}]'
        job_click = job.find_element(By.XPATH,job_click_path).click()
    except:
        print('error')
        pass
    time.sleep(4)
    #job_click = job.find_element(By.XPATH,'.//a[@class="base-card_full-link"]')
    
    #__________________________________________________________________________ JOB Description
    jd_path = '/html/body/div/div/section/div/div/section/div/div/section/div'
    try:
        jd0 = job.find_element(By.XPATH,jd_path).get_attribute('innerText')
        jd.append(jd0)
    except:
        jd.append(None)
        pass
    
    #__________________________________________________________________________ JOB Seniority
    seniority_path='/html/body/div/div/section/div/div/section/div/ul/li[1]/span'
    
    try:
        seniority0 = job.find_element(By.XPATH,seniority_path).get_attribute('innerText')
        seniority.append(seniority0)
    except:
        seniority.append(None)
        pass

    #__________________________________________________________________________ JOB Time
    emp_type_path='/html/body/div/div/section/div/div/section/div/ul/li[2]/span'
    
    try:
        emp_type0 = job.find_element(By.XPATH,emp_type_path).get_attribute('innerText')
        emp_type.append(emp_type0)
    except:
        emp_type.append(None)
        pass
    
    #__________________________________________________________________________ JOB Function
    function_path='/html/body/div/div/section/div/div/section/div/ul/li[3]/span'
    
    try:
        func0 = job.find_element(By.XPATH,function_path).get_attribute('innerText')
        job_func.append(func0)
    except:
        job_func.append(None)
        pass

    #__________________________________________________________________________ JOB Industry
    industry_path='/html/body/div/div/section/div/div/section/div/ul/li[4]/span'
    
    try:
        ind0 = job.find_element(By.XPATH,industry_path).get_attribute('innerText')
        job_ind.append(ind0)
    except:
        job_ind.append(None)
        pass
    

job_data = pd.DataFrame({
    'Date': date_list,
    'Company': company_name_list,
    'Title': job_title_list,
    'Location': location_list,
    'Description': jd,
    'Level': seniority,
    'Type': emp_type,
    'Function': job_func,
    'Industry': job_ind,
    'Link': job_link_list
})

job_data.to_csv(f'../data/linkedin_{JOB}_{today}.csv',index=False)
driver.quit()
