"""
This module is used to scrape job information on MyCareersFuture to monitor trends
"""
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
from datetime import date
import sys
import math

driver = webdriver.Chrome("ChromeDriver_Path/chromedriver")
today = date.today()

keys=[
'data scientist','machine learning engineer','data engineer','quant','crypto',  #jobs
'rio tinto','dyson','resmed','bhp','bp singapore','exxon','lego', 'visa','mastercard','american express',    #large companies      
'dbs','uob','ocbc','standard chartered', 'maybank',  #banks   
'jp morgan','nomura','citibank','goldman sachs','bnp paribas','credit agricole','societe generale','barclays','mizuho', 'credit suisse','ubs','deutsche bank', 'commerzbank', 'mufg', #investment banks    
'amazon web services',  'google asia pacific' ,'microsoft operations','meta', 'facebook singapore' ,'netflix entertainment','databricks asiapac', 'apple south', #tech
'tiktok','bytedance','shopee','lazada', 'grab','gojek','goto', 'tencent',  #asian tech         
'stripe','airwallex','gxs', 'foodpanda',   # tech startups
'paypal','asia wealth platform' ,'endowus singapore'    #fintech
'trafigura','glencore','dreyfus','vitol','gunvor','cargill',  #commodities.
'coinbase','binance','okx','nansen','bybit',  #crypto
'ingensoma', #HFT
'point72','balyasny','qube research & technologies','qrt','millenium management','squarepoint capital','citadel','hrt sg','jtp holdings','drw','tower research capital','alphalab capital','grasshopper pte ltd' # funds
]


search_keys=[]
for search in keys:
    key=""
    for item in search.split(" "):
        if item != search.split(" ")[-1]:
            key += item + "%20"
        else:
            key += item
    search_keys.append(key.lower())

print(search_keys)
# Create empty DataFrame to store the data.
columns = [
    'company'
    'title',
    'location',
    'employment_type',
    'level',
    'category',
    'salary',
    'salary_type',
    'job_link'
]

company_l=[]
title_l=[]
location_l=[]
employment_type_l=[]
level_l=[]
category_l=[]
salary_l=[]
salary_type_l=[]
job_link_l=[]
min_exp_l=[]
min_salary_l=[]
max_salary_l=[]
num_app_l=[]
last_post_l=[]
expiry_l=[]
jd_l=[]
company_info_l=[]


for key in search_keys:
    driver.get(f'https://www.mycareersfuture.sg/search?search={key}&sortBy=new_posting_date&page=0')
    try:
        total_jobs=int(driver.find_element(By.XPATH,'//*[@id="search-result-headers"]/div/div[1]').get_attribute("innerText").split("\n")[0].split(" ")[0])
    except:
        total_jobs=0

    pages= math.ceil(total_jobs/20)
    print(f'key: {key}, jobs : {total_jobs}, pages: {pages}')

    for page in range(pages):
        driver.get(f'https://www.mycareersfuture.sg/search?search={key}&sortBy=new_posting_date&page={page}')
        sleep(2)

        for i in range(20):
            try:
                sleep(3)
                # Extract job title and location.
                job_info=driver.find_element(By.XPATH,f'//*[@id="job-card-{i}"]').get_attribute("innerText").split("\n")
                try:
                    job_link=driver.find_element(By.XPATH,f'//*[@id="job-card-{i}"]/div/a').get_attribute("href")
                except:
                    try:
                        job_link=driver.find_element(By.XPATH,f'//*[@id="job-card-{i}"]/div/div[2]/a').get_attribute("href")
                    except:
                        job_link=None
                        


                company=job_info[0]
                title=job_info[2]
                location=job_info[4]
        
                sleep(3)

                


                driver.get(job_link)
                sleep(4)

                add=driver.find_element(By.XPATH,'//*[@id="address"]').get_attribute("innerText")
                emp_type=driver.find_element(By.XPATH,'//*[@id="employment_type"]').get_attribute("innerText")
                seniority=driver.find_element(By.XPATH,'//*[@id="seniority"]').get_attribute("innerText")
                try:
                    min_exp=driver.find_element(By.XPATH,'//*[@id="min_experience"]').get_attribute("innerText")
                except:
                    min_exp='None'
                job_cat=driver.find_element(By.XPATH,'//*[@id="job-categories"]').get_attribute("innerText")



                min_sal=driver.find_element(By.XPATH,'//*[@id="job-details"]/div[1]/div[3]/div/div/section[2]/div/span[2]/div/span[1]').get_attribute("innerText")
                max_sal=driver.find_element(By.XPATH,'//*[@id="job-details"]/div[1]/div[3]/div/div/section[2]/div/span[2]/div/span[2]').get_attribute("innerText")
                sal_type=driver.find_element(By.XPATH,'//*[@id="job-details"]/div[1]/div[3]/div/div/section[2]/div/span[3]').get_attribute("innerText")


                num_app=driver.find_element(By.XPATH,'//*[@id="num_of_applications"]').get_attribute("innerText")
                last_posted=driver.find_element(By.XPATH,'//*[@id="last_posted_date"]').get_attribute("innerText")
                expiry_date=driver.find_element(By.XPATH,'//*[@id="expiry_date"]').get_attribute("innerText")

                jd=driver.find_element(By.XPATH,'//*[@id="job_description"]').get_attribute("innerText")
                company_info=driver.find_element(By.XPATH,'//*[@id="job-details"]/div[2]/div[1]/div/div/section/section/div[2]').get_attribute("innerText")

                company_l.append(company)
                title_l.append(title)
                location_l.append(add)
                employment_type_l.append(emp_type)
                level_l.append(seniority)
                min_exp_l.append(min_exp)
                category_l.append(job_cat)
                min_salary_l.append(min_sal)
                max_salary_l.append(max_sal)

                salary_type_l.append(sal_type)
                num_app_l.append(num_app)
                last_post_l.append(last_posted)
                expiry_l.append(expiry_date)
                jd_l.append(jd)
                company_info_l.append(company_info)
                job_link_l.append(job_link)

                # Save data in DataFrame.
                driver.back()

                # Terminate the loop at the last page.
            except:
                sleep(4)
    
jobs = pd.DataFrame({
'Company': company_l,
'Title': title_l,
'Location': location_l,
'Employment Type': employment_type_l,
'Level': level_l,
'Minimum Experience': min_exp_l,
'Category': category_l,
'Min Salary': min_salary_l,
'Max Salary': max_salary_l,
'Salary Type': salary_type_l,
'Number of Applications': num_app_l,
'Last Posted': last_post_l,
'Closing Date': expiry_l,
'Job Description' : jd_l,
'Company Information' : company_info_l,
'Link': job_link_l
})

jobs.to_csv(f'../data/mycareersfuture_{today}.csv',index=False)
driver.quit()
