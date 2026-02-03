"""
This module is used to srape data science job information on MyCareersFuture.

reference for async scraping https://oxylabs.io/blog/asynchronous-web-scraping-python-aiohttp

"""
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import date
import time
import sys
import math
import asyncio
import aiohttp as aiohttp 
from arsenic import get_session
from arsenic.browsers import Chrome
from arsenic.services import  Chromedriver
import json
from tqdm import tqdm

SEM=2
today = date.today()


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
# Loop through 100 pages.


async def save_data(key):
    pass


async def save_product(job, job_info):
    json_file_name = job.replace(' ', '_')
    with open(f'../data/scrape_jsons/{json_file_name}.json', 'w') as json_file_name:
        json.dump(job_info, json_file_name)

async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

async def get_pages(url):
    service = Chromedriver(binary='/Users/admin/Downloads/chromedriver-mac-x64/chromedriver')
    browser = Chrome()
    browser.capabilities = {
        "goog:chromeOptions": {"args": ["--headless", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]}
    }

    async with get_session(service,browser) as driver:
        await driver.get(url)
        #job-details > div.w-70-l.w-60-ms.w-100.pr2-l.pr2-ms.relative > div:nth-child(3) > div > section:nth-child(1) > p
        try:
            company = await driver.get_elements("#job-details > div.w-70-l.w-60-ms.w-100.pr2-l.pr2-ms.relative > div:nth-child(3) > div > section:nth-child(1) > p")
            company = await company[0].get_text()
        except:
            company=""

        try:
            title = await driver.get_elements("#job_title")
            title = await title[0].get_text()
        except:
            title=""
        
        try:
            location= await driver.get_elements("#scroll_to_location_map")
            location= await location[0].get_text()
        except:
            location=""
        try:
            employment_type= await driver.get_elements("#employment_type")
            employment_type= await employment_type[0].get_text()
        except:
            employment_type=""

        try:
            level = await driver.get_elements("#seniority")
            level = await level[0].get_text()
        except:
            level=""

        try:
            min_exp = await driver.get_elements("#min_experience")
            min_exp = await min_exp[0].get_text()
        except:
            min_exp =""
        try:
            category = await driver.get_elements("#job-categories")
            category = await category[0].get_text()
        except:
            category=""

        try:
            min_salary = await driver.get_elements("#job-details > div.w-70-l.w-60-ms.w-100.pr2-l.pr2-ms.relative > div:nth-child(3) > div > div > section.salary.w-100.mt3.mb2.tr > div > span.dib.f2-5.fw6.black-80 > div > span:nth-child(1)")
            min_salary = await min_salary[0].get_text()
        except:
            min_salary=""

        try:
            max_salary = await driver.get_elements("#job-details > div.w-70-l.w-60-ms.w-100.pr2-l.pr2-ms.relative > div:nth-child(3) > div > div > section.salary.w-100.mt3.mb2.tr > div > span.dib.f2-5.fw6.black-80 > div > span:nth-child(2)")
            max_salary = await max_salary[0].get_text()
        except:
            max_salary=""
        try:
            salary_type = await driver.get_elements("#job-details > div.w-70-l.w-60-ms.w-100.pr2-l.pr2-ms.relative > div:nth-child(3) > div > div > section.salary.w-100.mt3.mb2.tr > div > span.ttc.dib.f5.fw4.black-60.pr1.i.pb")
            salary_type = await salary_type[0].get_text()
        except:
            salary_type=""
        try:
            num_app = await driver.get_elements("#num_of_applications")
            num_app = await num_app[0].get_text()
        except:
            num_app-""
        try:
            last_posted_date = await driver.get_elements("#last_posted_date")
            last_posted_date = await last_posted_date[0].get_text()
        except:
            last_posted_date=""
        try:
            expiry_date = await driver.get_elements("#expiry_date")
            expiry_date = await expiry_date[0].get_text()
        except:
            expiry_date=""
        try:
            jd = await driver.get_elements("#description-content")
            jd = await jd[0].get_text()
        except:
            jd=""
        try:
            company_info = await driver.get_elements("#job-details > div.w-30-l.w-40-ms.w-100.pl3-l.pl3-ms > div.db-l.dn-sp.mv3 > div > div > section > section > div.company-info > div > div.black-80.f6.lh-copy.break-word.op-after.pt1.mh100-sp.of-hide-sp")
            company_info = await company_info[0].get_text()
        except:
            company_info=""

        job_info={

            "company":company,
            "title":title,
            "employment type":employment_type,
            "location":location,
            "level":level,
            "min experience":min_exp,
            "category":category,
            "min salary":min_salary,
            "max salary":max_salary,
            "salary type":salary_type,
            "number of applications":num_app,
            "last posted date":last_posted_date,
            "expiry date":expiry_date,
            "job description":jd,
            "company info":company_info,
            "link":url
        }

        await save_product(url.split("-")[-1],job_info)
        return location



sem = asyncio.Semaphore(SEM)


async def safe_get_pages(i):
    async with sem:  # semaphore limits num of simultaneous downloads
        return await get_pages(i)


def old_function():

    for key in search_keys:
        sleep(5)
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
    pass


async def main():
    start_time = time.time()

    # keys=[
    #     'data scientist','machine learning engineer','ai','data engineer','quant','crypto',  #jobs
    #     'rio tinto','dyson','resmed asia','bhp','bp singapore','exxonmobil','lego', 'visa worldwide','mastercard','american express',    #large companies      
    #     'dbs','uob','ocbc','standard chartered', 'maybank', 'THE TORONTO-DOMINION BANK',  #banks   
    #     'jp morgan','nomura','citibank','MERRILL LYNCH GLOBAL','goldman sachs','bnp paribas','credit agricole',
    #     'societe generale','barclays','mizuho', 'credit suisse','ubs','deutsche bank', 'commerzbank', 'mufg','ing bank', #investment banks    
    #     'amazon web services',  'google asia pacific' ,'microsoft operations', 'facebook singapore' ,'netflix entertainment',
    #     'databricks asiapac', 'apple south', 'spotify singapore',#tech
    #     'tiktok pte ltd','pipo sg','bytedance','shopee singapore','lazada', 'grab','gojek','goto', 'tencent', 'proxima beta', #asian tech         
    #     'stripe','airwallex','gxs', 'foodpanda','coda payments',   # tech startups
    #     'paypal','asia wealth platform' ,'endowus singapore',    #fintech
    #     'trafigura','glencore','louis dreyfus','vitol','gunvor','cargill',  #commodities.
    #     'coinbase','binance','okx','nansen','bybit',  #crypto
    #     'ingensoma','alphagrep', 'xtx markets','world quant','flow traders','virtu financial',#HFT
    #     'kkr singapore','fiera capital','point72','MARSHALL WACE SINGAPORE','BLACKROCK ADVISORS SINGAPORE','balyasny',
    #     'qube research & technologies','qrt','millenium management','squarepoint capital','citadel','hrt sg',
    #     'jtp holdings','drw singapore','tower research capital','alphalab capital','grasshopper pte ltd' ,'EXODUSPOINT CAPITAL'# funds
    #     ]


    keys=[
        # 'data scientist','machine learning engineer','ai','data engineer','quant','crypto',  #jobs
        'rio tinto','dyson','resmed asia','bhp','bp singapore','exxonmobil','lego', 'visa worldwide','mastercard','american express',    #large companies      
        # 'dbs','uob','ocbc','standard chartered', 'maybank', 'THE TORONTO-DOMINION BANK',  #banks   
        # 'jp morgan','nomura','citibank','MERRILL LYNCH GLOBAL','goldman sachs','bnp paribas','credit agricole',
        # 'societe generale','barclays','mizuho', 'credit suisse','ubs','deutsche bank', 'commerzbank', 'mufg','ing bank', #investment banks    
        # 'amazon web services',  'google asia pacific' ,'microsoft operations', 'facebook singapore' ,'netflix entertainment',
        # 'databricks asiapac', 'apple south', 'spotify singapore',#tech
        # 'tiktok pte ltd','pipo sg','bytedance','shopee singapore','lazada', 'grab','gojek','goto', 'tencent', 'proxima beta', #asian tech         
        # 'stripe','airwallex','gxs', 'foodpanda','coda payments',   # tech startups
        # 'paypal','asia wealth platform' ,'endowus singapore',    #fintech
        # 'trafigura','glencore','louis dreyfus','vitol','gunvor','cargill',  #commodities.
        # 'coinbase','binance','okx','nansen','bybit',  #crypto
        # 'ingensoma','alphagrep', 'xtx markets','world quant','flow traders','virtu financial',#HFT
        # 'kkr singapore','fiera capital','point72','MARSHALL WACE SINGAPORE','BLACKROCK ADVISORS SINGAPORE','balyasny',
        # 'qube research & technologies','qrt','millenium management','squarepoint capital','citadel','hrt sg',
        # 'jtp holdings','drw singapore','tower research capital','alphalab capital','grasshopper pte ltd' ,'EXODUSPOINT CAPITAL'# funds
        ]

    search_keys=[]
    for search in tqdm(keys):
        key=""
        for item in search.split(" "):
            if item != search.split(" ")[-1]:
                key += item + "%20"
            else:
                key += item
        search_keys.append(key.lower())


    print(search_keys)

    tasks=[]
    df=pd.read_csv('../data/mycareersfuture_link_2023-11-04.csv')
    urls=[i.split("?")[0] for i in  list(df['Link'])]
    for url in urls:
        task = asyncio.create_task(safe_get_pages(url))
        tasks.append(task)

    print('Saving the output of extracted information')
    await asyncio.gather(*tasks) 
    time_difference = time.time() - start_time
    print(f'Scraping time: %.2f seconds.' % time_difference)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

