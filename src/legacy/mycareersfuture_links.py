"""
This module is used to srape data science job information on MyCareersFuture.

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
from selenium.webdriver.chrome.options import Options




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


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

async def save_data(key):
    pass

async def get_pages(file):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    service = Chromedriver(binary='/Users/admin/Downloads/chromedriver-mac-x64/chromedriver',options=chrome_options)
    browser = Chrome()

    async with get_session(service,browser) as driver:
        await driver.get(f'https://www.mycareersfuture.sg/search?search={key}&sortBy=new_posting_date&page=0')
        try:
            total_jobs_element= await driver.get_elements("#search-result-headers > div > div:nth-child(1)")   
            total_jobs = await total_jobs_element[0].get_text()
        except:
            total_jobs=0
    
        try: 
            pages= math.ceil(int(total_jobs.split(" ")[0])/20)
        except:
            pages=0
        print(f'key: {key}, jobs : {total_jobs}, pages: {pages}')
        return key


def old_function(search_keys):
    company_l=[]
    title_l=[]
    location_l=[]
    job_link_l=[]
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    driver = webdriver.Chrome("/Users/admin/Downloads/chromedriver-mac-x64/chromedriver",options=chrome_options)

    today = date.today()
    for key in search_keys:
        time.sleep(3)
        driver.get(f'https://www.mycareersfuture.sg/search?search={key}&sortBy=new_posting_date&page=0')
        try:
            total_jobs=int(driver.find_element(By.XPATH,'//*[@id="search-result-headers"]/div/div[1]').get_attribute("innerText").split("\n")[0].split(" ")[0])
        except:
            total_jobs=0

        pages= math.ceil(total_jobs/20)
        print(f'key: {key}, jobs : {total_jobs}, pages: {pages}')

        for page in range(pages):
            driver.get(f'https://www.mycareersfuture.sg/search?search={key}&sortBy=new_posting_date&page={page}')
            time.sleep(2)

            for i in range(20):
                try:
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
            
  
                    company_l.append(company)
                    title_l.append(title)
                    location_l.append(location)
                    job_link_l.append(job_link)

     
                    # Terminate the loop at the last page.
                except:
                    time.sleep(2)
        
    jobs = pd.DataFrame({
    'Company': company_l,
    'Title': title_l,
    'Location': location_l,
    'Link': job_link_l
    })

    jobs.to_csv(f'../data/mycareersfuture_link_{today}.csv',index=False)
    driver.quit()
    pass


def main():
    start_time = time.time()

    keys=[
        'data scientist','machine learning engineer','ai','data engineer','quant','crypto',  #jobs
        'rio tinto','dyson','resmed asia','bhp','bp singapore','exxonmobil','lego', 'visa worldwide','mastercard','american express',    #large companies      
        'dbs','uob','ocbc','standard chartered', 'maybank', 'THE TORONTO-DOMINION BANK',  #banks   
        'jp morgan','nomura','citibank','MERRILL LYNCH GLOBAL','goldman sachs','bnp paribas','credit agricole',
        'societe generale','barclays','mizuho', 'credit suisse','ubs','deutsche bank', 'commerzbank', 'mufg','ing bank', #investment banks    
        'amazon web services',  'google asia pacific' ,'microsoft operations', 'facebook singapore' ,'netflix entertainment',
        'databricks asiapac', 'apple south', 'spotify singapore',#tech
        'tiktok pte ltd','pipo sg','bytedance','shopee singapore','lazada', 'grab','gojek','goto', 'tencent', 'proxima beta', #asian tech         
        'stripe','airwallex','gxs', 'foodpanda','coda payments',   # tech startups
        'paypal','asia wealth platform' ,'endowus singapore',    #fintech
        'trafigura','glencore','louis dreyfus','vitol','gunvor','cargill',  #commodities.
        'coinbase','binance','okx','nansen','bybit',  #crypto
        'ingensoma','alphagrep', 'xtx markets','world quant','flow traders','virtu financial',#HFT
        'kkr singapore','fiera capital','point72','MARSHALL WACE SINGAPORE','BLACKROCK ADVISORS SINGAPORE','balyasny',
        'qube research & technologies','qrt','millenium management','squarepoint capital','citadel','hrt sg',
        'jtp holdings','drw singapore','tower research capital','alphalab capital','grasshopper pte ltd' ,'EXODUSPOINT CAPITAL'# funds
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
    old_function(search_keys)


    time_difference = time.time() - start_time
    print(f'Scraping time: %.2f seconds.' % time_difference)


if __name__ == "__main__":
    main()
