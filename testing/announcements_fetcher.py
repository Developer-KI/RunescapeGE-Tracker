#%%
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import date
import time
import random 
from typing import List
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
#%%


def webpage_request(month: int, year: int) -> str: 
    website_fetch = requests.get(f'https://secure.runescape.com/m=news/a=13/archive?oldschool=1&year={year}&month={month}', headers=headers)
    return website_fetch.text

def get_announcements(months_ago: int|None = None, years_ago: int|None = None) -> List[pd.Timestamp]:
    """
    Scrapes news article timestamps from the Runescape news archive within a specified lookback period.

    Args:
        months_ago (int, optional): The number of months to look back. Defaults to None.
        years_ago (int, optional): The number of years to look back. Defaults to None.
    
    Returns:
        List[pd.Timestamp]: A list of scraped timestamps.
    """
    timestamp_list = []

    earliest_valid_date = pd.to_datetime('2015-03-28')
    end_date = pd.to_datetime(date.today())
    
    if years_ago is None and months_ago is None:
        start_date = earliest_valid_date
    else:
        years = years_ago if years_ago is not None else 0
        months = months_ago if months_ago is not None else 0
        start_date = end_date - pd.DateOffset(years=years, months=months)

    if start_date < earliest_valid_date:
        raise ValueError(f"Lookback period is before the start of the data ({earliest_valid_date.date()}).")

    for year in range(start_date.year, end_date.year + 1):
        start_month = start_date.month if year == start_date.year else 1
        end_month = end_date.month if year == end_date.year else 12
        
        for month in range(start_month, end_month + 1):
            time.sleep(random.uniform(1, 2))
            webtext = webpage_request(year=year, month=month)
            
            if not webtext:
                raise requests.HTTPError("Unexpected return from site")
            
            parser = BeautifulSoup(webtext, 'html.parser')
            articles = parser.find_all('article')
            
            for article in articles:
                time_tag = article.find('time')
                
                if time_tag and 'datetime' in time_tag.attrs:
                    articles_timestamp = pd.to_datetime(time_tag['datetime'])
                    timestamp_list.append(articles_timestamp)

    return timestamp_list
#%%