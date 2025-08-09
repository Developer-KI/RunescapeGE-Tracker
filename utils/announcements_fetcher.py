#%%
import  pandas as pd, numpy as np
import  sys,os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: sys.path.append(project_root)
import  requests
from    bs4 import BeautifulSoup
from    datetime import date
import  time
import  random 
from    typing import List
#%%


def webpage_request2(session: requests.Session, month: int, year: int) -> str:
    """
    Fetches HTML content from the Runescape news archive.
    Uses a requests session with a full set of rotating headers.
    """
    USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'
    ]

    a_int = random.uniform(1,1000)
    url = f'https://secure.runescape.com/m=news/a={a_int}/archive?oldschool=1&year={year}&month={month}'
    
    # Define a complete set of headers
    full_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': url,
        'DNT': '1', # Do Not Track request
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': random.choice(USER_AGENTS) # This header will rotate
    }
    
    try:
        # Update session headers with the full set for each request
        session.headers.update(full_headers)
        
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {year}-{month}: {e}")
        return ""

def get_announcements_new(months_ago: int|None = None, years_ago: int|None = None) -> List[pd.Timestamp]:
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
    with requests.Session() as session:
        for year in range(start_date.year, end_date.year + 1):
            start_month = start_date.month if year == start_date.year else 1
            end_month = end_date.month if year == end_date.year else 12
            print(f"Starting {year}")
            for month in range(start_month, end_month + 1):
                time.sleep(random.uniform(10,25)) #over-request countermeasure
                webtext = webpage_request2(session=session,year=year, month=month)
                
                if not webtext:
                    raise requests.HTTPError("Unexpected return from site")
                
                parser = BeautifulSoup(webtext, 'html.parser')
                articles = parser.find_all('article')
                if not articles:
                    raise requests.HTTPError("Site returned no articles, possible timeout or IP ban")
                
                for article in articles:
                    time_tag = article.find('time')
                    
                    if time_tag and 'datetime' in time_tag.attrs:
                        articles_timestamp = pd.to_datetime(time_tag['datetime'])
                        print(articles_timestamp)
                        timestamp_list.append(articles_timestamp)
                print(f"Finished month {month}")

    return timestamp_list

def get_announcements(cache_file_path='../data/announcements_cache.csv', scrape: bool = False) -> pd.DataFrame:
    """
    Handles caching logic for the get_announcements function.
    Reads from cache if available, appends new data, and saves.
    Use a VPN with this function.
    """
    
    # Check if a cache file exists
    if os.path.exists(cache_file_path):
        print("Cache file found. Reading existing data...")
        # Load the existing data
        cached_df = pd.read_csv(cache_file_path, parse_dates=['timestamp'])
        
        if scrape:
            # Get the most recent timestamp to know where to resume scraping
            last_timestamp = cached_df['timestamp'].max()
            last_timestamp = pd.to_datetime(last_timestamp)
            print(f"Last scraped timestamp: {last_timestamp}")
            
            # Calculate the lookback period from the last timestamp
            time_since_last_scrape = pd.to_datetime(date.today()) - last_timestamp
            years_ago = time_since_last_scrape.days // 365
            months_ago = (time_since_last_scrape.days % 365) // 30
            
            print(f"Scraping new data since {last_timestamp}...")
            
            # Call the core scraping function for the new period
            new_timestamps = get_announcements_new(months_ago=months_ago, years_ago=years_ago)
            new_df = pd.DataFrame(new_timestamps, columns=['timestamp'])
            
            # Append and save the combined data
            combined_df = pd.concat([cached_df, new_df]).drop_duplicates().sort_values(by='timestamp').reset_index(drop=True)
            combined_df.to_csv(cache_file_path, index=False)
            print("Done!")

            return combined_df
        else:
            print("Done!")
            return cached_df 
    
    else:
        print("No cache file found. Performing full scrape...")
        # Perform   a full scrape from the beginning
        full_timestamps = get_announcements_new(months_ago=None, years_ago=None)
        full_df = pd.DataFrame(full_timestamps, columns=['timestamp'])
        
        full_df.to_csv(cache_file_path, index=False)
        print("Done!")
        
        return full_df
#%%
