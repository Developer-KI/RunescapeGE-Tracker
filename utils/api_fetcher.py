#%%
import requests
import pandas as pd
import time
from   datetime import datetime
#%%
headers = {
    'User-Agent': 'Price/Volume Tracker and Scraper- NoHFT',
    'From': 'mstavreff@outlook.com, discord: shrimpsalad'
}

def fetch_latest_deprecated(item_ids: list[int]) -> pd.DataFrame:
    item_ids = list(map(str, item_ids))
    item_call = '|'.join([i for i in item_ids])
    
    url = f"https://api.weirdgloop.org/exchange/history/osrs/latest?id={item_call}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        records = list(data.values())
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.reset_index()
        df = df.rename(columns={"index": "item_id"})
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical(item_id: int) -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id={item_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = []
        for item_id, entries in data.items():
            for entry in entries:
                records.append(entry)

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df["item_id"] = item_id
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_latest() -> pd.DataFrame:
    url = f"https://prices.runescape.wiki/api/v1/osrs/latest"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame.from_dict(data['data'], orient='index')
        df['highTime'] = pd.to_datetime(df['highTime'], unit='s')
        df['lowTime'] = pd.to_datetime(df['lowTime'], unit='s')
        df = df.reset_index()
        df = df.rename(columns={"index": "item_id"})
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_5min(timestamp: int = 0) -> pd.DataFrame:
    time_to_present = int(datetime.now().timestamp()) - timestamp

    if timestamp == 0 or time_to_present < 0:
        url = f"https://prices.runescape.wiki/api/v1/osrs/5m"
    else:
        url = f"https://prices.runescape.wiki/api/v1/osrs/5m?timestamp={timestamp}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame.from_dict(data['data'], orient='index')
        df = df.reset_index()
        df['timestamp'] = timestamp
        df = df.rename(columns={"index": "item_id"})
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical_5m(n = 10, mins=5, waits=1.1, timestamp: int = 0) -> pd.DataFrame:
    if timestamp != 0:
        unix_timestamp_seconds = timestamp
    else:
        unix_timestamp_seconds = int(datetime.now().timestamp())

    unix_timestamp_seconds = unix_timestamp_seconds - unix_timestamp_seconds % 300
    df = fetch_5min(unix_timestamp_seconds)

    if df.empty:
        print(f"API returned an empty dataset for initial timestamp {unix_timestamp_seconds}. Returning empty DataFrame.")
        return df

    for t in range(1, n):
        df_t = fetch_5min(unix_timestamp_seconds - (mins * 60) * t)
        
        # --- Check for empty DataFrame for all subsequent calls ---
        if df_t.empty:
            print(f"API returned an empty dataset for timestamp {unix_timestamp_seconds - (mins * 60) * t}. Skipping this chunk.")
            time.sleep(waits)
            continue
            
        df = pd.concat([df, df_t], ignore_index=True)
        time.sleep(waits)

    return df[['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp']]

### When forward mining its mandatory to include duplication deletion in case of getting to present. For opening mine forward direction is not possible
def writing_returns(filepath: str = "./data/data.csv", n_periods: int = 100, p_chunks: int= 10, timestamp=None, del_duplicates: bool = True, mining_forward: bool = False) -> None:
    #1 for backward and -1 for forward
    direction = 1

    #Line index 0: Timestamp of start of data 1: Timestamp of end of data 2: Lenght of timestamp data for each item
    with open("./data/data_properties.txt", "r") as file:
        lines = file.readlines()

    # Main logic for selecting the mining mode
    if not lines: # Scenario 1: No previous data exists
        if mining_forward:
            raise Exception("Cannot mine forward without previous data.")
        
        print('Starting new session: Mining Backward from Present')
        timestamp_start = int(datetime.now().timestamp())
        timestamp_start -= timestamp_start % 300
        series_length = 0
        direction = 1

    elif mining_forward: # Scenario 2: Resuming a forward session
        print('Resuming Mining Forward')
        timestamp_start = int(lines[0].strip())
        series_length = int(lines[2].strip())
        direction = -1

    elif timestamp is not None: # Scenario 3: Resuming backward from a specified timestamp
        print('Resuming Mining Backward from Specified Time Period')
        timestamp_start = timestamp
        timestamp_start -= timestamp_start % 300
        series_length = int(lines[2].strip())
        direction = 1

    else: # Scenario 4: Resuming a standard backward session
        print('Resuming Mining Backward')
        timestamp_start = int(lines[1].strip())
        series_length = int(lines[2].strip())
        direction = 1

    print(f"Initialized process. Expected mining time: {round(n_periods * p_chunks * 1.1 / 60, 3)} minutes")
    first_call_timestamp = timestamp_start
    last_call_timestamp = 0

    iter = 0    #accounts for dropped API calls
    for t in range(0, p_chunks):
        try:
            #Fetching Logic
            time_to_present = int(datetime.now().timestamp()) - (timestamp_start - direction * ((t * n_periods) * 300))
            df_t = fetch_historical_5m(n = n_periods, timestamp=timestamp_start - direction * ((t * n_periods) * 300))
            if df_t.empty:
                continue
            iter += 1
            last_call_timestamp = df_t.iloc[-1]['timestamp']
            first_call_timestamp = df_t.iloc[1]['timestamp']
            df_t = df_t[['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp']]
            df_t.to_csv(filepath, mode='a', header=False, index=False)

            # Determine first and last timestamps for this chunk
            current_first_call_timestamp = df_t.iloc[0]['timestamp']
            current_last_call_timestamp = df_t.iloc[-1]['timestamp']

            # Update overall timestamps based on mining direction
            if direction == 1: # Backward mining
                if t == 0: # For the first chunk, the start is the initial timestamp
                    first_call_timestamp = timestamp_start
                last_call_timestamp = current_last_call_timestamp
            else: # Forward mining
                if t == 0:
                    last_call_timestamp = timestamp_start
                first_call_timestamp = current_first_call_timestamp

            # Update series length based on direction
            if direction == 1:
                series_length += n_periods
            else:
                # Assuming you handle this logic correctly, this can stay as is
                series_length += n_periods

            # Saving to data properties
            with open("./data/data_properties.txt", "w") as file:
                file.write(f"{first_call_timestamp}\n")
                file.write(f"{last_call_timestamp}\n")
                file.write(f"{series_length}\n")
            print(f"{(iter) * n_periods} queries added! Time: {last_call_timestamp}")

            #Forward Mining reaching present redundency break
            if mining_forward == True and last_call_timestamp - first_call_timestamp == 0:
                print("Forward Mining reached present")
                break;
        
        except Exception as e:
            print(f"An error occurred during fetch for timestamp {last_call_timestamp}: {e}. Skipping this chunk.")
            continue
    print("Success!")
    
    #Handling of duplicates
    if del_duplicates:
        df = pd.read_csv(filepath, names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])
        df = df.drop_duplicates()
        df.to_csv(filepath, mode='w', header=False, index=False)
        print("Handled Duplicates")

def fetch_historical_common_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Common%20Trade%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Common Trade Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical_food_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Food%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Food Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
    
def fetch_historical_herb_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Herb%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Herb Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical_log_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Log%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Log Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
        
def fetch_historical_metal_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Metal%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Metal Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
    
def fetch_historical_rune_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Rune%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Rune Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
    
def fetch_latest_idex_df():
    index_calls = {"Common%20Trade": "GE Common Trade Index", "Food": "GE Food Index", "Herb": "GE Herb Index", "Log": "GE Log Index", "Metal": "GE Metal Index", "Rune": "GE Rune Index"}
    df = pd.DataFrame(columns=[])

    for index in index_calls.keys():
        url = f"https://api.weirdgloop.org/exchange/history/osrs/latest?id=GE%20{index}%20Index"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            df_t = pd.DataFrame.from_dict(data[index_calls[index]], orient='index').T
            df = pd.concat([df, df_t], axis=0)
        else:
            raise Exception("Failed to fetch data")
        
        time.sleep(1.1)
            
    df = df.reset_index()
    del df['index']
    return df
#%%
if __name__ == "__main__":
    #writing_returns(n=10, p=500, timestamp=1747701935,del_duplicates=False)
    writing_returns(n_periods=10, p_chunks=2000,del_duplicates=False, mining_forward=False)
#run as .py file