import pandas as pd
import json

def data_preprocess(read: bool, filepath: str = "../data", read_path: str = "../data/processed_data.csv", write: bool = False, interp_method: str = 'linear') -> pd.DataFrame:
    ### Read has higher priority than write
    if read:
        df = pd.read_csv(f'{read_path}', names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp', 'totalvol', 'wprice'])
        return df

    #Load the data
    raw_pricedata = pd.read_csv(f'{filepath}/data.csv', names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])
    #Load the data properties
    with open(f'{filepath}/data_properties.txt', "r") as file:
        lines = file.readlines()
        file.close()
    series_length = int(lines[1].replace("\n", ""))

    #Keep constantly only constantly traded items
    group_raw_pricedata = raw_pricedata.groupby('item_id').nunique()
    filtered_indexes = group_raw_pricedata[group_raw_pricedata['timestamp'] != series_length].index
    raw_pricedata = raw_pricedata[~raw_pricedata['item_id'].isin(filtered_indexes)]
    
    # interpolate missing values
    processed_priced_data = raw_pricedata.interpolate(method=interp_method)
    
    #Weighted average of High/Low Price by High/Low Volume
    processed_priced_data['totalvol'] = processed_priced_data['highPriceVolume'] + processed_priced_data['lowPriceVolume']
    processed_priced_data['wprice'] = (processed_priced_data['highPriceVolume']/processed_priced_data['totalvol']) * (processed_priced_data['avgHighPrice'] - processed_priced_data['avgLowPrice']) + processed_priced_data['avgLowPrice']

    #Saving output
    if write:
        processed_priced_data.to_csv(f'{filepath}/processed_data.csv', mode='w', header=False, index=False)
    
    return processed_priced_data

def alchemy_preprocess(read: bool, filepath: str = "../data", read_path: str = "../data/alchemy_data.csv", write: bool = False) -> pd.DataFrame:
    ### Read has higher priority than write
    if read:
        df = pd.read_csv(f'{read_path}', names=['item', 'price'], index_col=0)
        return df

    with open(f'{filepath}/namealchemy.json', "r") as file:
        alc_data = json.load(file)
        file.close()
    with open(f'{filepath}/nameID.json', "r") as file:
        name_data = json.load(file)
        file.close()

    # Convert dictionary to DataFrame
    high_alchemy = pd.DataFrame(list(alc_data.items()), columns=["item", "price"])
    nameID = pd.DataFrame(list(name_data.items()), columns=["item", "item_id"])

    #Processing tables
    reference = pd.merge(nameID, high_alchemy, on="item", how="inner")
    reference = reference.drop([0,1])
    reference.set_index('item_id', inplace=True)

    #Saving output
    if write:
        reference.to_csv(f'{filepath}/alchemy_data.csv', mode='w', header=False, index=True)

    return reference

def volatility_market(market_data: pd.DataFrame, smoothing: int = 20) -> pd.Series:
    price_market_data = market_data.pivot(index="timestamp", columns="item_id", values="wprice")
    #Aggregate volatility
    volatilityitems = price_market_data.rolling(window=smoothing).std()
    volatilitymarket = volatilityitems.sum(axis=1)
    #Scaling
    corr_price_market = price_market_data.corr()
    volatilitymarket = volatilitymarket/corr_price_market.shape[1]

    return pd.DataFrame(volatilitymarket, columns=['market_vix'])

if __name__ == "__main__":
    test = data_preprocess(read=False, write=True)



