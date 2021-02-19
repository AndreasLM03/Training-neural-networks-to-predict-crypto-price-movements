# Training-neural-networks-to-predict-crypto-price-movements


In the following, I want to download historical crypto data, calculate a classification of when is a good time to buy, export price data as CandleCharts, then perform supervised learning based on the classification, and ultimately then make predictions using the model


## Classification
### Load and prepare historical data from https://www.cryptodatadownload.com/data/. I choose the binance exchange

````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url="https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv"
data =pd.read_csv(url, skiprows=1) 
data.drop(['date', 'symbol', 'Volume USDT', 'tradecount'], axis=1, inplace = True) # Delete unnecessary data
data['unix'] = data['unix'].astype(str).str[:10] # Manual adjustment of the time dataset - If Unixtime has more than 10 digits, it must be divided by 1000
data['date'] =   pd.to_datetime(data.unix, unit='s') # Convert dates. Attention UNIX EPOCH time has three 0's too many.
data = data.iloc[::-1] # flip upside down
data.reset_index(drop=True, inplace = True) # Create a new index
````

| |	unix	| open	| high |	low |	close |	Volume BTC |	date |
|     :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | 	:---:      | 	:---:      | 	
| 0 |	1502946000 | 4308.83 |	4328.69 |	4291.37 |	4315.32 |	23.230000 |	2017-08-17 05:00:00 |	
| 1 | 1502949600 | 4315.32 | 4345.45 | 4309.37 | 4324.35 | 7.230000 | 2017-08-17 06:00:00 |
| 2 | 1502953200 | 4324.35 | 4349.99 | 4287.41 | 4349.99 | 4.440000 | 2017-08-17 07:00:00 |


````python
# figure
fig, ax1 = plt.subplots(figsize=(15,6))
ax1.grid(color='black', linestyle='--', linewidth=0.1)
ax1.set_xlabel('date')
ax1.set_ylabel('Value in USD')
ax1.plot(data.date, data.close, label='BTC')
ax1.tick_params(axis='y')
ax1.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
````

<img src= "00.png" width="800">
