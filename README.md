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
data
````

| |	unix	| open	| high |	low |	close |	Volume BTC |	date |
|     :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | 	:---:      | 	:---:      | 	
| 0 |	1502946000 | 4308.83 |	4328.69 |	4291.37 |	4315.32 |	23.230000 |	2017-08-17 05:00:00 |	
| 1 | 1502949600 | 4315.32 | 4345.45 | 4309.37 | 4324.35 | 7.230000 | 2017-08-17 06:00:00 |
| 2 | 1502953200 | 4324.35 | 4349.99 | 4287.41 | 4349.99 | 4.440000 | 2017-08-17 07:00:00 |


Here is a plot of the price development

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


### Classification when a good entry point is in place
I would like to train the model so that he buys at good times if possible and sells when the price goes down. For this I have thought of the following logic:

If for time span x hours (e.g. 24h) the price increases by y percent (e.g. 2 %), then the range must be labeled as a good entry point
However, since the price is very volatile, I calculate two moving averages

Create moving averages


````python
average_length_current = 60 # in hours
average_length_future = 24  # in hours
distance_time =  2          # in hours x-axis
distance_value = 2          # in precent y-axis
distance_value = 1+(distance_value/100) # 

# create moving averages
data['average_length_current'] = data.loc[:,"close"].rolling(window=average_length_current).mean()
data['average_length_future'] =  data.loc[:,"close"].rolling(window=average_length_future).mean()
data.tail()
````

 | | unix | open | high | low | close | Volume BTC | date | average_length_current | average_length_future  | 
 |     :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | 	:---:      | 	:---:      | 		:---:      | 		:---:      | 	
 | 33351 | 1613505600 | 48536.35 | 48784.29 | 48173.81 | 48769.27 | 2584.417826 | 2021-02-16 20:00:00 | 48454.874000 | 48785.043333  | 
 | 33352 | 1613509200 | 48769.28 | 48809.09 | 48355.80 | 48575.07 | 2220.458707 | 2021-02-16 21:00:00 | 48445.553000 | 48800.357083 | 
 | 33353 | 1613512800 | 48575.06 | 49180.00 | 48524.65 | 49108.68 | 1763.412213 | 2021-02-16 22:00:00 | 48446.960833 | 48834.681667 |
 | 33354 | 1613516400 | 49113.91 | 49290.00 | 48933.33 | 49133.45 | 1959.461793 | 2021-02-16 23:00:00 | 48448.265667 | 48885.612917 | 
 | 33355 | 1613520000 | 49133.45 | 49476.00 | 49133.44 | 49361.52 | 615.901013 | 2021-02-17 00:00:00 | 48447.010000 | 48956.133750 | 
