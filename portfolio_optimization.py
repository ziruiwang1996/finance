#This program attempts to optimize a user's portfolio using the Efficient Frontier
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock tickers in portfolio
assets = ['AAPL', 'TSLA', 'GS', 'NVDA', 'BILI']
#Assign weights to stocks (total addup to 1)
weights = np.array([1000/2700, 1000/2700, 500/2700, 100/2700, 100/2700])
#Get portfolio start date
stockStartDate = '2021-01-01'
#Get the ending day
today = datetime.today().strftime('%Y-%m-%d')

#Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()
#Store the adjusted close price of the stock into the df
for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo', start=stockStartDate, end=today)['Adj Close']
#print(df)

#Visually show the portfolio
title = 'Portfolio Adjusted Close Price History'
my_stocks = df
for i in my_stocks.columns.values:
    plt.plot(my_stocks[i], label=i)
plt.title(title)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Adj. Price ($)', fontsize=15)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show()

returns = df.pct_change() #Daily simple return
#print(returns)
#Create and show annualized covariance matrix
annual_cov_matrix = returns.cov()*252 #number of trading days in a yr

#Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(annual_cov_matrix, weights))
#Calculate the portfolio volatility (StDev)
port_volalitity = np.sqrt(port_variance)
#Calculate the annual portfolio return
port_annual_return = np.sum(returns.mean()*weights)*252
#Show the expected annual return, volatility & variance
percent_van = str(round(port_variance, 2)*100)+'%'
percent_vol = str(round(port_volalitity, 2)*100)+'%'
percent_ret = str(round(port_annual_return, 2)*100)+'%'
print('The portfolio variance is: ', percent_van)
print('The portfolio volatility is: ', percent_vol)
print('The portfolio annual return is: ', percent_ret)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
#Portfolio optimization
#Calculate the expected returns and the annualised sample covariance matrix of assets returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
#Optimize for max sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#Get the discrete allocation of each share/stock
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=2700)
allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))
