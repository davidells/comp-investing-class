import sys

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import homework1 as hw1

def bollinger(df_prices, lookback=20):
    df_rolling_mean = pd.rolling_mean(df_prices, lookback)
    df_rolling_std = pd.rolling_std(df_prices, lookback)
    df_upper_band = df_rolling_mean + df_rolling_std
    df_lower_band = df_rolling_mean - df_rolling_std
    df_bollinger_val = (df_prices - df_rolling_mean) / df_rolling_std

    return {
        "mean": df_rolling_mean, 
        "upper": df_upper_band, 
        "lower": df_lower_band, 
        "value": df_bollinger_val
    }


if __name__ == '__main__':
    dt_mindate = dt.datetime(2010, 1, 1)
    dt_maxdate = dt.datetime(2010, 12, 31)
    ls_symbols = ['AAPL','GOOG','IBM','MSFT']

    # Fetch data across this date range and tickers
    d_data = hw1.fetch_data(dt_mindate, dt_maxdate, ls_symbols)
    df_prices = d_data['close']

    # Calculate bollinger values
    d_bollinger = bollinger(df_prices, lookback=20)

    # Plotting
    s_symbol = 'GOOG'
    times = df_prices.index
    plt.clf()
    plt.figure(figsize=(10,8))

    plt.subplot(211)
    plt.plot(times, df_prices[s_symbol])
    plt.plot(times, d_bollinger['mean'][s_symbol])
    plt.plot(times, d_bollinger['upper'][s_symbol])
    plt.plot(times, d_bollinger['lower'][s_symbol])
    plt.legend([s_symbol, "Rolling Mean", "Upper band", "Lower band"], loc=3)
    plt.ylabel('Close')
    plt.xlabel('Date')

    plt.subplot(212)
    plt.plot(times, d_bollinger['value'][s_symbol].fillna(0))
    plt.ylabel('Bollinger Value')
    plt.xlabel('Date')
    plt.savefig(s_symbol + '-bollinger.pdf', format='pdf')
