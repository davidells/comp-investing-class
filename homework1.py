# Computational Investing - Homework 1
#
# Some example output to verify correctness:
#
#  Start Date: January 1, 2011
#  End Date: December 31, 2011
#  Symbols: ['AAPL', 'GLD', 'GOOG', 'XOM']
#  Optimal Allocations: [0.4, 0.4, 0.0, 0.2]
#  Sharpe Ratio: 1.02828403099
#  Volatility (stdev of daily returns):  0.0101467067654
#  Average Daily Return:  0.000657261102001
#  Cumulative Return:  1.16487261965
#
#  Start Date: January 1, 2010
#  End Date: December 31, 2010
#  Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
#  Optimal Allocations:  [0.0, 0.0, 0.0, 1.0]
#  Sharpe Ratio: 1.29889334008
#  Volatility (stdev of daily returns): 0.00924299255937
#  Average Daily Return: 0.000756285585593
#  Cumulative Return: 1.1960583568

#Python library imports
import itertools

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fetch_data(dt_start, dt_end, ls_symbols):
    # Convenience, allow passing of single symbol
    if type(ls_symbols) == str:
        ls_symbols = [ls_symbols]

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(
        dt_start, dt_end + dt.timedelta(days=1), dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    return d_data

def assess_portfolio(d_data, ls_allocations):
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    na_allocations = np.array(ls_allocations)

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]

    #Calculate portfolio normalized price
    na_portfolio_norm_value = np.dot(na_normalized_price, na_allocations)

    return analyze_value_series(na_portfolio_norm_value)

    # Alternative calculation...
    # Arbitrary portfolio starting value of 1 million
    #i_starting_value = 1000000
    #na_shares_held = (i_starting_value * na_allocations) / na_price[0, :]
    #na_portfolio_value = np.dot(na_price, na_shares_held)
    #return analyze_value_series(na_portfolio_value)

def analyze_value_series(na_values):
    # Normalize prices
    na_normalized_values = na_values / na_values[0]

    # Copy the prices to a new ndarry to find returns.
    na_rets = na_normalized_values.copy()

    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(na_rets)

    f_std_dev = np.std(na_rets)
    f_avg_ret = np.ma.average(na_rets)
    f_sharpe = np.sqrt(252) * (f_avg_ret / f_std_dev)
    f_cumulative_ret = na_normalized_values[-1]

    return {
        'std_dev': f_std_dev,
        'avg_ret': f_avg_ret,
        'sharpe': f_sharpe,
        'cum_ret': f_cumulative_ret
    }

def simulate(dt_start, dt_end, ls_symbols, ls_allocations):
    d_data = fetch_data(dt_start, dt_end, ls_symbols)
    return assess_portfolio(d_data, ls_allocations)

def max_sharpe_portfolio(dt_start, dt_end, portfolio):
    max_measure = None
    max_allocations = None
    max_portfolio_stats = None

    ranges = list(itertools.repeat(range(11), len(symbols)))
    for point in itertools.product(*ranges):

        allocations = np.array(point) * 0.1
        if sum(allocations) != 1:
            continue

        portfolio_stats = simulate(dt_start, dt_end, symbols, allocations)
        sharpe = portfolio_stats['sharpe']
        if max_measure is None or sharpe > max_measure:
            max_measure = sharpe
            max_allocations = allocations
            max_portfolio_stats = portfolio_stats

    return {
        "allocations": max_allocations,
        "volatility": max_portfolio_stats["std_dev"],
        "average_daily_return": max_portfolio_stats["avg_ret"],
        "sharpe_ratio": max_portfolio_stats["sharpe"],
        "cumulative_return": max_portfolio_stats["cum_ret"]
    }


if __name__ == '__main__':
    #main()
    
    symbols = ["AAPL", "GLD", "GOOG", "XOM"]
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)

    #ls_allocations = [0.4, 0.4, 0.0, 0.2]
    #simulate(dt_start, dt_end, symbols, ls_allocations)

    #symbols = ["BRCM", "ADBE", "AMD", "ADI"]
    #dt_start = dt.datetime(2010, 1, 1)
    #dt_end = dt.datetime(2010, 12, 31)

    #symbols = ["BRCM", "TXN", "IBM", "HNZ"]
    #dt_start = dt.datetime(2010, 1, 1)
    #dt_end = dt.datetime(2010, 12, 31)

    #allocations = [0.4, 0.4, 0.0, 0.2]
    #vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, symbols, allocations)

    portfolio = max_sharpe_portfolio(dt_start, dt_end, symbols)

    print "Start Date: %s" % dt_start
    print "End Date: %s" % dt_end
    print "Symbols: %s" % symbols
    print "Optimal Allocations: %s" % portfolio['allocations']
    print "Sharpe Ratio: %s" % portfolio['sharpe_ratio']
    print "Volatility: %s" % portfolio['volatility']
    print "Average Daily Return: %s" % portfolio['average_daily_return']
    print "Cumulative Return: %s" % portfolio['cumulative_return']



# This function is left around as notes about how to fetch and plot data
def notes():
    ''' Main Function'''

    # List of symbols
    ls_symbols = ["AAPL", "GLD", "GOOG", "$SPX", "XOM"]

    # Start and End date of the charts
    dt_start = dt.datetime(2006, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)

    d_data = fetch_data(dt_start, dt_end, ls_symbols)

    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values

    # Plotting the prices with x-axis=timestamps
    plt.clf()
    plt.plot(ldt_timestamps, na_price)
    plt.legend(ls_symbols)
    plt.ylabel('Adjusted Close')
    plt.xlabel('Date')
    plt.savefig('adjustedclose.pdf', format='pdf')

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]

    # Plotting the prices with x-axis=timestamps
    plt.clf()
    plt.plot(ldt_timestamps, na_normalized_price)
    plt.legend(ls_symbols)
    plt.ylabel('Normalized Close')
    plt.xlabel('Date')
    plt.savefig('normalized.pdf', format='pdf')

    # Copy the normalized prices to a new ndarry to find returns.
    na_rets = na_normalized_price.copy()

    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(na_rets)

    # Plotting the plot of daily returns
    plt.clf()
    plt.plot(ldt_timestamps[0:50], na_rets[0:50, 3])  # $SPX 50 days
    plt.plot(ldt_timestamps[0:50], na_rets[0:50, 4])  # XOM 50 days
    plt.axhline(y=0, color='r')
    plt.legend(['$SPX', 'XOM'])
    plt.ylabel('Daily Returns')
    plt.xlabel('Date')
    plt.savefig('rets.pdf', format='pdf')

    # Plotting the scatter plot of daily returns between XOM VS $SPX
    plt.clf()
    plt.scatter(na_rets[:, 3], na_rets[:, 4], c='blue')
    plt.ylabel('XOM')
    plt.xlabel('$SPX')
    plt.savefig('scatterSPXvXOM.pdf', format='pdf')

    # Plotting the scatter plot of daily returns between $SPX VS GLD
    plt.clf()
    plt.scatter(na_rets[:, 3], na_rets[:, 1], c='blue')  # $SPX v GLD
    plt.ylabel('GLD')
    plt.xlabel('$SPX')
    plt.savefig('scatterSPXvGLD.pdf', format='pdf')

