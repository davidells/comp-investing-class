# Computational Investing - Homework 3
#
# Some example output to verify correctness:
#
#  orders.csv:
#    The final value of the portfolio using the sample file is -- 2011,12,20,1133860
#    
#    Details of the Performance of the portfolio :
#    
#    Data Range :  2011-01-10 16:00:00  to  2011-12-20 16:00:00
#    
#    Sharpe Ratio of Fund : 1.21540462111
#    Sharpe Ratio of $SPX : 0.0183391412227
#    
#    Total Return of Fund :  1.13386
#    Total Return of $SPX : 0.97759401457
#    
#    Standard Deviation of Fund :  0.00717514512699
#    Standard Deviation of $SPX : 0.0149090969828
#    
#    Average Daily Return of Fund :  0.000549352749569
#    Average Daily Return of $SPX : 1.72238432443e-05
#
#  orders2.csv
#    The final value of the portfolio using the sample file is -- 2011,12,14,1078753
#    
#    Details of the Performance of the portfolio
#    
#    Data Range :  2011-01-14 16:00:00  to  2011-12-14 16:00:00
#    
#    Sharpe Ratio of Fund : 0.788988545538
#    Sharpe Ratio of $SPX : -0.177204632551
#    
#    Total Return of Fund :  1.078753
#    Total Return of $SPX : 0.937041848381
#    
#    Standard Deviation of Fund :  0.00708034656073
#    Standard Deviation of $SPX : 0.0149914504972
#    
#    Average Daily Return of Fund :  0.000351904599618
#    Average Daily Return of $SPX : -0.000167347202139

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

def market_sim(i_starting_amount, na_orders):
    # Parse list of dates from orders array
    ldt_dates = [dt.datetime(o[0],o[1],o[2]) for o in na_orders]

    #for rec in na_orders:
    #    dt_recdate = dt.datetime(rec[0], rec[1], rec[2])
    #    ldt_dates.append(dt_recdate)
        
    # Note date range (min/max) and unique list of symbols
    dt_mindate = min(ldt_dates)
    dt_maxdate = max(ldt_dates)
    ls_symbols = list(set(na_orders['f3']))

    # Fetch data across this date range and tickers
    d_data = hw1.fetch_data(dt_mindate, dt_maxdate, ls_symbols)
    df_prices = d_data['close']

    # Setup aquired dataframe
    df_acquired = d_data['close'].copy()
    df_acquired[:] = 0

    for rec in na_orders:
        s_date = '-'.join([
            str(item) for item in [rec[0],rec[1],rec[2]]
        ])
        s_symbol, s_buysell, f_size = (rec[3], rec[4], rec[5])

        f_amount = f_size * (s_buysell == "Buy" and 1 or -1)
        df_acquired[s_symbol][s_date] += f_amount

    # Setup owned (running total of acquired)
    df_owned = df_acquired.cumsum()

    df_value = df_owned * df_prices
    df_cashflow = df_acquired * df_prices * -1

    # Setup running cash total, make sure starting amount is correct
    df_cash = i_starting_amount + df_cashflow.sum(axis=1).cumsum()
    
    # Calculate total value from stock value plus cash, result is time series
    return df_value.sum(axis=1) + df_cash


if __name__ == '__main__':

    if len(sys.argv) < 2 or not sys.argv[1] in ["market_sim", "analyze"]:
        print 'usage: [market_sim|analyze] [options]'
        sys.exit(1)

    s_mode = sys.argv[1]
        
    # Market Simulator Mode
    if s_mode == "market_sim":
        if len(sys.argv) < 5:
            print 'usage: %s market_sim [starting amount] [orders file name] [output file name]'
            sys.exit(1)

        # Get command line options
        i_starting_amount = int(sys.argv[2])
        s_orders_fname = sys.argv[3]
        s_output_fname = sys.argv[4]

        # Import orders file as numpy recarray
        na_orders = np.loadtxt(s_orders_fname, dtype='i2,i2,i2,S5,S4,f4',
                            delimiter=',', comments="#", skiprows=0)

        # Run the simulation
        ts_portfolio_value = market_sim(i_starting_amount, na_orders)

        # Print a debug statement
        time_last = ts_portfolio_value.index[-1]
        print "The final value of the portfolio is %s -> %s" % \
            (time_last, ts_portfolio_value[time_last])

        # Write the output file
        file_output = open(s_output_fname, 'w')
        for time in ts_portfolio_value.index:
            file_output.write(
                "%s, %s, %s, %s\n" % \
                   (time.year, 
                    time.month, 
                    time.day, 
                    ts_portfolio_value[time]))

    # Analyze Mode
    elif s_mode == "analyze":
        if len(sys.argv) < 3:
            print 'usage: %s analyze [values file name] [benchmark symbol (ex: $SPX)]'
            sys.exit(1)

        # Get command line options
        s_values_fname = sys.argv[2]
        s_benchmark_symbol = sys.argv[3]

        # Import values file into pandas time series
        ts_portfolio_value = pd.io.parsers.read_csv(
            s_values_fname, header=None, parse_dates={'date':[0,1,2]}, 
            index_col=0, squeeze=True)

        # Fetch data for benchmark
        dt_start = ts_portfolio_value.index[0].to_datetime()
        dt_end = ts_portfolio_value.index[-1].to_datetime()
        d_data = hw1.fetch_data(dt_start, dt_end, s_benchmark_symbol)

        # Extract price arrays
        na_portfolio_value = ts_portfolio_value.values
        na_benchmark_price = d_data['close'][s_benchmark_symbol].values

        # Analyze price arrays
        t_portfolio_analysis = hw1.analyze_value_series(na_portfolio_value)
        t_benchmark_analysis = hw1.analyze_value_series(na_benchmark_price)

        # Plotting normalized prices
        times = ts_portfolio_value.index
        plt.clf()
        plt.plot(times, na_portfolio_value / na_portfolio_value[0])
        plt.plot(times, na_benchmark_price / na_benchmark_price[0])
        plt.legend(["Portfolio", s_benchmark_symbol])
        plt.ylabel('Normalized Close')
        plt.xlabel('Date')
        s_plot_fname = '%s-vs-%s' % \
            (s_values_fname.replace('.csv',''), s_benchmark_symbol)
        plt.savefig(s_plot_fname + '.pdf', format='pdf')

        # Print analysis
        print "Details of the performance of the portfolio:"
        print 
        print "Data Range: %s to %s" % (dt_start, dt_end)
        print

        ls_datakey_title =[ 
            ("sharpe", "Sharpe Ratio"),
            ("cum_ret", "Total Value"),
            ("std_dev", "Standard Deviation"),
            ("avg_ret", "Average Return")
        ]
        for (s_datakey, s_title) in ls_datakey_title:
            print "%s of Fund : %s" % (s_title, t_portfolio_analysis[s_datakey])
            print "%s of %s : %s" % (s_title, s_benchmark_symbol, t_benchmark_analysis[s_datakey])
            print
