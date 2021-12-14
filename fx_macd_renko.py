# =============================================================================
# TRADING SYSTEM WHICH CAN EXTRACT PERIODICALLY, PERFORM ANALYSIS, EXECUTE TRADING STRATEGY, AND 
# PLACE/CLOSE ORDERS IN AN AUTOMATED FASHION

# ANUSHKA GUPTA NIT PATNA
# Automated trading script I - MACD
# 
# Using FXCM as our broker for APIs
# Used yahoofinance and tradingview
# =============================================================================

#IMPORTING LIBRARIES
from os import linesep
import fxcmpy
# Using FXCM as our broker for APIs
import numpy as np
from stocktrends import Renko
# coding renko from candlesticks data is a very difficult task, hence using a library
import statsmodels.api as sm
import time
import copy

#initiating API connection and defining trade parameters
token_path = "--" #PLACE WHERE PASSWORDS ARE STORED, WHICH ARE USUALLY NOT HARD CODED ;) 
con = fxcmpy.fxcmpy(access_token = open(token_path,'r').read(), log_level = 'error', server='demo')
        # Establising a connection with fmcx
        # access_token = open(token_path,'r').read() : giving access to the token
        # log_level = 'error' : whether we want to check for an error or not
        # server='demo' : because we using a free account.

#defining strategy parameters
#These are the assets we are performing our strategy on
pairs = ['EUR/USD','GBP/USD','USD/CHF','AUD/USD','USD/CAD'] #currency pairs to be included in the strategy

#pairs = ['EUR/JPY','USD/JPY','AUD/JPY','AUD/NZD','NZD/USD']
pos_size = 10 #max capital allocated/position size for any currency pair


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MACD : Moving Average Convergance Divergence
# it is a trend following momentum indicator
# It calculates the difference between two moving averages of an asset
# Typically 12 period MA and 26 period MA
# 12 period : fast moving average and 26 is slow moving average
# A SIGNAL line is also calculated which is again a moving average (typically 9 period) of the MACD line.
# Bullish period : MACD line cuts the signal line from below
# Bearish period : MACD line cuts the signal line from aobve
# This indicator should be used in conjunction with other indicators
# MACD line = (12-day EMA) - (26-day EMA), EMA = exponential movig average 
# TO CALCULATE EMA:
    # calculate SMA = (period value/number of period)
    # calculate the multiplier = 2/(number of periods + 1)
    # first EMA is equal to SMA
    # EMA for nect days = Closing price x multiplier + EMA (previous day) x (1-multiplier)
# In the function we will first pass our dataframe, then period for fast moving average 
# and then period for slow moving average
#__________________________________________________________________________________________________________________


def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    # Since changes will be made to our dataframe, its better we make changes to the copy of the data frame

    df["MA_Fast"]=df["Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Close"].ewm(span=b,min_periods=b).mean()
    # evm is alsmost same as or close to EMA, 
    # span means the number of periods we give for calculation of multiplier
    # min_period signifies the minimum number of periods required for the calculation of ewm

    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    # MACD = (macd-fast) - (macd-low)

    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    # calculation of signal which is the ewm of macd line

    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ATR = Average True Range
# Its a volatality based technical indicator
# ATR focuses on the total price movement and conveys how widly the markets are swinging as it moves.
# It takes into account 3 price movements 
    # Difference between low and high of a period
    # Difference between High and previous period's close
    # Difference between low and previous period's close
# To calculate the ATR, we need to first calculate the true range:
    # (true_range) = max[(high - low), abs(high - pervious low), abs(low - pervious low)]
    # ATR is simply the EMA of the true range
# In the function we need to pass dataframe and n which is the rolling window, usually of 14
#__________________________________________________________________________________________________________________


def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    # Since changes will be made to our dataframe, its better we make changes to the copy of the data frame

    df['H-L']=abs(df['High']-df['Low'])
    # Difference between low and high of a period

    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    # Difference between High and previous period's close

    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    # Difference between low and previous period's close

    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    # calculation of true range
    # axis = 1 ensures that max is calculated along the rows and cols
    # skipna = false, ensures that nan are not skipped and max of nan and other values is nan

    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    # Calculation of ATR
     
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RENCO is built using price movement and not against standarized time interval, 
# this filters out noise and we are able to see the true trend.
# Price movements are represented as bricks stacked at 45 degrees to each other
# A new brick is added to the chart only when the price moves by a predetermined amount in either direction.
# renco chart has a time axis, but the time scale is NOT FIXED. Some bricks may take longer time to form than others
# depending on how long it takes to move the required block size.
#__________________________________________________________________________________________________________________

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"

    df = DF.copy()
    # Since changes will be made to our dataframe, its better we make changes to the copy of the data frame

    df.reset_index(inplace=True)
    # Used to update index, implace = true means changes need to be made in our datarame 

    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","close","high","low","volume"]
    # We need to change the names of the columns of our dataframe to make it compatible for our Renko library
    # this is an impirtant step

    df2 = Renko(df)
    # Creation of the Renko object

    df2.brick_size = round(ATR(DF,120)["ATR"][-1],4)
    # Brick size is the most important parameter for our renko
    # We will use brick as 3 times the ATR of our hourly data
    # And we are using the latest ATR (-1 which is used above)

    renko_df = df2.get_bricks()
    #  Gives us the transformed data set

    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

def renko_merge(DF):
    "function to merging renko df with original ohlc df"
    df = copy.deepcopy(DF)
    df["Date"] = df.index
    renko = renko_DF(df)
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    merged_df = df.merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
    merged_df["bar_num"].fillna(method='ffill',inplace=True)
    merged_df["macd"]= MACD(merged_df,12,26,9)[0]
    merged_df["macd_sig"]= MACD(merged_df,12,26,9)[1]
    merged_df["macd_slope"] = slope(merged_df["macd"],5)
    merged_df["macd_sig_slope"] = slope(merged_df["macd_sig"],5)
    return merged_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We are using MACD and renko for making our trading strategies
# Buy signal:
#   renko bar greater than equal to 2
#   macd line is above the signal line
#   MACD line's slope is greater than the signal's slope for over period of 5
#   exit when the aove conditions don;t match 

# for sell singal do the opposite
#__________________________________________________________________________________________________________________


def trade_signal(MERGED_DF,l_s):
    "function to generate signal"
    signal = ""
    df = copy.deepcopy(MERGED_DF)
    if l_s == "":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Buy"
        elif df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Sell"
            
    elif l_s == "long":
        if df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Sell"
        elif df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
            
    elif l_s == "short":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Buy"
        elif df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
    return signal
    

def main():
    try:
        # In case the function doesnt run due to some glitches we can skip the 
        # We can simply skip the iteration.
        
        open_pos = con.get_open_positions()
        # Before one gets started, a check of the open positions might be helpful.
        # con is related to fmcx
        # An open position in investing is any established or entered trade that has yet to close with an opposing trade

        for currency in pairs:
            long_short = ""
            if len(open_pos)>0:
                # We need to check if open_pos is empty, because if its empty then we cannot run the programs ahead

                open_pos_cur = open_pos[open_pos["currency"]==currency]
                if len(open_pos_cur)>0:
                    if open_pos_cur["isBuy"].tolist()[0]==True:
                        long_short = "long"
                    elif open_pos_cur["isBuy"].tolist()[0]==False:
                        long_short = "short" 

            # The above function is for only analysing open positions and wheter we are in short term positions and long term positons
            
            data = con.get_candles(currency, period='m5', number=250)
            # 250 rows of 5 min candles
            # Input the pair of currency
            # set the candle width of 5 min (m5)

            ohlc = data.iloc[:,[0,1,2,3,8]]
            ohlc.columns = ["Open","Close","High","Low","Volume"]
            signal = trade_signal(renko_merge(ohlc),long_short)
            # We will pass our ohlc and long_shirt in the renko merge which will than call other functions 
            # and take the signal 

            # Based on the signal received we perform our trading
            if signal == "Buy":
                # fxcmpy allows to place more complex orders, this can be done with the method con.open_trade()
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New long position initiated for ", currency)
                
            elif signal == "Sell":
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency)
            
            elif signal == "Close":
                con.close_all_for_symbol(currency)
                print("All positions closed for ", currency)
            
            elif signal == "Close_Buy":
                con.close_all_for_symbol(currency)
                print("Existing Short position closed for ", currency)
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New long position initiated for ", currency)
            
            elif signal == "Close_Sell":
                con.close_all_for_symbol(currency)
                print("Existing long position closed for ", currency)
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency)
    except:
        print("error encountered....skipping this iteration")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TIME FOR WHICH WE WANT TO RUN OUR BOT
# _______________________________________________________________________________________________________________
      
starttime=time.time()
# returns the system's time

timeout = time.time() + 60*60*1  # 60 seconds times 60 meaning the script will run for 1 hr
# For how long do we want to run the script, we want to run it for an hour

while time.time() <= timeout:
    try:
        # While the time is available do this:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(300 - ((time.time() - starttime) % 300.0)) # 5 minute interval between each new execution
    
    except KeyboardInterrupt:
        # The time is over
        print('\n\nKeyboard exception received. Exiting.')
        exit()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Close all positions and exit
# _______________________________________________________________________________________________________________
      
for currency in pairs:
    print("closing all positions for ",currency)
    con.close_all_for_symbol(currency)
con.close()

