# -*- coding: utf-8 -*-
"""
Testing my crude implementation of letting backtest/walkforward return
data based on the inputs it recieved (rules,indicators,models)

"""

import numpy as np
import pybroker
from numba import njit
from pybroker import Strategy, StrategyConfig, YFinance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pybroker import highv

def cmma(bar_data, lookback):

    @njit  # Enable Numba JIT.
    def vec_cmma(values):
        # Initialize the result array.
        n = len(values)
        out = np.array([np.nan for _ in range(n)])

        # For all bars starting at lookback:
        for i in range(lookback, n):
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            # Subtract the moving average from value.
            out[i] = values[i] - ma
        return out

    # Calculate with close prices.
    return vec_cmma(bar_data.close)

def buy_low(ctx):
    # If shares were already purchased and are currently being held, then return.
    if ctx.long_pos():
        return
    # If the latest close price is less than the previous day's low price,
    # then place a buy order.
    if ctx.bars >= 2 and ctx.close[-1] < ctx.low[-2]:
        # Buy a number of shares that is equal to 25% the portfolio.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        # Set the limit price of the order.
        ctx.buy_limit_price = ctx.close[-1] - 0.01
        # Hold the position for 3 bars before liquidating (in this case, 3 days).
        ctx.hold_bars = 3

def hhv(bar_data, period):
    return highv(bar_data.high, period)




def buy_cmma_cross(ctx):
    if ctx.long_pos():
        return
    # Place a buy order if the most recent value of the 20 day CMMA is < 0:
    if ctx.indicator('cmma_20')[-1] < 0:
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.hold_bars = 3
        
        z=ctx.indicator('hhv_5')[-1]


def train_slr(symbol, train_data, test_data):
    # Train
    # Previous day close prices.
    train_prev_close = train_data['close'].shift(1)
    # Calculate daily returns.
    train_daily_returns = (train_data['close'] - train_prev_close) / train_prev_close
    # Predict next day's return.
    train_data['pred'] = train_daily_returns.shift(-1)
    train_data = train_data.dropna()
    # Train the LinearRegession model to predict the next day's return
    # given the 20-day CMMA.
    X_train = train_data[['cmma_20']]
    y_train = train_data[['pred']]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test
    test_prev_close = test_data['close'].shift(1)
    test_daily_returns = (test_data['close'] - test_prev_close) / test_prev_close
    test_data['pred'] = test_daily_returns.shift(-1)
    test_data = test_data.dropna()
    X_test = test_data[['cmma_20']]
    y_test = test_data[['pred']]
    # Make predictions from test data.
    y_pred = model.predict(X_test)
    # Print goodness of fit.
    r2 = r2_score(y_test, np.squeeze(y_pred))
    print(symbol, f'R^2={r2}')

    # Return the trained model and columns to use as input data.
    return model, ['cmma_20'] #-> feed cmma_20 to model to get prediction.

def train_slr0(symbol, train_data, test_data):
    # Train
    # Previous day close prices.
    train_prev_close = train_data['close'].shift(1)
    # Calculate daily returns.
    train_daily_returns = (train_data['close'] - train_prev_close) / train_prev_close
    # Predict next day's return.
    train_data['pred'] = train_daily_returns.shift(-1)
    train_data = train_data.dropna()
    # Train the LinearRegession model to predict the next day's return
    # given the 20-day CMMA.
    X_train = train_data[['close']]
    y_train = train_data[['pred']]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test
    test_prev_close = test_data['close'].shift(1)
    test_daily_returns = (test_data['close'] - test_prev_close) / test_prev_close
    test_data['pred'] = test_daily_returns.shift(-1)
    test_data = test_data.dropna()
    X_test = test_data[['close']]
    y_test = test_data[['pred']]
    # Make predictions from test data.
    y_pred = model.predict(X_test)
    # Print goodness of fit.
    r2 = r2_score(y_test, np.squeeze(y_pred))
    print(symbol, f'R^2={r2}')

    # Return the trained model and columns to use as input data.
    return model , ['close'] #-> feed cmma_20 to model to get prediction.

def hold_long(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('slr')[-1] > 0:
            ctx.buy_shares = 100
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('slr')[-1] < 0:
            ctx.sell_shares = 100
            
def hold_long_rls(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('rls')[-1] > 0:
            ctx.buy_shares = 100
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('rls')[-1] < 0:
            ctx.sell_shares = 100            

def hold_long0(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('slr0')[-1] > 0:
            ctx.buy_shares = 100
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('slr0')[-1] < 0:
            ctx.sell_shares = 100      
            
def buy_low(ctx):
    # If shares were already purchased and are currently being held, then return.
    if ctx.long_pos():
        return
    # If the latest close price is less than the previous day's low price,
    # then place a buy order.
    if ctx.bars >= 2 and ctx.close[-1] < ctx.low[-2]:
        # Buy a number of shares that is equal to 25% the portfolio.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        # Set the limit price of the order.
        ctx.buy_limit_price = ctx.close[-1] - 0.01
        # Hold the position for 3 bars before liquidating (in this case, 3 days).
        ctx.hold_bars = 3
        
           
def hhv(bar_data, period):
    return highv(bar_data.high, period)

#%% INITIALIZE
pybroker.disable_caches()
    
#REGISTER
cmma_20 = pybroker.indicator('cmma_20', cmma, lookback=20)
hhv_5 = pybroker.indicator('hhv_5', hhv, period=5)
model_slr0 = pybroker.model('slr0', train_slr0)
model_slr = pybroker.model('slr', train_slr, indicators=[cmma_20,hhv_5])
model_rls = pybroker.model('rls', train_slr, indicators=[cmma_20])

config = StrategyConfig(bootstrap_sample_size=100)
strategy = Strategy(YFinance(), '3/1/2017', '2/19/2021', config)

#%% 0) Rules only
strategy.clear_executions()
strategy.add_execution(buy_low, ['PG', 'AMD'])

df_r,result_r = strategy.backtest()


#%% 1) rule + indicator
#---------------------------------
strategy.clear_executions()
strategy.add_execution(buy_cmma_cross, 'PG', indicators=[cmma_20,hhv_5])
df_ri, result_ri = strategy.backtest(warmup=20)


#%% 2) rules + model
#---------------------------------------
strategy.clear_executions()
strategy.add_execution(hold_long0, ['NVDA'], models=model_slr0)

df_rm,result_rm = strategy.walkforward(
            warmup=20,
            windows=3,
            train_size=0.5,
            lookahead=1,
            calc_bootstrap=False
        )

a=result_rm.orders


#%% 3) rule + indicator + model
#------------------------------------
strategy.clear_executions()
strategy.add_execution(hold_long_rls, ['PG'], models=model_rls)
strategy.add_execution(hold_long, ['NVDA', 'AMD'], models=model_slr)

df_rim,result_rim = strategy.walkforward(
    timeframe="1m",
    warmup=20,
    windows=3,
    train_size=0.5,
    lookahead=1,
    calc_bootstrap=False
)


#%% 4) indicator only
strategy.clear_executions()
strategy.add_execution(None, 'PG', indicators=[cmma_20,hhv_5])
df_i = strategy.backtest(warmup=20)


#%% 5) model only
#---------------------------------------
strategy.clear_executions()
strategy.add_execution(None, ['NVDA'], models=model_slr0)

df_m = strategy.walkforward(
            warmup=20,
            windows=3,
            train_size=0.5,
            lookahead=1,
            calc_bootstrap=False
        )


#%% 6) indicator and model
#---------------------------------------
strategy.clear_executions()
strategy.add_execution(None, ['PG'], models=model_rls)
strategy.add_execution(None, ['NVDA', 'AMD'], models=model_slr)

# df_im =strategy.backtest(train_size=0.5)
df_im = strategy.walkforward(
            warmup=20,
            windows=3,
            train_size=0.5,
            lookahead=1,
            calc_bootstrap=False
        )


#%% Other
o_r=result_r.orders
o_ri=result_ri.orders
o_rm=result_rm.orders
o_rim=result_rim.orders

t_r=result_r.trades
t_ri=result_ri.trades
t_rm=result_rm.trades
t_rim=result_rim.trades
