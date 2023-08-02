# Pairs Trading Strategy using a Kalman Filter

This project involves the use of Kalman Filters to implement a long-short pairs trading strategy. Click [here](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_86f1d7d37553e53a1b67c4baf6874ddc.html)  to see the full back testing results.


## Trading strategy
The idea behind the strategy is to take two equities that are cointegrated and create a long-short portfolio. The premise of this is that the spread between the value of our two positions should be mean-reverting. Anytime the spread deviates from its expected value, one of the assets moved in an unexpected direction and is due to revert back. When the spread diverges, we will take advantage of this by going long or short on the spread.

The pair of equities we will use in this project are `ING` and `TCB`. The choice of this pair was informed by the results of a [research article](https://www.quantconnect.com/research/15347/intraday-dynamic-pairs-trading-using-correlation-and-cointegration-approach/p1) that found pairs of cointegrated (and correlated) equities. The results from this showed that the `ING`-`TCB` pair ranked the highest in terms of ADF test value and correlation coefficient.

The synthetic "spread" between `ING` and `TCB` is the time series that we are actually interested in longing or shorting. The `Kalman Filter` is used to dynamically track the hedging ratio between the two in order to keep the spread stationary (and hence mean reverting).


## Kalman Filter

A _Kalman Filter_ is a state-space model designed for linear dynamic systems, where the state varies with time and changes are represented linearly. Its main purpose is to estimate unknown states of a variable based on past values. The filter predicts (estimates) the current state of the variable and the uncertainty associated with the estimate. As new data becomes available, these estimates are then updated. 

<img src="diagram.png?raw=true"/>

Kalman filters have diverse applications - in this case, we will employ a Kalman Filter to estimate the hedge ratio between `ING` and `TCB`. 

## Code

### Custom KalmanFilter class 

Although there are various pre-existing libraries that implement a Kalman filter, for this project I chose to create my own custom Kalman filter class in order to improve my understanding. 

#### Initialize
First we initialize the variables we'll need for to update the Kalman Filter. These are as follows:
- delta: Constant used to calculate the initial state variance `R`
- $w_t$: Weight matrix used to calculate the initial state variance
- $v_t$: Represents the variance of the measurement noise in the Kalman Filter
- $\theta$: 1-d array with 2 elements that stores the state estimates
- $P$: 2x2 array representing the variance-covariance matrix
- $R$: Prior state variance-covariance matrix
- qty: Integer representing the quantity used for trading positions

```python
class KalmanFilter:
    def __init__(self):
        """Initializer function"""
        self.delta = 1e-4
        self.wt = self.delta / (1 - self.delta) * np.eye(2)
        self.vt = 1e-3
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None #
        self.qty = 25000 
```


#### Updating the Kalman Filter
Now that the required variables have been initialized, we can implement the Kalman Filter using the following equations:

<img src="kalmanEqns.png?raw=true"/>

where 
- $a_t$: Prior mean
- $R_t$: Prior variance-covariance
    - Prior ~ Multivariate-Normal($a_t$, $R_t$)
- $m_t$: Posterior mean
- $C_t$: Posterior state variance-covariance matrix
    - Posterior ~ Multivariate-Normal($m_t$, $C_t$)
- $e_t$: Error term (difference between the actual observation and the prediction)
- $F_t$: Observation matrix of the latest prices
- $Q_t$: Variance of the predictions
- $A_t$: Kalman gain, used to update the posterior state estimate


The article I used to help me with this can be found [here](https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter/)

```python
    def update(self, price_one, price_two):
            """
            Updates the Kalman Filter 
    
            Args:
                price_one (float): Latest price of the first asset 
                price_two (float): Latest price of the second asset 
    
            Returns:
                et (float): Forecast error of the Kalman Filter update.
                sqrt_Qt (float): Standard deviation of the predictions of observations.
                hedge_quantity (int): Calculated hedge quantity for generating trading signals.
            """
            # Create the observation matrix of the latest prices and the intercept value (1.0)
            F = np.asarray([price_one, 1.0]).reshape((1, 2))
            y = price_two
    
            # Update prior state variance-covariance matrix `self.R`
            if self.R is not None:
                self.R = self.C + self.wt
            else: # Initialized to zero if this is the first update
                self.R = np.zeros((2, 2))
    
            # Calculate the Kalman Filter update
            yhat = F.dot(self.theta) # Prediction of new observation
            et = y - yhat # Error
    
            # Calculate the variance of the predictions
            Qt = F.dot(self.R).dot(F.T) + self.vt
    
            # Calculate the Kalman Gain
            At = self.R.dot(F.T) / Qt
    
            # Update posterior state estimate
            self.theta = self.theta + At.flatten() * et
    
            # Update posterior state variance-covariance matrix
            self.C = self.R - At * F.dot(self.R)
    
            # Calculate the hedge quantity using estimated value of `theta`
            hedge_quantity = int(floor(self.qty*self.theta[0]))
            
            return et, np.sqrt(Qt), hedge_quantity 
```


### Trading Algorithm
Now that the Kalman Filter class has been created, we can implement and evaluate the Pairs Trading strategy using QuantConnect's backtesting API. 

#### Initialize
First we need to define the initialize method, specifying the backtest timeframe, strategy cash, and the brokerage model to be used. We also define an event scheduler that will prompt the algorithm to trade every day, 5 minutes before markets close.

```python
#region imports
from AlgorithmImports import *
#endregion

import numpy as np
from math import floor
from KalmanFilter import KalmanFilter

class PairsTradingStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2016, 1, 1)  # Set Start Date
        self.SetEndDate(2016, 3, 8)
        self.SetCash(1000000)  # Set Strategy Cash
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())

        self.symbols = [self.AddEquity(x, Resolution.Minute).Symbol for x in ['ING', 'TCB']]
        self.kf = KalmanFilter()
        self.invested = None
    
        self.Schedule.On(self.DateRules.EveryDay('ING'), self.TimeRules.BeforeMarketClose('ING', 5), self.UpdateAndTrade)
```

#### Trade execution

To create the trading rules we need to determine when the spread has moved too far from its expected value. We do this by considering a multiple of the standard deviation of the spread and using these as the bounds (we _long_ the spread if the forecast error drops below the negative standard deviation of the spread, and _short_ the spread if the forecast error exceeds the positive standard deviation of the spread).

Formally, the rules are specified as:
1. $e_t \lt - \sqrt{Q_t}$ - Long the spread: Go long $N$ shares of ING and go short $\lfloor{\theta_t^0N}\rfloor$ units of TCB
2. $e_t \geq - \sqrt{Q_t}$ - Exit long: Close all long positions of ING and TCB
3. $e_t \gt \sqrt{Q_t}$ - Short the spread : Go short $N$ shares of ING and go long
$\lfloor{\theta_t^0N}\rfloor$ units of TCB
5. $e_t \leq \sqrt{Q_t}$ - Exit short: Close all short positions of ING and TCB

where $e_t$ : Error term, $Q_t$: Variance of predictions, $\theta_0^t$: Dynamic hedge ratio at time t

We implement this using `market orders` to place buy or sell orders at the current market price for the specified equities:

```python
def UpdateAndTrade(self):
        # Get recent price and holdings information
        equity1 = self.CurrentSlice[self.symbols[0]].Close
        equity2 = self.CurrentSlice[self.symbols[1]].Close
        holdings = self.Portfolio[self.symbols[0]]
        
        # Calculate forecast error, prediction standard deviation, and hedging quantity 
        # using the Kalman Filter
        forecast_error, prediction_std_dev, hedge_quantity = self.kf.update(equity1, equity2)
        
        # If we're not currently invested, check current spread 
        if not holdings.Invested:
            # Long the spread
            if forecast_error < -prediction_std_dev:
                self.MarketOrder(self.symbols[1], self.kf.qty)
                self.MarketOrder(self.symbols[0], -hedge_quantity)

            # Short the spread
            elif forecast_error > prediction_std_dev:
                self.MarketOrder(self.symbols[1], -self.kf.qty)
                self.MarketOrder(self.symbols[0], hedge_quantity)

        # If we are invested, check whether assets have reverted back
        if holdings.Invested:
            # Close long position
            if holdings.IsShort and (forecast_error >= -prediction_std_dev):
                self.Liquidate()
                self.invested = None
            
            # Close short position
            elif holdings.IsLong and (forecast_error <= prediction_std_dev):
                self.Liquidate()
                self.invested = None
```


## Results

### Strategy Equity

<img src="backtestSummary.png?raw=true"/>

### Performance Metrics
<img src="overview.png?raw=true"/>

### Conclusions
Upon analyzing the strategy's performance over the specified backtesting period, we observe significant positive indicators, such as a promising `Alpha` (0.211) and a high `Sharpe Ratio` (3.262). These metrics suggest that the strategy has the potential to outperform the market and is relatively less risky in comparison to its returns. However, it is essential to acknowledge a caveat: the specific backtesting time period was deliberately chosen due to the anticipated cointegration and high correlation between ING and TCB during that timeframe. This introduced a look-ahead bias, implying that the strategy might not perform as effectively if a different backtesting period were employed. Additionally, this strategy is especially profitable when the market is performing poorly - the profit is resulted from mispricing, and mispricings are likely to happen when the market goes down or volatility increases.

To ensure greater robustness, it is recommended to backtest the strategy with multiple pairs and over a longer timeframe.


## End note
Whilst initially struggling to wrap my head around Kalman Filters, creating one from scratch greatly helped with my understanding. Using it to implement a pair trading strategy was interesting and it was satisfying to find a backtesting period in which this strategy performed well. 
