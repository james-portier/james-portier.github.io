# Trading with several stocks using Coarse and Fine Universe Selection
In this project I use Coarse and Fine Universe Selection to trade several stocks simultaneously,  Click [here](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_86f1d7d37553e53a1b67c4baf6874ddc.html) to see the full code and backtesting results.


## Trading strategy
The idea behind this strategy is to take two equities that are cointegrated and create a long-short portfolio. The premise of this is that the spread between the values of these two positions tends to revert to its mean over time. Whenever the spread deviates from its anticipated value, it indicates that one of the assets has moved in an unexpected direction and is likely to revert back. When the spread diverges, we will take advantage of this by going long or short on the spread.

The pair of equities we will use in this project are `ING` and `TCB`. The choice of this pair was informed by the results of a [research article](https://www.quantconnect.com/research/15347/intraday-dynamic-pairs-trading-using-correlation-and-cointegration-approach/p1) that found pairs of cointegrated (and correlated) equities over a specific timeframe. The results from this showed that the `ING`-`TCB` pair ranked the highest in terms of ADF test value and correlation coefficient.

The synthetic "spread" between `ING` and `TCB` is the time series that we are actually interested in longing or shorting. The `Kalman Filter` is used to dynamically track the hedging ratio between the two in order to keep the spread stationary (and hence mean reverting).


## Kalman Filter

A _Kalman Filter_ is a state-space model designed for linear dynamic systems, where the state varies with time and changes are represented linearly. Its main purpose is to estimate unknown states of a variable based on past values. The filter predicts (estimates) the current state of the variable and the uncertainty associated with the estimate. As new data becomes available, these estimates are then updated. 

<img src="diagram.png?raw=true"/>

Kalman filters have diverse applications - in this case, we will employ a Kalman Filter to estimate the hedge ratio between `ING` and `TCB`. 

## Code

### Custom KalmanFilter class 
