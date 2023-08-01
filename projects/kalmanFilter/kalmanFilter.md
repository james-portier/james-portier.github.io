# Pair Trading using a Kalman Filter

**Project description:** This project involves the use of Kalman Filters  The full back testing results can be viewed [here](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_bb0e42fcff0513e3f0f7020dd838d399.html)
## Kalman Filters

A _Kalman Filter_ is a state-space model designed for linear dynamic systems, where the state varies with time and changes are represented linearly. Its main purpose is to estimate unknown states of a variable based on past values. The filter predicts (estimates) the current state of the variable and the uncertainty associated with the estimate. As new data becomes available, these estimates are then updated. 


Kalman filters have diverse applications - in this case, we will employ a Kalman Filter to estimate the hedge ratio between a pair of equities. 
## Trading strategy

This trading strategy was inspired by results I achieved through fitting an ARIMA model in a jupter notebook. The details are as follows:
1. Train an ARIMA model on the first half of the back testing period (called a `warmup period`in quantconnect), and use this model to optimise the hyperparameters (p and q)
2. For days remaining after the warmup period has ended (second half of the back testing period):
    - Fit an ARIMA model on all the available data, using the optimal values for p,q calculated in step 1
    - Produce a next day forecast of JPM's closing price
      - If the forecast is higher than the previous closing price (an upwards prediction), update strategy holdings to a `long` position on JPM's stock
      - Else, maintain a neutral position, or `liquidate` the stock if currently holding a long position


## Code

### Custom KalmanFilter class 

Although there are various pre-existing libraries that implement a Kalman filter, for this project I chose to create my own custom Kalman filter class. 
The equations needed to implement this are as follows:

The article I used to help me with this can be found [here](https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter/)

```python
#region imports
```

This strategy is especially profitable when the market is performing poorly.
The profit is resulted from mispricing, and mispricings are likely to happen when the market goes down or volatility increases.
