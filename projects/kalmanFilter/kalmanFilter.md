# Pair Trading using a Kalman Filter

**Project description:** This project involves the use of Kalman Filters  The full back testing results can be viewed [here](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_bb0e42fcff0513e3f0f7020dd838d399.html)
## Kalman Filters

A _Kalman Filter_ is a state-space model designed for linear dynamic systems, where the state varies with time and changes are represented linearly. Its main purpose is to estimate unknown states of a variable based on past values. The filter predicts (estimates) the current state of the variable and the uncertainty associated with the estimate. As new data becomes available, these estimates are then updated. 

<img src="kalmanFilterDiagram.png?raw=true"/>

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

#### Initialize
First we initialize the variables we'll need for to update the Kalman Filter. These are as follows:
- `delta`: Constant used to calculate the initial state variance `R`
- `wt`: Weight matrix used to calculate the initial state variance
- `vt`: Represents the variance of the measurement noise in the Kalman Filter
- `theta`: 1-d array with 2 elements that stores the state estimates
- `P`: 2x2 array representing the variance-covariance matrix
- `R`: Prior state variance-covariance matrix
- `qty`: Integer representing the quantity used for trading positions

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
        self.qty = 2000 
```


#### Updating the Kalman Filter
Now that the required variables have been initialized, we can implement the Kalman Filter using the following equations:

<img src="kalmanEqns.png?raw=true"/>

where 
- `et`: Error term
- `Ft`: Observation matrix of the latest prices
- `Qt`: Variance of the predictions
- `At`: Kalman gain
- `Ct`: Posterior state variance-covariance matrix

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

This strategy is especially profitable when the market is performing poorly.
The profit is resulted from mispricing, and mispricings are likely to happen when the market goes down or volatility increases.
