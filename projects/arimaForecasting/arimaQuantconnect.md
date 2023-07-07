# ARIMA Forecasting Strategy using QuantConnect

**Project description:** In this project I used QuantConnect's back testing API to implement a long/short strategy. The strategy uses ARIMA forecasting to predict whether the next day closing price will increase or decrease, and adjusts the portfolio holdings accordingly. The full back testing results can be viewed [here]()


## Trading strategy

This trading strategy was inspired by results I achieved through fitting an ARIMA model in a jupter notebook. The details are as follows:
1. Train an ARIMA model on the first half of the back testing period (called a `warmup period`in quantconnect), and use this model to optimise the hyperparameters (p and q)
2. For days remaining after the warmup period has ended (second half of the back testing period):
    - Fit an ARIMA model on all the available data, using the optimal values for p,q calculated in step 1
    - Produce a next day forecast of JPM's closing price
      - If the forecast is higher than the previous closing price (an upwards prediction), update strategy holdings to a `long` position on JPM's stock
      - Else, maintain a neutral position, or `liquidate` the stock if currently holding a long position


## Code

### Library imports 

```python
#region imports
from AlgorithmImports import *
#endregion

# Library imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm # ARIMA
import warnings
```


### Initialize Method

QuantConnect features an `Initialize()` method that is called once at the start of the algorithm. Here we specify settings such as the back testing timeframe, the starting cash amount, and any variables we wish to initialize. The initialize method here consists of the following:
- **Back testing timeframe**: The back testing timeframe consists of all days between January 1st 2013 and January 1st 2018. This specific timeframe was chosen to because it is the same timeframe used in the research notebook, upon which this trading strategy is based.
- **Strategy cash**: A starting cash amount of $100,000 was used here.
- **Warmup Period**: Variables that implement the warmup period were set up such that the algorithm's doesn't start trading until half the warmup period has passed. This ensures the hyperparameters are optimised using a sufficient number of historical closing prices, as well as giving the ARIMA models used to forecast enough data to fit to.
- **EveryDayAfterMarketOpen**: Event scheduler that defines how often the algorithm should make a prediction and trade. Here this is defined to be 5 minutes after markets open, repeated every day (exluding during the warmup period).

```python
class SmoothSkyBlueMosquito(QCAlgorithm):
    def Initialize(self):
        """Initialize Class"""

        # Ignore redundant warnings that may arise during model fitting 
        warnings.filterwarnings("ignore")  

        # Setting the backtest timeframe for the algorithm
        self.SetStartDate(2013, 1, 1)  # Set Start Date
        self.SetEndDate(2018, 1, 1)  # Set Start Date
        
        self.SetCash(100000)  # Set Strategy Cash
        self.symbol = self.AddEquity("JPM", Resolution.Daily).Symbol # Add JPM ticker
        self.SetBenchmark(self.symbol) # Used to compare strategy to a simple long and hold of JPM's stock

        # Splitting the data and defining the warmup period 
        self.n_data = (self.EndDate - self.StartDate).days
        self.split = 0.5 # Initialising the desired test/train split over the fixed time period
        self.warmupStart = self.StartDate
        self.days_to_add = int(self.n_data*self.split)
        self.warmupEnd = self.warmupStart + timedelta(days=self.days_to_add)

        # Boolean variable that ensures ARIMA p,q parameters chosen 
        self.isOptimised = False
        self.p = 0
        self.q = 0

        # Used to log start and end price of trading period
        self.startPrice = 0
        self.endPrice = 0

        # Event scheduler - defines when the algorithm should trade 
        self.Schedule.On(self.DateRules.EveryDay(), 
                self.TimeRules.AfterMarketOpen("JPM", 5),         
                self.EveryDayAfterMarketOpen)

        # List that will keep track of predictions
        self.pred = []
```


### EveryDayAfterMarketOpen()

Here is where the actual trading strategy is implemented. The algorithm is very similar to that used in the research notebook, with the following additions:
- An `IF` statment at the start of the function is included to enforce the warmup period
- A `History()` request is made on every trading day. This is a built in method provided by the `QCAlgorithm` class. This fetches all the available data before the current trading day (starting from 1/1/2013 in this case). We use this to get the most recent closing price.
    - *note: the use of a rolling window would've been more efficient here*
- `Nested IF` statements that update the strategies holdings based on the outcome of the most recent price prediction

```python
def EveryDayAfterMarketOpen(self) -> None:
        """Method that predicts + executes trade order 5 mins after markets open"""
        
        # Condition that ensures OnData only runs once "warmup period" has passed
        if (self.Time < self.warmupEnd):
            return

        modelFailed = False

        # Fetch the latest history on our universe
        history = self.History(self.symbol, Resolution.Daily).loc[self.symbol]
        self.close_prices = history['close'].dropna() # extract the closing prices
        
        # Assign a recognized frequency to the date index
        self.close_prices.index = pd.date_range(start=self.close_prices.index[0], periods=len(self.close_prices), freq='B')

        # Optimise ARIMA parameters on first day after warmup
        if (self.isOptimised == False):
            param = self.optimiseArima()
            self.p = param[0]
            self.q = param[1]
            self.isOptimised = True # Update to True as we only need to optimise once
            # Store close price of first trading day
            self.startPrice = self.Securities[self.symbol].Close 
    
        # Fit the model
        try:
            model = sm.tsa.ARIMA(self.close_prices, order=(self.p, 1, self.q))
            model_fit = model.fit()
            self.pred += [model_fit.forecast()]
        except:
            modelFailed = True
            
        # Update holdings using latest prediction
        if modelFailed: # If model failed to converge, go neutral
            self.SetHoldings(self.symbol, 0)
        elif (float(self.pred[-1]) < float(self.close_prices[-1])):
            self.SetHoldings(self.symbol, 0) # Go neutral in case of "Down" prediction
        else:
            self.SetHoldings(self.symbol, 1) # Go long in case of "Up" prediction
```


### Optimise Arima

Method that returns the optimised hyperparameters p and q

```python
def optimiseArima(self):
        """
        Returns the optimal order for ARIMA model using AIC 
        Args:
        self: QCAlgorithm class
        Returns
        (p,q): array containing optimal values of p and q
        """
        # Initialize
        best_aic = np.inf
        best_order = None
        d = 1 # Use d=1 to achieve stationarity (from research notebook)

        # Manual "grid search" on candidate values for p and q
        for p in range(1,5):  
            for q in range(1,5): 
                try: # use this to handle any exceptions that may arise during fitting
                    model = sm.tsa.ARIMA(self.close_prices, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, q)
                except:
                    continue

        return best_order
```


### End of algorithm logging

The built in event scheduler `OnEndOfAlgorithm()` is used to log the values of p and q used in the model, the prices at the start and end of the trading timeframe, and the return a long and hold strategy would've achieved.

```python
def OnEndOfAlgorithm(self) -> None:
        # Logging the optimal values of p and q used 
        self.Log(f"P - Value: {self.p}")
        self.Log(f"Q - Value: {self.q}")

        # Logging the start and end prices 
        self.endPrice = self.Securities[self.symbol].Close
        self.Log(f"The price of {self.symbol} on the first trading day: {self.startPrice}")
        self.Log(f"The price of {self.symbol} on the final trading day: {self.endPrice}")

        # Calculating the long and hold strategy return
        longHoldRet = self.endPrice / self.startPrice
        self.Log(f"A long and hold strategy would've had a return of: {longHoldRet}")
```

## Backtesting Results


test


### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
