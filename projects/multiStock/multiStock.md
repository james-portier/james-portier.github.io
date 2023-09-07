# Trading with several stocks using Coarse and Fine Universe Selection
In this project I use Coarse and Fine Universe Selection to trade several stocks simultaneously, whilst using brokerage to implement Reality Modelling. Click [here](https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_512b3ea0756ee2303df68180e1503088.html) to see the full code and backtesting results. The strategy itself is an extension of the long/short arima forecasting strategy discussed in more detail [here](https://james-portier.github.io/projects/arimaForecasting/arimaQC.html).


## Code
### Initialize Method
In the Initialize method, we specify the backtesting timeframe, an initial cash amount of $1,000,000, the benchmark which we will use to compare the strategy's performance, universe settings, and the number of stocks to pass the coarse and market cap selection:

```python
def Initialize(self):
        warnings.filterwarnings("ignore")

        # Setting the backtest timeframe for the algorithm
        self.SetStartDate(2013, 2, 8)  # Set Start Date
        self.SetEndDate(2018, 2, 7)  # Set Start Date
        
        self.SetCash(1000000)  # Set Strategy Cash
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.SetBenchmark(self.spy) # Used to compare strategy to S&P 500 performance
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction,self.FineSelectionFunction)

        # Number of stocks to pass CoarseSelection process
        self.num_coarse = 10
        # Number of stocks to pass marketcap selection
        self.num_marketcap = 5
        
        # Setting up the data
        self.n_data = (self.EndDate - self.StartDate).days
        self.split = 0.5 # Initialising the desired test/train split over the fixed time period
        self.warmupStart = self.StartDate
        self.days_to_add = int(self.n_data*self.split)
        self.warmupEnd = self.warmupStart + timedelta(days=self.days_to_add)

        # Boolean variable that ensures ARIMA p,q parameters chosen 
        self.pred = []

        # Initializing and filling dictionary used to store the optimal p,q values for each stock
        self.orderDict = {}
    
        # Set Scheduled Event (trade order) Method for our model
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.BeforeMarketClose(self.spy, 5),
                         self.EveryDayBeforeMarketClose)
```


### Coarse Selection Method
This Coarse Selection Method filters stocks based on fundamental data and minimum price criteria. It returns a list of the top N stocks based on dollar volume.
```python
def CoarseSelectionFunction(self, coarse):   
        # Condition that ensures OnData only runs once "warmup period" has passed
        if (self.Time < self.warmupEnd):
            pass

        # drop stocks which have no fundamental data or have too low prices
        selected = sorted([x for x in coarse if (x.HasFundamentalData) 
                    and (float(x.Price) > 5)], key=lambda x: x.DollarVolume, reverse=True)
        top = selected[:self.num_coarse]

        return [i.Symbol for i in top]
```


### Fine Selection Method
This function further filters the selected stocks based on additional fundamental criteria such as P/E ratio, EPS growth, and EBITDA margin. It returns a list of the top N stocks by market capitalization.
```python
def FineSelectionFunction(self, fine):
        # Condition that ensures OnData only runs once "warmup period" has passed
        if (self.Time < self.warmupEnd):
            pass

        # drop stocks which don't have the information we need.
        # you can try replacing those factor with your own factors here
        filtered_fine = sorted([x for x in fine if x.ValuationRatios.PERatio and 
                                x.EarningRatios.DilutedEPSGrowth and x.OperationRatios.EBITDAMargin], 
                                key = lambda k: k.MarketCap, reverse=True)
        filtered_fine_marketcap = filtered_fine[:self.num_marketcap]
        self.symbols = [x.Symbol for x in filtered_fine_marketcap]

        return self.symbols
```


### Portfolio Rebalance Method
This function checks the currently invested stocks and rebalances the portfolio based on the latest long and short positions.
```python
def rebalance_long_short(self):
        # Retrieve currently invested stocks
        invested_stocks = [x.Symbol for x in self.Portfolio.Values if x.Invested]

        newHoldings = self.long + self.short
        for symbol in invested_stocks:
            if symbol not in newHoldings:
                self.Liquidate(symbol)
```


### Strategy Execution
This function predicts and executes trade orders for selected stocks 5 minutes before the market closes. It then fetches historical price data for each stock and fits an ARIMA model, and makes predictions for the next closing price and determines whether to go long or short based on the prediction. Finally, it rebalances the portfolio by liquidating old positions and setting new positions for long and short stocks.

```python
def EveryDayBeforeMarketClose(self) -> None:
        # Condition that ensures OnData only runs once "warmup period" has passed
        if (self.Time < self.warmupEnd):
            return

        # Initialize/clear lists storing short and long positions
        self.long = []
        self.short = []

        for symbol in self.symbols:
            # Fetch history on our universe
            # history = self.History(self.symbol, number_of_data, Resolution.Daily)
            history = self.History(symbol, Resolution.Daily).loc[symbol]
            self.close_prices = history['close'].dropna()
            
            # Assign a recognized frequency to the date index
            self.close_prices.index = pd.date_range(start=self.close_prices.index[0], 
                                                    periods=len(self.close_prices), freq='B')

            # Check if stock is new to universe 
            # If so, optimise hyper-parameters and add to dictionary
            if (symbol not in self.orderDict):
                self.orderDict[symbol] = self.optimiseArima()
            
            p = self.orderDict[symbol][0]
            q = self.orderDict[symbol][1]
            model = sm.tsa.ARIMA(self.close_prices, order=(p,1,q))                                             
            model_fit = model.fit()

            # Forecast the next closing price and add to list of predictions
            # self.pred.append(model_fit.forecast()) # forecasting one time-point ahead
            self.pred += [model_fit.forecast()]

            if (float(self.pred[-1]) < float(self.close_prices[-1])):
                self.short.append(symbol)
            else:
                self.long.append(symbol)

        # Removing old stocks
        self.rebalance_long_short()
            
        # Rebalancing holdings
        # Long positions
        for symbol in self.long:
            self.SetHoldings(symbol, 1/len(self.long))
        # Short Positions
        for symbol in self.short:
            self.SetHoldings(symbol, 0) # set to 0 for this strategy (bad down predictions)
```


## Results
### Strategy Equity

<img src="backtestSummary.png?raw=true"/>

From the strategy equity chart above we see that the algorithmic trading strategy has been quite successful. It has generated a substantial profit relative to the initial capital, resulting in a strong return on investment. Additionally, the PSR suggests that the strategy has managed risk effectively, as the profit generated is significantly higher than the maximum drawdown.

### Performance Metrics
<img src="overview.png?raw=true"/>

`Sharpe Ratio`: The Sharpe Ratio is a risk-adjusted performance metric. Your value of 0.645 suggests that the strategy has achieved a reasonable return for the level of risk taken. 
`Alpha` and `Beta`: Alpha measures the strategy's excess return above a benchmark (in this case, the S&P 500). An alpha of 0.018 suggests that the strategy outperformed the benchmark slightly, and a Beta of 0.571 indicates that the strategy is less volatile than the benchmark.
`Win Rate`: A winrate of 54% indicates that there were more correct predictions than incorrect predictions, and that more than half the trades were profitable.

### Logs
<img src="logs.png?raw=true"/>

Finally, the Logs show that a total of *25 stocks* were traded over the backtesting timeframe. There are several different values for p,q, indicating that a seperate hyperparameter optimisation for each stock was beneficial.
Lastly, the logs reveal that 25 unique stocks were traded throughout the backtesting period. The varying combinations of p and q values employed for each stock underscore the advantage of conducting individual hyperparameter optimization for each stock.




