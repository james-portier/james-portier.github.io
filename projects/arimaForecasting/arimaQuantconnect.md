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

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
