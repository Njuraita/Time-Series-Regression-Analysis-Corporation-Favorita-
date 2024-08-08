# Time series regression analysis for Corporate Favorita
A time series regression analysis to predict store sales aimed at ensuring that there is always the right quantity of products in stock

## **Business Understanding**

### Project Description
Predicting sales is crucial for business planning. This project aims to build a machine learning model to forecast store sales for Corporation Favorita, a leading grocery retailer in Ecuador. By analyzing sales trends over time, we aim to understand customer behavior, identify seasonal patterns, and recognize factors affecting sales such as oil prices, holidays, and promotions. This analysis will provide valuable insights for grocery retailers, aiding in inventory optimization, cost reduction, and profit increase.

### Project Objective
The primary goal is to create a machine learning model that predicts unit sales for various items sold in Favorita stores. We will:
- Analyze sales trends to understand customer behavior.
- Identify seasonal patterns and other factors affecting sales.
- Provide insights into customer preferences and behavior.

### Project Approach and Methodology
This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework. We will explore various machine learning techniques to achieve accurate sales predictions, including but not limited to:

1. **Time Series Forecasting**: Using ARIMA, SARIMA, or LSTM to detect sales trends and seasonal variations.
2. **Regression Models**: Utilizing linear regression, decision trees, random forests, or gradient boosting to evaluate factors such as promotions, product attributes, and store specifics.

### Hypothesis Testing
- **Null Hypothesis (H0)**: Promotion activities have no significant effect on sales.
- **Alternative Hypothesis (H1)**: Promotion activities have a significant effect on sales.

### Analytical Questions
1. Is the train dataset complete (has all the required dates)?
2. Which dates have the lowest and highest sales for each year (excluding days the store was closed)?
3. Compare the sales for each month across the years and determine which month of which year had the highest sales.
4. Did the earthquake impact sales?
5. Are certain stores or groups of stores selling more products? (Cluster, city, state, type)
6. Are sales affected by promotions, oil prices, and holidays?
7. What analysis can we get from the date and its extractable features?
8. Which product family and stores did the promotions affect?
9. What is the difference between RMSLE, RMSE, MSE (or why is the MAE greater than all of them)?
10. Does the payment of wages in the public sector on the 15th and last days of the month influence store sales?

## **Data Understanding**

What one will need as important packages for this project 

```dotnetcli
# Environment Setup
import dotenv
from dotenv import dotenv_values

# Data Handling
import pyodbc
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Statistical Analysis
import scipy
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine Learning
from pmdarima import auto_arima
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer

# Utility and Miscellaneous
import warnings
warnings.filterwarnings("ignore")
import zipfile
import os
import requests 
import joblib

```
Three of the the train datasets ie ; table dbo.oil, dbo.holidays_events table and dbo.stores table, are in the MYSQL Database after connecting with .env credentials we will Querry them after using the below connection

```dotnetcli
# Connect to the database
connection = pyodbc.connect(connection_string)

```
Two datasets which are part of the train datasets can be found in the release of this repository on GitHub ie; transaction.csv and train.csv and we'll read them directly using url and untrack the datasets in .gitignore

**NB: This projects contains a large dataset and there will be need for use of Git for Large files to track the datasets as a plan B or plan A load directly from github using a url and untruck in .gitignore**

*For each the datasets we do data inspection using the .head(), .duplicated(), .describe(), .isnull(), .info(), .unique()*

We will go ahead and do the following to better understand the data we are working with
1. Perform EDA (Exploratory Data Analysis) and do some plotting where necessarry
2. Then we will answer a few of the analytical Questions as listed above
3. We wil also anaswer our hypothesis testing as listed above
4. Do a decomposition
5. Check for stationerity

## **Data Preparation**




## **Modeling and Evaluation**
*Models in this project*
- SARIMA
- ARIMA
- Linear Regression
- XGB Regressor


*For the best model we will perform hyperparameter tuning using the Randomized Grid Search*

The outcome will determine which model will be used for the test dataset

#### How to Contribute: 
 Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change

#### **For More info:**

*An article on this same project can be viewed here* <https://medium.com/@njerisharon06/time-series-regression-8a35fed56b10>

*To reach me*
Email njerisharon06@gmail.com















 




