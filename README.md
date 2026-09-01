# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 28.08.2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv('/content/Walmart_Sales.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Aggregate sales by date to get a total weekly sales time series
ts = df.groupby('Date')['Weekly_Sales'].sum().sort_index()

display(ts.head())
plt.figure(figsize=(12, 5))
plt.plot(ts)
plt.title('Total Weekly Sales')
plt.show()
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] <= 0.05:
        print("Result: The series is stationary.")
    else:
        print("Result: The series is non-stationary.")

adf_test(ts)

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(ts, ax=axes[0], lags=40)
plot_pacf(ts, ax=axes[1], lags=40, method='ywm')
plt.show()
# Split data: use last 10 weeks for testing
train, test = ts[:-10], ts[-10:]

# 1. AutoRegressive Model (using lag=4 based on seasonality/PACF)
ar_model = AutoReg(train, lags=4).fit()
ar_pred = ar_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# 2. Exponential Smoothing (Holt-Winters)
es_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=52).fit()
es_pred = es_model.forecast(len(test))

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(train.index[-20:], train[-20:], label='Train')
plt.plot(test.index, test, label='Actual', color='black', linewidth=2)
plt.plot(test.index, ar_pred, label='AR Prediction', linestyle='--')
plt.plot(test.index, es_pred, label='Exp Smoothing Prediction', linestyle='--')
plt.legend()
plt.title('Walmart Sales Prediction: Actual vs Predicted')
plt.show()

print(f'AR RMSE: {np.sqrt(mean_squared_error(test, ar_pred)):.2f}')
print(f'Exp Smoothing RMSE: {np.sqrt(mean_squared_error(test, es_pred)):.2f}')
from sklearn.metrics import mean_squared_error

# Fit AR model with 13 lags as per algorithm
ar_model_13 = AutoReg(train, lags=13).fit()
ar_pred_13 = ar_model_13.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate MSE
mse = mean_squared_error(test, ar_pred_13)
print(f'AR (13 Lags) Mean Squared Error: {mse:,.2f}')

# Plot test data vs predictions
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Test Data', marker='o')
plt.plot(test.index, ar_pred_13, label='AR (13 Lags) Prediction', linestyle='--', marker='x')
plt.title('Actual vs Predicted Sales (AR Model)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:
<img width="1121" height="660" alt="image" src="https://github.com/user-attachments/assets/434a8136-68ef-46de-a24d-c8fb456dbfcb" />

<img width="1124" height="586" alt="image" src="https://github.com/user-attachments/assets/fdeae1d6-42df-4b44-9d25-55de0e3a1e7a" />
<img width="1116" height="589" alt="image" src="https://github.com/user-attachments/assets/dbe34a0f-73bb-4adc-b9b7-f3920f2d4575" />
<img width="1048" height="510" alt="image" src="https://github.com/user-attachments/assets/10ee7854-921f-4fb0-92f4-bb7c9f37cdbf" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
