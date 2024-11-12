### Developed by : SAFEEQ FAZIL A
### Register no: 212222240086
### Date: 26.10.2024
# EX.NO.09        A Project for forecasting using ARIMA model in Python

 

### AIM:
To Create a project on Time series analysis on Vegetable price forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Load and prepare the data.

2. Filter the data for the specified commodity.
 
3. Resample the data to a monthly average for simplicity.
 
4. Apply the ARIMA model for forecasting.
    
5. Plot the forecasted values.
   
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the dataset
file_path = '/content/vegetable.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Select a specific commodity (for example, 'Tomato Big(Nepali)')
commodity = 'Tomato Big(Nepali)'
commodity_data = data[data['Commodity'] == commodity]

# Set the Date column as the index and resample to monthly average
commodity_data.set_index('Date', inplace=True)
monthly_data = commodity_data['Average'].resample('M').mean()

# Handle NaN values (if any) by filling with previous valid value
monthly_data = monthly_data.fillna(method='ffill')  

# Split the data into training and testing sets (last 12 months as test data)
train_data = monthly_data[:-12]
test_data = monthly_data[-12:]

# Fit the ARIMA model on the training data
model = ARIMA(train_data, order=(1, 1, 1))
arima_result = model.fit()

# Forecast the next 12 months
forecast = arima_result.forecast(steps=12)

# Ensure forecast and test_data have the same index for comparison
# This is important to avoid alignment issues when calculating RMSE
forecast = forecast.reindex(test_data.index) #reindexing to make dataframes aligned


# Calculate the Root Mean Squared Error (RMSE)
# convert forecast values to NumPy array for compatibility
rmse = sqrt(mean_squared_error(test_data, forecast.to_numpy())) 
print(f'Root Mean Squared Error: {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data', color='orange')
plt.plot(forecast, label='Forecasted Data', color='red')  
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'ARIMA Forecasting for {commodity} with RMSE: {rmse:.2f}')
plt.legend()
plt.show()

# Print forecasted values
print("Forecasted values for the next 12 months:")
print(forecast)

```

### OUTPUT:
![image](https://github.com/user-attachments/assets/b1020b31-011b-46b8-b645-75d38b3e7fa2)



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
