import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
file_path = 'boardgames.csv'
df = pd.read_csv(file_path)

df = df[df['yearpublished'] > 1600]

# Convert 'yearpublished' to integer, handle NaNs
df['yearpublished'] = pd.to_numeric(df['yearpublished'], errors='coerce').dropna().astype(int)

# Ensure that all yearpublished entries are within a reasonable range
df = df[(df['yearpublished'] >= 1000) & (df['yearpublished'] <= 3000)]

# Convert to PeriodIndex directly from integers
df['yearpublished'] = df['yearpublished'].apply(lambda x: pd.Period(year=x, freq='Y'))

# Aggregate data to calculate the average complexity by year
time_series_data = df.groupby('yearpublished')['avgweight'].mean()

# Check stationarity
result = adfuller(time_series_data.values)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Plotting the time series
plt.figure(figsize=(10, 5))
plt.plot(time_series_data.index.astype(str), time_series_data.values)
plt.title('Average Game Complexity Over Time')
plt.xlabel('Year')
plt.ylabel('Average Complexity (avgweight)')
plt.xticks(rotation=45)
plt.show()

# ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
plot_acf(time_series_data.diff().dropna(), lags=15, ax=ax1)
plot_pacf(time_series_data.diff().dropna(), lags=15, ax=ax2, method='ywm')  # Changed method to 'ywm'
plt.show()

# Fitting the ARIMA model
model = ARIMA(time_series_data, order=(1,1,1))  # Adjust p, d, q based on ACF and PACF
model_fit = model.fit()

# Print out the summary
print(model_fit.summary())

# Forecast
forecast = model_fit.get_forecast(steps=5)  # Forecasting next 5 years
forecast_index = pd.period_range(start=time_series_data.index[-1], periods=6, freq='Y')[1:]
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plot the forecast alongside the historical data
plt.figure(figsize=(10, 5))
plt.plot(time_series_data.index.astype(str), time_series_data, label='Historical')
plt.plot(forecast_index.astype(str), forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_index.astype(str), forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast of Game Complexity')
plt.xlabel('Year')
plt.ylabel('Forecasted Average Complexity')
plt.legend()
plt.xticks(rotation=45)
plt.show()

