import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def simple_arima_forecast(time_series_data, order=(1,1,1), steps=5):
    """
    time_series_data: pandas Series with time index
    order: tuple (p,d,q) parameters for ARIMA
    steps: number of future points to forecast
    """
    # Fit ARIMA model
    model = ARIMA(time_series_data, order=order)
    model_fit = model.fit()

    print(model_fit.summary())

    # Forecast future values
    forecast = model_fit.forecast(steps=steps)

    # Plot original series and forecast
    plt.figure(figsize=(10,5))
    plt.plot(time_series_data, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast', color='red')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.show()

    return forecast

# Usage example:
# Load sample data (monthly shampoo sales dataset)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
data = pd.read_csv(url, index_col=0, parse_dates=True)
data = data.iloc[:, 0]  # convert first column DataFrame to Series
data.index.freq = 'MS'  # set month start frequency

# Call function with sample data
forecasted_values = simple_arima_forecast(data, order=(1,1,1), steps=5)
print(forecasted_values)
