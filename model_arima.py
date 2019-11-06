import os
import time
import logging
import numpy as np
from prometheus_api_client import Metric
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
from pandas.plotting import autocorrelation_plot

class MetricPredictor:
    """docstring for Predictor."""

    model_name = "SARIMA"
    model_description = "Forecast value based on fourier analysis"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize metric object."""
        self.metric = Metric(metric, rolling_data_window_size)


    def sarima_exploration(self, input, range, freq):

        model = SARIMAX(input, order=(1, 2, 2), seasonal_order=(2, 2, 2, freq), enforce_stationarity=True,
                        enforce_invertibility=False)
        model_fit = model.fit(dsip=-1)
        forecast = model_fit.forecast(range)
        return forecast


    def train(self, metric_data=None, prediction_duration=15, freq="15Min"):
        """Train the Prophet model and store the predictions in predicted_df."""
        sfrequency = 96
        if freq == '1h':
            sfrequency = 24
        elif freq == '30Min':
            sfrequency = 48
        elif freq == '15Min':
            sfrequency = 96
        # convert incoming metric to Metric Object
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained
        data = self.metric.metric_values
        vals = np.array(data["y"].tolist())

        _LOGGER.debug("training data start time: %s", self.metric.start_time)
        _LOGGER.debug("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")

        forecast_values = self.sarima_exploration(
            vals, prediction_duration, sfrequency
        )

        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast_values)

        _LOGGER.debug("Creating Dummy Timestamps.....")

        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="min"
        )

        _LOGGER.debug("Computing Bounds .... ")
        upper_bound = np.array(
            [
                (
                        float(np.ma.average(
                            forecast_values[:i], axis=0,
                            weights=np.linspace(0, 1, num=len(forecast_values[:i]))))
                        + (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        upper_bound[0] = np.mean(forecast_values[0])
        upper_bound[1] = np.mean(forecast_values[:1])
        # to account for no std of a single value
        lower_bound = np.array(
            [
                (
                        float(np.ma.average(
                            forecast[:i], axis=0,
                            weights=np.linspace(0, 1, num=len(forecast_values[:i])), ))
                        - (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        lower_bound[0] = np.mean(forecast_values[0])
        lower_bound[1] = np.mean(forecast_values[:1])


        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast_values)

        # create series and index into predictions_dict

        _LOGGER.debug("Formatting Forecast to Pandas ..... ")
        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index("timestamp")

        self.predicted_df = forecast
        _LOGGER.debug(forecast)


    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        nearest_index = self.predicted_df.index.get_loc(
            prediction_datetime, method="nearest"
        )
        return self.predicted_df.iloc[[nearest_index]]