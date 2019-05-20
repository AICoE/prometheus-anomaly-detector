import datetime
import pandas as pd
import numpy as np
from metric import Metric
from prometheus_client import Gauge
from numpy import fft


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "Fourier"
    model_description = "Forecast value based on fourier analysis"
    model = None
    predicted_df = None
    metric = None
    predicted_metric_gauge = None

    def __init__(self, metric):
        self.metric = Metric(metric)

    def fourier_extrapolation(self, input_series, n_predict, n_harmonics):
        n = input_series.size
        t = np.arange(0, n)
        p = np.polyfit(t, input_series, 1)
        input_no_trend = input_series - p[0] * t
        frequency_domain = fft.fft(input_no_trend)
        frequencies = fft.fftfreq(n)
        indexes = np.arange(n).tolist()
        indexes.sort(key=lambda i: np.absolute(frequencies[i]))

        time_steps = np.arange(0, n + n_predict)
        restored_signal = np.zeros(time_steps.size)

        for i in indexes[:1 + n_harmonics * 2]:
            amplitude = np.absolute(frequency_domain[i]) / n
            phase = np.angle(frequency_domain[i])
            restored_signal += amplitude * np.cos(2 * np.pi * frequencies[i]
                                                  * time_steps + phase)

        return restored_signal + p[0] * time_steps



    def train(self, metric_data, prediction_range = 1440):

        # convert incoming metric to Metric Object
        self.metric = self.metric + Metric(metric_data)  # !!!!!! Memory bloat !!!!
        total_label_num = len(self.metric.metric_values)
        PREDICT_DURATUION = prediction_range
        current_label_num = 0
        limit_iterator_num = 0
        predictions_dict = {}
        data = self.metric.metric_values
        vals = np.array(data["y"].tolist())
        print("Training Model .....")
        forecast_values = self.fourier_extrapolation(vals, prediction_range, int(len(vals)/3))
        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast_values)

        # find most recent timestamp from original data and extrapolate new timestamps
        print("Creating Dummy Timestamps.....")
        minimum_time = min(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(minimum_time, periods=len(forecast_values), freq='min')

        # create dummy upper and lower bounds

        print("Computing Bounds .... ")
        upper_bound = np.mean(forecast_values) + np.std(forecast_values)
        lower_bound = np.mean(forecast_values) - np.std(forecast_values)
        dataframe_cols["yhat_upper"] = np.full((len(forecast_values)), upper_bound)
        dataframe_cols["yhat_lower"] = np.full((len(forecast_values)), lower_bound)

        # create series and index into predictions_dict
        print("Formatting Forecast to Pandas ..... ")

        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index('timestamp')

        self.predicted_df = forecast
        print(forecast)

    def predict_value(self, prediction_datetime):
        """
        This function returns the predicted value of the metric for the prediction_datetime
        """
        nearest_index = self.predicted_df.index.get_loc(prediction_datetime, method="nearest")
        return self.predicted_df.iloc[[nearest_index]]


def get_df_from_metric_json(metric):
    """
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....

    This method can also be used to update an existing dictionary with new data
    """
    df = pd.DataFrame(metric["values"], columns=["ds", "y"]).apply(
        pd.to_numeric, args=({"errors": "coerce"})
    )
    df["ds"] = pd.to_datetime(df["ds"], unit="s")

    return df
