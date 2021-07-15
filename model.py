"""doctsring for packages."""
import datetime
import logging
import pandas
from prophet import Prophet
from prometheus_api_client import Metric

# Set up logging
_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "prophet"
    model_description = "Forecasted value from Prophet model"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

    def train(self, metric_data=None, prediction_duration=15):
        """Train the Prophet model and store the predictions in predicted_df."""
        prediction_freq = "1MIN"
        # convert incoming metric to Metric Object
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained
        self.model = Prophet(
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
        )

        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")

        self.model.fit(self.metric.metric_values)
        future = self.model.make_future_dataframe(
            periods=int(prediction_duration),
            freq=prediction_freq,
            include_history=False,
        )
        forecast = self.model.predict(future)
        forecast["timestamp"] = forecast["ds"]
        forecast = forecast[["timestamp", "yhat", "yhat_lower", "yhat_upper"]]
        forecast = forecast.set_index("timestamp")
        self.predicted_df = forecast
        _LOGGER.debug(forecast)

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        nearest_index = self.predicted_df.index.get_loc(
            prediction_datetime, method="nearest"
        )
        return self.predicted_df.iloc[[nearest_index]]
