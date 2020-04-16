"""doctsring for packages."""
import datetime
import logging
import pandas
from fbprophet import Prophet
from prometheus_api_client import Metric

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Set up logging
_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "lstm"
    model_description = "Forecasted value from Lstm model"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

    def train(self, metric_data=None, prediction_duration=15):
        # convert incoming metric to Metric Object
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained
        model = Sequential()
		model.add(LSTM(64,return_sequences=True,input_shape=(1,self.number_of_features)))
		model.add(LSTM(32))
		model.add(Dense(32))
		model.add(Dense(1))
		self.model = model

        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")

        data_x = self.metric.metric_values['y'].values
		for i in range(number_of_features):
  			data_x = np.concatenate((data_x, np.roll(train[:,1],-i)[np.newaxis,:].T), axis=1)
  		data_y = np.roll(self.metric.metric_values['y'].values[:,1],-number_of_features)

  		_LOGGER.debug("printing the shape of processed data")
  		_LOGGER.debug(data.shape)

		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(data_x, data_y, epochs=10, batch_size=1)

		data_test = np.zeros(3)
		data_test[:2] = data_x[-2:]
		data_test[-1] = data_y[-1]

		forecast_values = []
		for i in range(prediction_duration):
			forecast_values.append(model.predict(data_test))
			np.roll(data_test,-1)
			data_test[-1] = forecast_values[-1]


        
		dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast_values)

        upper_bound = np.array(
            [
                (
                    np.ma.average(
                        forecast_values[:i],
                        weights=np.linspace(0, 1, num=len(forecast_values[:i])),
                    )
                    + (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        upper_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        lower_bound = np.array(
            [
                (
                    np.ma.average(
                        forecast_values[:i],
                        weights=np.linspace(0, 1, num=len(forecast_values[:i])),
                    )
                    - (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        lower_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound

        data = self.metric.metric_values
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="min"
        )

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
