"""doctsring for packages."""
import logging
from prometheus_api_client import Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Set up logging
_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "lstm"
    model_description = "Forecasted value from Lstm model"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d", number_of_feature=10, validation_ratio=0.2,
                 parameter_tuning=True):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

        self.number_of_features = number_of_feature
        self.scalar = MinMaxScaler(feature_range=(0, 1))
        self.parameter_tuning = parameter_tuning
        self.validation_ratio = validation_ratio

    def prepare_data(self, data):
        """Prepare the data for LSTM."""
        train_x = np.array(data[:, 1])[np.newaxis, :].T

        for i in range(self.number_of_features):
            train_x = np.concatenate((train_x, np.roll(data[:, 1], -i)[np.newaxis, :].T), axis=1)

        train_x = train_x[:train_x.shape[0] - self.number_of_features, :self.number_of_features]

        train_yt = np.roll(data[:, 1], -self.number_of_features + 1)
        train_y = np.roll(data[:, 1], -self.number_of_features)
        train_y = train_y - train_yt
        train_y = train_y[:train_y.shape[0] - self.number_of_features]

        train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
        return train_x, train_y

    def get_model(self, lstm_cell_count, dense_cell_count):
        """Build the model."""
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(1, self.number_of_features)))
        model.add(LSTM(lstm_cell_count))
        model.add(Dense(dense_cell_count))
        model.add(Dense(1))
        return model

    def train(self, metric_data=None, prediction_duration=15):
        """Train the model."""
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # normalising
        metric_values_np = self.metric.metric_values.values
        scaled_np_arr = self.scalar.fit_transform(metric_values_np[:, 1].reshape(-1, 1))
        metric_values_np[:, 1] = scaled_np_arr.flatten()

        if self.parameter_tuning:
            x, y = self.prepare_data(metric_values_np)
            lstm_cells = [2 ** i for i in range(5, 8)]
            dense_cells = [2 ** i for i in range(5, 8)]
            loss = np.inf
            lstm_cell_count = 0
            dense_cell_count = 0
            for lstm_cell_count_ in lstm_cells:
                for dense_cell_count_ in dense_cells:
                    model = self.get_model(lstm_cell_count_, dense_cell_count_)
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    history = model.fit(np.asarray(x).astype(np.float32),
                                        np.asarray(y).astype(np.float32),
                                        epochs=50, batch_size=512, verbose=0,
                                        validation_split=self.validation_ratio)
                    val_loss = history.history['val_loss']
                    loss_ = min(val_loss)
                    if loss > loss_:
                        lstm_cell_count = lstm_cell_count_
                        dense_cell_count = dense_cell_count_
                        loss = loss_
            self.lstm_cell_count = lstm_cell_count
            self.dense_cell_count = dense_cell_count
            self.parameter_tuning = False

        model = self.get_model(self.lstm_cell_count, self.dense_cell_count)
        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")
        data_x, data_y = self.prepare_data(metric_values_np)
        _LOGGER.debug(data_x.shape)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(np.asarray(data_x).astype(np.float32), np.asarray(data_y).astype(np.float32), epochs=50, batch_size=512)
        data_test = np.asarray(metric_values_np[-self.number_of_features:, 1]).astype(np.float32)
        forecast_values = []
        prev_value = data_test[-1]
        for i in range(int(prediction_duration)):
            prediction = model.predict(data_test.reshape(1, 1, self.number_of_features)).flatten()[0]
            curr_pred_value = data_test[-1] + prediction
            scaled_final_value = self.scalar.inverse_transform(curr_pred_value.reshape(1, -1)).flatten()[0]
            forecast_values.append(scaled_final_value)
            data_test = np.roll(data_test, -1)
            data_test[-1] = curr_pred_value
            prev_value = data_test[-1]

        dataframe_cols = {"yhat": np.array(forecast_values)}

        upper_bound = np.array(
            [
                (
                        forecast_values[i] + (np.std(forecast_values[:i]) * 2)
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
                        forecast_values[i] - (np.std(forecast_values[:i]) * 2)
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
