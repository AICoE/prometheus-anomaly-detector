import os
import time
import logging
import numpy as np
import mlflow
import mlflow.sklearn
from itertools import product

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



    def sarima_exploration(self, input, range, freq, p, q, d, P, Q, D):

        model = SARIMAX(input, order=(p, d, q), seasonal_order=(P, D, Q, freq), enforce_stationarity=True,
                        enforce_invertibility=False)
        model_fit = model.fit(dsip=-1)
        forecast = model_fit.forecast(range)
        return forecast

    def _mlflow_sarima(self, train, test):
        mlflow.set_experiment("/Shared/experiments/cryptocurrency/analysis-forecasting-2")
        Qs = range(0, 4)
        qs = range(0, 5)
        Ps = range(0, 6)
        ps = range(0, 6)
        D = 2
        d = 2
        parameters = product(ps, qs, Ps, Qs)
        parameters_list = list(parameters)
        len(parameters_list)

        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')
        for param in parameters_list:
            with mlflow.start_run(run_name='SARIMAX_param'):
                mlflow.log_param('order-Qs', param[0])
                mlflow.log_param('order-qs', param[1])
                mlflow.log_param('seasonal-order-Ps', param[2])
                mlflow.log_param('seasonal-order-ps', param[3])

                try:
                    model = SARIMAX(train, order=(param[0], d, param[1]),
                                    seasonal_order=(param[2], D, param[3], 24)).fit(disp=-1)

                except ValueError:
                    print('bad parameter combination:', param)
                    continue

                aic = model.aic
                if aic < best_aic:
                    best_model = model
                    best_aic = aic
                    best_param = param
                results.append([param, model.aic])

                # log metric
                mlflow.log_metric('aic', aic)
                mlflow.log_metric('dickey-fuller-test', adfuller(model.resid[13:])[1])

                # log artifact: model summary
                plt.rc('figure', figsize=(12, 7))
                plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties='monospace')
                plt.axis('off')
                plt.tight_layout()
                summary_fn = 'model_sarimax_summary_{}_{}_{}_{}.png'.format(param[0], param[1], param[2], param[3])
                plt.savefig(summary_fn)
                mlflow.log_artifact(summary_fn)  # logging to mlflow
                plt.close()

                # log artifact: diagnostics plot
                model.plot_diagnostics(figsize=(15, 12))
                fig1_fn = 'figure_diagnostics_{}_{}_{}_{}.png'.format(param[0], param[1], param[2], param[3])
                plt.savefig(fig1_fn)
                mlflow.log_artifact(fig1_fn)  # logging to mlflow
                plt.close()

                # log artifact: residuals and pacf plot
                plt.subplot(211)
                model.resid[13:].plot()
                plt.ylabel(u'Residuals')
                ax = plt.subplot(212)
                plot_acf(model.resid[13:].values.squeeze(), lags=12, ax=ax)
                plt.tight_layout()
                fig2_fn = 'figure_res_pacf_{}_{}_{}_{}.png'.format(param[0], param[1], param[2], param[3])
                plt.savefig(fig2_fn)
                mlflow.log_artifact(fig2_fn)  # logging to mlflow
                plt.close()

    def train(self, metric_data=None, prediction_duration=15, freq="15Min", p,d,q, P,D,Q):
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
            vals, prediction_duration, sfrequency, p,d,q,P,D,Q
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