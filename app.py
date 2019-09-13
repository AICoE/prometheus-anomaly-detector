"""docstring for packages."""
import time
import os
import logging
from datetime import datetime
import tornado.ioloop
import tornado.web
import tornado
from prometheus_client import Gauge, generate_latest, REGISTRY
from apscheduler.schedulers.tornado import TornadoScheduler
from prometheus_api_client import PrometheusConnect, Metric
from configuration import Configuration
import model

# Set up logging
_LOGGER = logging.getLogger(__name__)

METRICS_LIST = Configuration.metrics_list


PREDICTOR_MODEL_LIST = []

pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=True,
)
for metric in METRICS_LIST:
    # Initialize a predictor for all metrics first
    metric_init = pc.get_current_metric_value(metric_name=metric)
    for unique_metric in metric_init:
        PREDICTOR_MODEL_LIST.append(
            model.MetricPredictor(
                unique_metric,
                rolling_data_window_size=Configuration.rolling_training_window_size,
            )
        )

# A gauge set for the predicted values
GAUGE_DICT = dict()
for predictor in PREDICTOR_MODEL_LIST:
    unique_metric = predictor.metric
    label_list = list(unique_metric.label_config.keys())
    label_list.append("value_type")
    if unique_metric.metric_name not in GAUGE_DICT:
        GAUGE_DICT[unique_metric.metric_name] = Gauge(
            unique_metric.metric_name + "_" + predictor.model_name,
            predictor.model_description,
            label_list,
        )


class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        # update metric value on every request and publish the metric
        for predictor_model in PREDICTOR_MODEL_LIST:
            # get the current metric value so that it can be compared with the
            # predicted values
            current_metric_value = Metric(
                pc.get_current_metric_value(
                    metric_name=predictor_model.metric.metric_name,
                    label_config=predictor_model.metric.label_config,
                )[0]
            )

            prediction = predictor_model.predict_value(datetime.now())
            metric_name = predictor_model.metric.metric_name

            # Check for all the columns available in the prediction
            # and publish the values for each of them
            for column_name in list(prediction.columns):
                GAUGE_DICT[metric_name].labels(
                    **predictor_model.metric.label_config, value_type=column_name
                ).set(prediction[column_name][0])

            # Calculate for an anomaly (can be different for different models)
            anomaly = 1
            if (
                current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0]
            ) and (
                current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0]
            ):
                anomaly = 0

            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


def make_app():
    """Initialize the tornado web app."""
    _LOGGER.info("Initializing Tornado Web App")
    return tornado.web.Application([(r"/metrics", MainHandler), (r"/", MainHandler)])


def train_model(initial_run=False):
    """Train the machine learning model."""
    for predictor_model in PREDICTOR_MODEL_LIST:
        metric_to_predict = predictor_model.metric

        data_start_time = datetime.now() - Configuration.metric_chunk_size
        if initial_run:
            data_start_time = (
                datetime.now() - Configuration.rolling_training_window_size
            )

        # Download new metric data from prometheus
        new_metric_data = pc.get_metric_range_data(
            metric_name=metric_to_predict.metric_name,
            label_config=metric_to_predict.label_config,
            start_time=data_start_time,
            end_time=datetime.now(),
        )[0]

        # Train the new model
        start_time = datetime.now()
        predictor_model.train(
            new_metric_data, Configuration.retraining_interval_minutes
        )
        _LOGGER.info(
            "Total Training time taken = %s, for metric: %s %s",
            str(datetime.now() - start_time),
            metric_to_predict.metric_name,
            metric_to_predict.label_config,
        )


if __name__ == "__main__":
    # Initial run to generate metrics, before they are exposed
    train_model(initial_run=True)

    # Start up the server to expose the metrics.
    app = make_app()
    app.listen(8080)
    scheduler = TornadoScheduler()
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )
    scheduler.add_job(
        train_model, "interval", minutes=Configuration.retraining_interval_minutes
    )
    scheduler.start()
    tornado.ioloop.IOLoop.instance().start()
