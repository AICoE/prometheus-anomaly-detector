import time
import os
import logging
from datetime import datetime
import tornado.ioloop
import tornado.web
import tornado
from prometheus_client import Gauge, generate_latest, REGISTRY
from apscheduler.schedulers.tornado import TornadoScheduler
import model
from prometheus_api_client import PrometheusConnect
from configuration import Configuration

# Set up logging
_LOGGER = logging.getLogger(__name__)
if os.getenv("FLT_DEBUG_MODE", "False") == "True":
    LOGGING_LEVEL = logging.DEBUG  # Enable Debug mode
else:
    LOGGING_LEVEL = logging.INFO
# Log record format
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s: %(message)s", level=LOGGING_LEVEL)


METRICS_LIST = Configuration.metrics_list


PREDICTOR_MODEL_LIST = []

HEADERS = {"Authorization": "bearer " + Configuration.oath_token}
pc = PrometheusConnect(url=Configuration.prometheus_url, headers=HEADERS, disable_ssl=True)
for metric in METRICS_LIST:
    # Initialize a predictor for all metrics first
    metric_init = pc.get_current_metric_value(metric_name=metric)
    for unique_metric in metric_init:
        PREDICTOR_MODEL_LIST.append(model.MetricPredictor(unique_metric))

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
    def get(self):
        # update metric value on every request and publish the metric
        for predictor_model in PREDICTOR_MODEL_LIST:
            prediction = predictor_model.predict_value(datetime.now())
            metric_name = predictor_model.metric.metric_name
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="yhat"
            ).set(prediction["yhat"][0])
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="yhat_lower"
            ).set(prediction["yhat_lower"][0])
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="yhat_upper"
            ).set(prediction["yhat_upper"][0])

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


def make_app():
    _LOGGER.info("Initializing Tornado Web App")
    return tornado.web.Application([(r"/metrics", MainHandler)])


def train_model():
    for predictor_model in PREDICTOR_MODEL_LIST:
        metric_to_predict = predictor_model.metric

        # Download new metric data from prometheus
        new_metric_data = pc.get_metric_range_data(
            metric_name=metric_to_predict.metric_name,
            label_config=metric_to_predict.label_config,
            start_time="15m",
        )[0]
        # Train the new model
        predictor_model.train(new_metric_data)


if __name__ == "__main__":
    # Initial run to generate metrics, before they are exposed
    train_model()

    # Start up the server to expose the metrics.
    app = make_app()
    app.listen(8080)
    scheduler = TornadoScheduler()
    scheduler.add_job(train_model, "interval", minutes=15)
    scheduler.start()
    tornado.ioloop.IOLoop.instance().start()
