"""docstring for packages."""
import os
import logging
from prometheus_api_client.utils import parse_datetime, parse_timedelta

if os.getenv("FLT_DEBUG_MODE", "False") == "True":
    LOGGING_LEVEL = logging.DEBUG  # Enable Debug mode
else:
    LOGGING_LEVEL = logging.INFO
# Log record format
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s: %(message)s", level=LOGGING_LEVEL)
# set up logging
_LOGGER = logging.getLogger(__name__)


class Configuration:
    """docstring for Configuration."""

    # url for the prometheus host
    prometheus_url = os.getenv("FLT_PROM_URL")

    # any headers that need to be passed while connecting to the prometheus host
    prom_connect_headers = None
    # example oath token passed as a header
    if os.getenv("FLT_PROM_ACCESS_TOKEN"):
        prom_connect_headers = {"Authorization": "bearer " + os.getenv("FLT_PROM_ACCESS_TOKEN")}

    # list of metrics that need to be scraped and predicted
    # multiple metrics can be separated with a ";"
    # if a metric configuration matches more than one timeseries,
    # it will scrape all the timeseries that match the config.
    metrics_list = str(
        os.getenv(
            "FLT_METRICS_LIST", "up{app='openshift-web-console', instance='172.44.0.18:8443'}"
        )
    ).split(";")

    # uri for the mlflow tracking server
    mlflow_tracking_uri = str(os.getenv("MLFLOW_TRACKING_URI"))

    # threshold value to calculate true anomalies using a linear function
    true_anomaly_threshold = float(os.getenv("FLT_TRUE_ANOMALY_THRESHOLD", "0.001"))

    metric_start_time = parse_datetime(os.getenv("FLT_DATA_START_TIME", "2019-08-05 18:00:00"))

    metric_end_time = parse_datetime(os.getenv("FLT_DATA_END_TIME", "2019-08-08 18:00:00"))

    # this will create a rolling data window on which the model will be trained
    # example: if set to 15d will train the model on past 15 days of data,
    # every time new data is added, it will truncate the data that is out of this range.
    rolling_training_window_size = parse_timedelta(
        "now", os.getenv("FLT_ROLLING_TRAINING_WINDOW_SIZE", "2d")
    )

    metric_train_data_end_time = metric_start_time + rolling_training_window_size

    # How often should the anomaly detector retrain the model (in minutes)
    retraining_interval_minutes = int(os.getenv("FLT_RETRAINING_INTERVAL_MINUTES", "120"))
    metric_chunk_size = parse_timedelta("now", str(retraining_interval_minutes) + "m")

    _LOGGER.info("Metric train data start time: %s", metric_start_time)
    _LOGGER.info("Metric train data end time/test data start time: %s", metric_train_data_end_time)
    _LOGGER.info("Metric test end time: %s", metric_end_time)
    _LOGGER.info("Metric data rolling training window size: %s", rolling_training_window_size)
    _LOGGER.info("Model retraining interval: %s minutes", retraining_interval_minutes)
    _LOGGER.info("True anomaly threshold: %s", true_anomaly_threshold)
    _LOGGER.info("MLflow server url: %s", mlflow_tracking_uri)
