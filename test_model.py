"""docstring for installed packages."""
import logging
import numpy as np
from prometheus_api_client import PrometheusConnect, MetricsList, Metric
from test_configuration import Configuration
import mlflow

# import model_fourier as model
import model


# Set up logging
_LOGGER = logging.getLogger(__name__)

MLFLOW_CLIENT = mlflow.tracking.MlflowClient(
    tracking_uri=Configuration.mlflow_tracking_uri
)

METRICS_LIST = Configuration.metrics_list

# Prometheus Connection details
pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=True,
)


def calculate_rmse(predicted, true):
    """Calculate the Root Mean Squared Error (RMSE) between the predicted and true values."""
    return (((predicted - true) ** 2).mean()) ** 0.5


def calculate_accuracy(predicted, true):
    """Calculate the accuracy of the predictions."""
    return (1 - sum(abs(predicted - true)) / len(true)) * 100


def label_true_anomalies(true_value_df, threshold_value):
    """Label the true anomalies."""
    # label true anomalies based on a simple linear threshold,
    # can be replaced with a more complex calculation based on metric data
    return np.where(true_value_df["y"] > threshold_value, 1, 0)


def label_predicted_anomalies(true_value_df, predicted_value_df):
    """Label the predicted anomalies."""
    return np.where(
        (
            ((true_value_df["y"].values >= predicted_value_df["yhat_upper"].values))
            | (true_value_df["y"].values <= predicted_value_df["yhat_lower"].values)
        ),
        1,
        0,
    )


def compute_true_positive_rate(forecasted_anomalies, labeled_anomalies):
    """Calculate the true positive rate."""
    num_true_positive = sum(
        (forecasted_anomalies.values == 1) & (labeled_anomalies.values == 1)
    )
    true_postive_rate = num_true_positive / sum(labeled_anomalies.values)

    return true_postive_rate


# Run for every metric defined in the METRICS_LIST
for metric in METRICS_LIST:
    # Download the the train data from Prometheus
    train_data = MetricsList(
        pc.get_metric_range_data(
            metric_name=metric,
            start_time=Configuration.metric_start_time,
            end_time=Configuration.metric_train_data_end_time,
            chunk_size=Configuration.metric_chunk_size,
        )
    )

    # If the training data list downloaded is empty
    if not train_data:
        _LOGGER.error("No Metric data received, please check the data window size")
        raise ValueError

    # If more than one time-series match the given metric, raise an error
    if len(train_data) > 1:
        _LOGGER.error("Multiple timeseries matching %s were found")
        _LOGGER.error("The timeseries matched were: ")
        for timeseries in train_data:
            print(timeseries.metric_name, timeseries.label_config)
        _LOGGER.error("One metric should be specific to a single time-series")
        raise ValueError

    # Download test data
    test_data_list = pc.get_metric_range_data(
        metric_name=train_data[0].metric_name,
        label_config=train_data[0].label_config,
        start_time=Configuration.metric_train_data_end_time,
        end_time=Configuration.metric_end_time,
        chunk_size=Configuration.metric_chunk_size,
    )
    _LOGGER.info("Downloaded metric data")

    model_mp = model.MetricPredictor(
        train_data[0],
        rolling_data_window_size=Configuration.rolling_training_window_size,
    )

    mlflow.set_experiment(train_data[0].metric_name)
    mlflow.start_run()
    mlflow_run_id = mlflow.active_run().info.run_id

    # keep track of the model name as a mlflow run tag
    mlflow.set_tag("model", model_mp.model_name)

    # keep track of labels as tags in the mlflow experiment
    for label in train_data[0].label_config:
        mlflow.set_tag(label, train_data[0].label_config[label])

    # store the metric with labels as a tag so it can be copied into grafana to view the real metric
    mlflow.set_tag("metric", metric)

    # log parameters before run
    mlflow.log_param(
        "retraining_interval_minutes", str(Configuration.retraining_interval_minutes)
    )
    mlflow.log_param(
        "rolling_training_window_size", str(Configuration.rolling_training_window_size)
    )
    mlflow.log_param(
        "true_anomaly_threshold", str(Configuration.true_anomaly_threshold)
    )

    # initial run with just the train data
    model_mp.train(prediction_duration=Configuration.retraining_interval_minutes)

    # store the predicted dataframe and the true dataframe
    predicted_df = model_mp.predicted_df
    true_df = Metric(test_data_list[0]).metric_values.set_index("ds")

    # Label True Anomalies
    true_df["anomaly"] = label_true_anomalies(
        true_df, Configuration.true_anomaly_threshold
    )

    # track true_positives & ground truth anomalies
    num_true_positives = 0
    num_ground_truth_anomalies = 0

    for item in range(len(test_data_list) - 1):
        # the true values for this training period
        true_values = Metric(test_data_list[item + 1])
        true_values.metric_values = true_values.metric_values.set_index("ds")
        true_df += true_values.metric_values

        # for each item in the test_data list, update the model (append new data and train it)
        model_mp.train(test_data_list[item], len(true_values.metric_values))

        # get the timestamp for the median value in the df
        metric_timestamp = true_values.metric_values.index.values[
            int(len(true_values.metric_values) / 2)
        ]
        metric_timestamp = int(metric_timestamp.astype("uint64") / 1e6)

        # calculate predicted anomaly
        model_mp.predicted_df["anomaly"] = label_predicted_anomalies(
            true_values.metric_values, model_mp.predicted_df
        )

        # store the prediction df for every interval
        predicted_df = predicted_df + model_mp.predicted_df

        # Label True Anomalies
        true_values.metric_values["anomaly"] = label_true_anomalies(
            true_values.metric_values, Configuration.true_anomaly_threshold
        )

        true_df += true_values.metric_values

        # Total number of predicted and ground truth anomalies
        sum_predicted_anomalies = sum(model_mp.predicted_df["anomaly"])
        sum_ground_truth_anomalies = sum(true_values.metric_values["anomaly"])

        num_true_positives += sum(
            (model_mp.predicted_df["anomaly"] == 1)
            & (true_values.metric_values["anomaly"] == 1)
        )
        num_ground_truth_anomalies += sum_ground_truth_anomalies

        # Calculate accuracy
        accuracy = calculate_accuracy(
            model_mp.predicted_df["anomaly"].values,
            true_values.metric_values["anomaly"].values,
        )
        # Calculate RMSE
        rmse = calculate_rmse(model_mp.predicted_df.yhat, true_values.metric_values.y)

        # Calculate True positive rate for anomalies
        true_positive_rate = compute_true_positive_rate(
            model_mp.predicted_df["anomaly"], true_values.metric_values["anomaly"]
        )

        # log some accuracy metrics here
        MLFLOW_CLIENT.log_metric(mlflow_run_id, "RMSE", rmse, metric_timestamp, item)
        MLFLOW_CLIENT.log_metric(
            mlflow_run_id, "Accuracy", accuracy, metric_timestamp, item
        )

        MLFLOW_CLIENT.log_metric(
            mlflow_run_id,
            "Ground truth anomalies",
            sum_ground_truth_anomalies,
            metric_timestamp,
            item,
        )
        MLFLOW_CLIENT.log_metric(
            mlflow_run_id,
            "Forecasted anomalies",
            sum_predicted_anomalies,
            metric_timestamp,
            item,
        )
        MLFLOW_CLIENT.log_metric(
            mlflow_run_id,
            "Number of test data points",
            len(true_values.metric_values),
            metric_timestamp,
            item,
        )

        # Only log non Nan values for the true_anomaly_postive_rate
        if true_positive_rate:
            MLFLOW_CLIENT.log_metric(
                mlflow_run_id,
                "true_anomaly_postive_rate",
                true_positive_rate,
                metric_timestamp,
                item,
            )

    # collect and log metrics for the entire test run
    total_true_positive_rate = np.nan
    if num_ground_truth_anomalies:
        # check if num_ground_truth_anomalies is not 0 to avoid division by zero errors
        total_true_positive_rate = num_true_positives / num_ground_truth_anomalies

    MLFLOW_CLIENT.log_metric(
        mlflow_run_id,
        "Total true positive rate",
        total_true_positive_rate,
        metric_timestamp,
    )
    MLFLOW_CLIENT.log_metric(
        mlflow_run_id, "Total true positive count", num_true_positives, metric_timestamp
    )
    MLFLOW_CLIENT.log_metric(
        mlflow_run_id,
        "Total ground truth count",
        num_ground_truth_anomalies,
        metric_timestamp,
    )

    mlflow.end_run()
