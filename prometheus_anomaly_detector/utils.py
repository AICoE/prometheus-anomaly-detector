"""Utility functions"""
import numpy as np


def NaN(): return np.nan


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
            ((true_value_df["y"] >= predicted_value_df["yhat_upper"]))
            | (true_value_df["y"] <= predicted_value_df["yhat_lower"])
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
