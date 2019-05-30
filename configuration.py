import os


class Configuration:
    """docstring for Configuration."""

    # url for the prometheus host
    prometheus_url = os.getenv("FLT_PROM_URL")

    # any headers that need to be passed while connecting to the prometheus host
    prom_connect_headers = None
    # example oath token passed as a header
    if os.getenv("FLT_PROM_ACCESS_TOKEN"):
        prom_connect_headers = {"Authorization": "bearer " + os.getenv("FLT_PROM_ACCESS_TOKEN")}

    # list of metrics that need to be scraped and predicted multiple metrics can be separated with a ";"
    # if a metric configuration matches more than one timeseries, it will scrape all the timeseries that match the config.
    metrics_list = str(
        os.getenv(
            "FLT_METRICS_LIST",
            "up{app='openshift-web-console', instance='172.44.0.18:8443'}; up{app='openshift-web-console', instance='172.44.4.18:8443'}; es_process_cpu_percent{instance='172.44.17.134:30290'}",
        )
    ).split(";")
    # How often should the anomaly detector retrain the model
    retraining_interval_minutes = int(os.getenv("FLT_RETRAINING_INTERVAL_MINUTES", "15"))

    # this will create a rolling data window on which the model will be trained
    # example: if set to 15d will train the model on past 15 days of data,
    # every time new data is data, it will truncate the data out of this range.
    rolling_data_window_size = str(os.getenv("FLT_ROLLING_DATA_WINDOW_SIZE", "15d"))
