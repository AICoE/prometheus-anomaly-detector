import os


class Configuration:
    """docstring for Configuration."""

    prometheus_url = os.getenv("FLT_PROM_URL")
    prom_connect_headers = None
    if os.getenv("FLT_PROM_ACCESS_TOKEN"):
        prom_connect_headers = {"Authorization": "bearer " + os.getenv("FLT_PROM_ACCESS_TOKEN")}
    metrics_list = str(
        os.getenv(
            "FLT_METRICS_LIST",
            "up{app='openshift-web-console', instance='172.44.0.18:8443'}; up{app='openshift-web-console', instance='172.44.4.18:8443'}; es_process_cpu_percent{instance='172.44.17.134:30290'",
        )
    ).split(";")
    retraining_interval_minutes = int(os.getenv("FLT_RETRAINING_INTERVAL_MINUTES", "15"))
    rolling_data_window_size = str(os.getenv("FLT_ROLLING_DATA_WINDOW_SIZE", "15d"))
