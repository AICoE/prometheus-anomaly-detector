

def test_imports():
    """Trying not to break stuff."""

    # app.py
    import time
    import logging
    from datetime import datetime
    from multiprocessing import Process, Queue
    from queue import Empty as EmptyQueueException
    import tornado.ioloop
    import tornado.web
    from prometheus_client import Gauge, generate_latest, REGISTRY
    from prometheus_api_client import PrometheusConnect, Metric
    from configuration import Configuration
    import prometheus_anomaly_detector.model as model
    import schedule

    # configuration.py
    import os
    import logging
    from prometheus_api_client.utils import parse_timedelta

    # model.py
    import logging
    from fbprophet import Prophet
    from prometheus_api_client import Metric

    # model_fourier.py
    import logging
    import pandas as pd
    import numpy as np
    from prometheus_api_client import Metric
    from numpy import fft
