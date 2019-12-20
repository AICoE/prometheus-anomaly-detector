import os
import pytest
from prometheus_api_client.utils import parse_datetime


@pytest.fixture
def metric_start_time_fixture():
    return parse_datetime(
        os.getenv("FLT_DATA_START_TIME", "2019-08-05 18:00:00")
    )


@pytest.fixture
def metric_end_time_fixture():
    return parse_datetime(
        os.getenv("FLT_DATA_END_TIME", "2019-08-08 18:00:00")
    )
