import logging
import os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

base_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path="{dir}/.env".format(dir=base_path))

logger = logging.getLogger('AB Pipeline')
logger.setLevel(os.getenv("LOG_LEVEL"))
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

PATH_METRIC_CONFIGS = "params/metrics/"
MONTE_CARLO_CONFIGS = "params/montecarlo/"
DEFAULT_ESTIMATOR = "t_test_linearization"
DEFAULT_METRIC_TYPE = "ratio"
DEFAULT_UNIT_LEVEL = "client_id"
DEFAULT_VALUE = "Unknown"
VARIANT_COL = "experiment_variant"
USER_ID_COL = "client_id"
DEFAULT_LIFTS_VALUE = {'start': 0,
                       'end': 10,
                       'by': 1}
DEFAULT_START_VALUE = 0
DEFAULT_END_VALUE = 10
DEFAULT_STEP_VALUE = 1

