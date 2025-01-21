# -*- coding: utf-8 -*-

from datetime import datetime
from tests.test_analysis import test_analysis
from src.utilities.config import Config
from src.utilities.utils import format_time_interval


def test(data_ref: str, config: Config) -> None:
    data_path = f"{config.test.data_dir}/{data_ref}.csv"
    analysis_path = f"{config.test.analysis_dir}/{data_ref}.csv"
    map_dir = f"{config.test.map_dir}/{data_ref}"
    test_config = Config(f"{config.test.config_dir}/{data_ref}.ini")
    print(f"Running analysis test for {data_ref}.")
    start_time = datetime.now()
    test_analysis(data_path, analysis_path, map_dir, test_config)
    time_taken = (datetime.now() - start_time).total_seconds()
    print(f"Test completed in: {format_time_interval(time_taken)}")
