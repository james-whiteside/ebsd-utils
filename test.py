# -*- coding: utf-8 -*-

from os import listdir
from src.scripts.test_analysis import test_analysis
from src.utilities.config import Config


DATA_DIR = "test/data"
ANALYSIS_DIR = "test/analyses"
MAP_DIR = "test/maps"
CONFIG_DIR = "test/config"


def test() -> None:
    data_refs = [path.split("/")[-1].split(".")[0] for path in listdir(DATA_DIR)]

    for data_ref in data_refs:
        data_path = f"{DATA_DIR}/{data_ref}.csv"
        analysis_path = f"{ANALYSIS_DIR}/{data_ref}.csv"
        map_dir = f"{MAP_DIR}/{data_ref}"
        config = Config(f"{CONFIG_DIR}/{data_ref}.ini")
        test_analysis(data_path, analysis_path, map_dir, config)

    print()
    print("All tests complete.")


if __name__ == "__main__":
    test()
