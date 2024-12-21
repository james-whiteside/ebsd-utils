# -*- coding: utf-8 -*-

from datetime import datetime
from src.utilities.config import Config
from src.utilities.utilities import get_file_paths, get_directory_path, format_time_interval
from src.scan import Scan


def analyse() -> None:
    config = Config()
    data_dir = get_directory_path(config.project.data_dir)
    data_paths = get_file_paths(directory_path=data_dir, recursive=True, extension="csv")

    for data_path in data_paths:
        scan = Scan.from_csv(data_path, config)

        print(f"Making analysis for {scan.params.data_ref}.")
        start_time = datetime.now()

        if config.analysis.reduce_resolution:
            scan = scan.reduce_resolution(config.resolution.reduction_factor)

        scan.to_csv(get_directory_path(config.project.analysis_dir))
        scan.to_maps(get_directory_path(config.project.map_dir))
        time_taken = (datetime.now() - start_time).total_seconds()
        print(f"Analysis completed in: {format_time_interval(time_taken)}")

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


if __name__ == "__main__":
    analyse()
