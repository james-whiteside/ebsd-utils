# -*- coding: utf-8 -*-

from datetime import datetime
from src.utilities.config import Config
from src.utilities.utilities import get_file_paths, get_directory_path, format_time_interval
from src.scan import Scan


def analyse() -> None:
    config = Config()
    data_paths = get_file_paths(directory_path=get_directory_path(config.data_dir), recursive=True, extension="csv")

    for data_path in data_paths:
        data_ref = data_path.split("/")[-1].split(".")[0].lstrip("p")
        scan = Scan.from_csv(data_path, config, data_ref)

        print(f"Making analysis for p{scan.params.data_ref}.")
        start_time = datetime.now()

        if config.reduce_resolution:
            scan = scan.reduce_resolution(config.reduction_factor)

        output_path = f"{get_directory_path(config.analysis_dir)}/q{scan.params.analysis_ref}.csv"
        map_dir = f"{get_directory_path(config.map_dir)}/{scan.params.analysis_ref}"
        scan.to_csv(output_path)
        scan.to_maps(map_dir)
        time_taken = (datetime.now() - start_time).total_seconds()
        print(f"Analysis completed in: {format_time_interval(time_taken)}")

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


if __name__ == "__main__":
    analyse()
