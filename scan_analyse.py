# -*- coding: utf-8 -*-

import os
from datetime import datetime
from src.data_structures.map import MapType
from src.utilities.config import Config
from src.utilities.utilities import get_file_paths, get_directory_path, format_time_interval
from src.scan import Scan


def analyse() -> None:
    config = Config()
    filepaths = get_file_paths(directory_path=get_directory_path(config.data_dir), recursive=True, extension="csv")

    for filepath in filepaths:
        scan = Scan.from_pathfinder_file(filepath, config)

        print(f"Making analysis for p{scan.params.data_ref}.")
        start_time = datetime.now()
        map_types = [MapType.P, MapType.EA, MapType.PQ, MapType.IQ, MapType.OX, MapType.OY, MapType.OZ, MapType.KAM]

        if config.compute_dislocation:
            map_types += [MapType.GND]

        if config.compute_channelling:
            map_types += [MapType.OB, MapType.CF]

        if config.compute_clustering:
            map_types += [MapType.OC]

        if config.reduce_resolution:
            scan = scan.reduce_resolution(config.reduction_factor)

        output_path = f"{get_directory_path(config.analysis_dir)}/q{scan.params.analysis_ref}.csv"
        os.makedirs(f"{get_directory_path(config.map_dir)}/{scan.params.analysis_ref}", exist_ok=True)

        scan.to_pathfinder_file(
            path=output_path,
            show_phases=True,
            show_map_size=True,
            show_map_scale=config.compute_dislocation,
            show_channelling_params=config.compute_channelling,
            show_clustering_params=config.compute_clustering,
            show_cluster_aggregates=config.compute_clustering,
            show_row_coordinates=True,
            show_phase=True,
            show_euler_angles=True,
            show_index_quality=True,
            show_pattern_quality=True,
            show_principal_ipf_coordinates=False,
            show_beam_ipf_coordinates=config.compute_channelling,
            show_average_misorientation=True,
            show_gnd_density=config.compute_dislocation,
            show_channelling_fraction=config.compute_channelling,
            show_orientation_cluster=config.compute_clustering,
        )

        for map_type in map_types:
            map_path = f"{get_directory_path(config.map_dir)}/{scan.params.analysis_ref}/{map_type.name}.png"
            # print(map_type.name)
            scan.map.get(map_type).image.save(map_path)

        time_taken = (datetime.now() - start_time).total_seconds()
        print(f"Analysis completed in: {format_time_interval(time_taken)}")

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


if __name__ == "__main__":
    analyse()
