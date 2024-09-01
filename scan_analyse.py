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
    reduce_resolution = input('Reduce map resolution? (Y/N): ').lower() == 'y'
    show_defect_density = input("Perform defect density analysis? (Y/N): ").lower() == "y"
    show_channelling_fraction = input("Perform channelling fraction analysis? (Y/N): ").lower() == "y"
    show_orientation_cluster = input("Perform orientation cluster analysis? (Y/N): ").lower() == "y"

    for filepath in filepaths:
        scan = Scan.from_pathfinder_file(filepath)

        print(f"Making analysis for p{scan.parameters.data_reference}.")
        start_time = datetime.now()
        map_types = [MapType.P, MapType.EA, MapType.PQ, MapType.IQ, MapType.OX, MapType.OY, MapType.OZ, MapType.KAM]

        if show_defect_density:
            scan.scale_parameters.set(config.pixel_size)
            map_types += [MapType.GND]

        if show_channelling_fraction:
            scan.channelling_parameters.set(config.beam_atomic_number, config.beam_energy, config.beam_tilt)
            map_types += [MapType.OB, MapType.CF]

        if show_orientation_cluster:
            scan.clustering_parameters.set(config.neighbour_threshold, config.neighbourhood_radius)
            map_types += [MapType.OC]

        if reduce_resolution:
            scan = scan.reduce_resolution(config.reduction_factor)

        output_path = f"{get_directory_path(config.analysis_dir)}/q{scan.parameters.analysis_reference}.csv"
        os.makedirs(f"{get_directory_path(config.map_dir)}/{scan.parameters.analysis_reference}", exist_ok=True)

        scan.to_pathfinder_file(
            path=output_path,
            show_phases=True,
            show_map_size=True,
            show_map_scale=show_defect_density,
            show_channelling_params=show_channelling_fraction,
            show_clustering_params=show_orientation_cluster,
            show_cluster_aggregates=show_orientation_cluster,
            show_row_coordinates=True,
            show_phase=True,
            show_euler_angles=True,
            show_index_quality=True,
            show_pattern_quality=True,
            show_inverse_principal_pole_figure_coordinates=False,
            show_inverse_beam_pole_figure_coordinates=show_channelling_fraction,
            show_kernel_average_misorientation=True,
            show_geometrically_necessary_dislocation_density=show_defect_density,
            show_channelling_fraction=show_channelling_fraction,
            show_orientation_cluster=show_orientation_cluster,
        )

        for map_type in map_types:
            map_path = f"{get_directory_path(config.map_dir)}/{scan.parameters.analysis_reference}/{map_type.name}.png"
            # print(map_type.name)
            scan.map.get(map_type).image.save(map_path)

        time_taken = (datetime.now() - start_time).total_seconds()
        print(f"Analysis completed in: {format_time_interval(time_taken)}")

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


if __name__ == "__main__":
    analyse()
