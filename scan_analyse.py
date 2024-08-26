# -*- coding: utf-8 -*-

import os
from datetime import datetime
from src.data_structures.map import MapType
from src.utilities.utilities import get_file_paths, get_directory_path, format_time_interval
from src.scan import Scan


def analyse(path: str = "data") -> None:
    filepaths = get_file_paths(directory_path=get_directory_path(path), recursive=True, extension="csv")

    if input('Reduce map resolution? (Y/N): ').lower() == 'y':
        reduce_resolution = True
        reduction_factor = int(input("  Enter resolution reduction factor (power of 2): "))
    else:
        reduce_resolution = False

    if input("Perform defect density analysis? (Y/N): ").lower() == "y":
        show_defect_density = True
        pixel_size_micrometres = float(input("  Enter pixel size (μm): "))
    else:
        show_defect_density = False

    if input("Perform channelling fraction analysis? (Y/N): ").lower() == "y":
        show_channelling_fraction = True
        beam_atomic_number = int(input("  Enter beam species atomic number: "))
        beam_energy = float(input("  Enter beam energy (eV): "))
        beam_tilt_degrees = float(input("  Enter beam tilt (deg): "))
    else:
        show_channelling_fraction = False

    if input("Perform orientation cluster analysis? (Y/N): ").lower() == "y":
        show_orientation_cluster = True
        core_point_neighbour_threshold = int(input("  Enter core point neighbour threshold: "))
        neighbourhood_radius_degrees = float(input("  Enter point neighbourhood radius (deg): "))
    else:
        show_orientation_cluster = False

    for filepath in filepaths:
        scan = Scan.from_pathfinder_file(filepath)

        print(f"Making analysis for p{scan.parameters.data_reference}.")
        start_time = datetime.now()
        map_types = [MapType.P, MapType.EA, MapType.PQ, MapType.IQ, MapType.OX, MapType.OY, MapType.OZ, MapType.KAM]

        if show_defect_density:
            scan.scale_parameters.set(pixel_size_micrometres)
            map_types += [MapType.GND]

        if show_channelling_fraction:
            scan.channelling_parameters.set(beam_atomic_number, beam_energy, beam_tilt_degrees)
            map_types += [MapType.OB, MapType.CF]

        if show_orientation_cluster:
            scan.clustering_parameters.set(core_point_neighbour_threshold, neighbourhood_radius_degrees)
            map_types += [MapType.OC]

        if reduce_resolution:
            scan = scan.reduce_resolution(reduction_factor)

        output_path = f"{get_directory_path("analyses")}/q{scan.parameters.analysis_reference}.csv"
        os.makedirs(f"{get_directory_path("maps")}/{scan.parameters.analysis_reference}", exist_ok=True)

        for map_type in map_types:
            map_path = f"{get_directory_path("maps")}/{scan.parameters.analysis_reference}/{map_type.name}.png"
            scan.map.get(map_type).image.save(map_path)

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

        time_taken = (datetime.now() - start_time).total_seconds()
        print(f"Analysis completed in: {format_time_interval(time_taken)}")

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


if __name__ == "__main__":
    analyse()
