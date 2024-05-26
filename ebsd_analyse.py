# -*- coding: utf-8 -*-

from datetime import datetime
from utilities import get_file_paths, get_directory_path, format_time_interval
from scan import Scan


def analyse(path: str = "data") -> None:
    filepaths = get_file_paths(directory_path=get_directory_path(path), recursive=True, extension="csv")

    # if input('Reduce map resolution? (Y/N): ').lower() == 'y':
    #     reduce_resolution = True
    #     reduction_factor = int(input("  Enter resolution reduction factor (power of 2): "))
    # else:
    #     reduce_resolution = False

    if input("Perform defect density analysis? (Y/N): ").lower() == "y":
        show_defect_density = True
        pixel_size_micrometres = float(input("  Enter pixel size (μm): "))

        # if reduce_resolution:
        #     pixel_size_micrometres *= 2 ** reduction_factor

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
        materials_path = f"{get_directory_path("example_data")}/materials.csv"
        scan = Scan.from_pathfinder_file(filepath, materials_path)

        fileref = scan.file_reference
        print(f"Making analysis for p{fileref}.")
        start_time = datetime.now()

        # if cNum >= 0:
        #     fileref += '-' + str(cNum)
        #     data = ebsd.crunches(data, cNum)

        output_path = f"{get_directory_path("analyses")}/q{fileref}.csv"

        if show_defect_density:
            scan.pixel_size_micrometres = pixel_size_micrometres

        if show_channelling_fraction:
            scan.beam_atomic_number = beam_atomic_number
            scan.beam_energy = beam_energy
            scan.beam_tilt_degrees = beam_tilt_degrees

        if show_orientation_cluster:
            scan.core_point_neighbour_threshold = core_point_neighbour_threshold
            scan.neighbourhood_radius_degrees = neighbourhood_radius_degrees

        scan.to_pathfinder_file(
            path=output_path,
            show_phases=True,
            show_map_size=True,
            show_map_scale=show_defect_density,
            show_channelling_params=show_channelling_fraction,
            show_clustering_params=show_orientation_cluster,
            show_scan_coordinates=True,
            show_phase=True,
            show_euler_angles=True,
            show_index_quality=True,
            show_pattern_quality=True,
            show_inverse_x_pole_figure_coordinates=False,
            show_inverse_y_pole_figure_coordinates=False,
            show_inverse_z_pole_figure_coordinates=True,
            show_kernel_average_misorientation=True,
            show_geometrically_necessary_dislocation_density=show_defect_density,
            show_channelling_fraction=show_channelling_fraction,
            show_orientation_cluster=show_orientation_cluster,
        )

        time_taken = (datetime.now() - start_time).total_seconds()
        print(f"Analysis completed in: {format_time_interval(int(round(time_taken)))}")

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


# ebsd.analyse(path)
# path: Working directory for code

if __name__ == "__main__":
    analyse()
