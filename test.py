import os
from datetime import datetime
from utilities import format_time_interval
from scan import Scan

data_path = f"{os.getcwd()}/example_data/p2205031525.csv".replace("\\", "/")
materials_path = f"{os.getcwd()}/example_data/materials.csv".replace("\\", "/")
output_path = f"{os.getcwd()}/test_data/q2205031525.csv".replace("\\", "/")

scan = Scan.from_pathfinder_file(data_path, materials_path)
print("Making analysis for p" + scan.file_reference + ".")
start_time = datetime.now()

# if input("Reduce map resolution? (Y/N): ").lower() == "y":
#     cNum = int(input("  Enter resolution reduction factor (power of 2): "))

if input("Perform defect density analysis? (Y/N): ").lower() == 'y':
    scan.pixel_size_micrometres = 0.808  # float(input("  Input pixel width (μm) for p" + scan.file_reference + ": "))

if input('Perform channelling fraction analysis? (Y/N): ').lower() == "y":
    scan.beam_atomic_number = 31  # int(input("  Enter beam species atomic number: "))
    scan.beam_energy = 30000.0  # float(input("  Enter beam energy (eV): "))
    scan.beam_tilt_degrees = 0.0  # float(input("  Enter beam tilt (deg): "))

if input("Perform orientation cluster analysis? (Y/N): ").lower() == "y":
    scan.core_point_neighbour_threshold = 20  # int(input("  Enter core point neighbour threshold: "))
    scan.neighbourhood_radius_degrees = 1.0  # float(input("  Enter point neighbourhood radius (deg): "))

scan.to_pathfinder_file(
    path=output_path,
    show_phases=True,
    show_map_size=True,
    show_map_scale=True,
    show_channelling_params=True,
    show_clustering_params=False,
    show_scan_coordinates=True,
    show_phase=True,
    show_euler_angles=True,
    show_index_quality=True,
    show_pattern_quality=True,
    show_inverse_x_pole_figure_coordinates=False,
    show_inverse_y_pole_figure_coordinates=False,
    show_inverse_z_pole_figure_coordinates=True,
    show_kernel_average_misorientation=True,
    show_geometrically_necessary_dislocation_density=True,
    show_channelling_fraction=False,
    show_orientation_cluster=False,
)

print("Analysis completed in " + format_time_interval(int(round((datetime.now() - start_time).total_seconds()))) + ".")
