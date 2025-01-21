# -*- coding: utf-8 -*-

from datetime import datetime
from src.data_structures.phase import PhaseMissingError
from src.scripts.add_phase import add_phase
from src.utilities.config import Config
from src.utilities.filestore import load_from_data, dump_analysis, dump_maps
from src.utilities.utils import format_time_interval


def analyse(data_path: str, config: Config) -> str:
    data_loaded = False

    while not data_loaded:
        try:
            analysis = load_from_data(data_path, config)
            data_loaded = True
        except PhaseMissingError as error:
            print(f"Warning: No data found for phase with ID {error.global_id}.")

            if input("Enter phase information now? (Y/N): ").lower() == "y":
                add_phase(error.global_id, config)
            else:
                raise error

    print(f"Making analysis for {analysis.params.data_ref}.")
    start_time = datetime.now()

    if config.analysis.reduce_resolution:
        analysis = analysis.reduce_resolution(config.resolution.reduction_factor)

    dump_analysis(analysis, config.project.analysis_dir)
    dump_maps(analysis, config.project.map_dir)
    time_taken = (datetime.now() - start_time).total_seconds()
    print(f"Analysis completed in: {format_time_interval(time_taken)}")
    return analysis.params.analysis_ref
