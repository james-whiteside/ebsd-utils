# -*- coding: utf-8 -*-

from datetime import datetime
from src.utilities.config import Config
from src.utilities.filestore import load_from_data, dump_analysis, dump_maps
from src.utilities.utils import format_time_interval


def analyse(data_path: str, config: Config) -> str:
    analysis = load_from_data(data_path, config)
    print(f"Making analysis for {analysis.params.data_ref}.")
    start_time = datetime.now()

    if config.analysis.reduce_resolution:
        analysis = analysis.reduce_resolution(config.resolution.reduction_factor)

    dump_analysis(analysis, config.project.analysis_dir)
    dump_maps(analysis, config.project.map_dir)
    time_taken = (datetime.now() - start_time).total_seconds()
    print(f"Analysis completed in: {format_time_interval(time_taken)}")
    return analysis.params.analysis_ref
