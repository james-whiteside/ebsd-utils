# -*- coding: utf-8 -*-

from datetime import datetime
from src.utilities.config import Config
from src.utilities.utils import format_time_interval
from src.analysis import Analysis


def analyse(data_path: str, config: Config) -> str:
    analysis = Analysis.from_csv(data_path, config)
    print(f"Making analysis for {analysis.params.data_ref}.")
    start_time = datetime.now()

    if config.analysis.reduce_resolution:
        analysis = analysis.reduce_resolution(config.resolution.reduction_factor)

    analysis.to_csv(config.project.analysis_dir)
    analysis.to_maps(config.project.map_dir)
    time_taken = (datetime.now() - start_time).total_seconds()
    print(f"Analysis completed in: {format_time_interval(time_taken)}")
    return analysis.params.analysis_ref
