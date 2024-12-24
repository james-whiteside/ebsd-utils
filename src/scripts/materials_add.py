# -*- coding: utf-8 -*-

from src.data_structures.phase import Phase
from src.utilities.config import Config


CONFIG_PATH = "config.ini"


def add(global_id: int, config: Config) -> Phase:
	print(f"Adding phase {global_id}.")
	phase = Phase.build(global_id, config.project.database_path)
	phase.save(config.project.phase_dir)
	print("Phase added.")
	return phase
