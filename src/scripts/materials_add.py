# -*- coding: utf-8 -*-

from src import utilities
from src.data_structures.phase import Phase
from src.utilities.config import Config


CONFIG_PATH = "config.ini"


def add(database_path: str | None):
	print()
	ids = utilities.utils.parse_ids(input('Enter material IDs to add separated by commas/hyphens: '))

	for id in ids:
		phase = Phase.build(id, database_path)
		phase.cache(Config(CONFIG_PATH).project.phase_dir)

	print()
	print('All phases added.')
