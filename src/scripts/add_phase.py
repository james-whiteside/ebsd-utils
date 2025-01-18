# -*- coding: utf-8 -*-

from src.utilities.config import Config
from src.utilities.filestore import add_phase as add_phase_


def add_phase(global_id: int, config: Config) -> int:
    print(f"Adding phase {global_id}.")
    add_phase_(global_id, config)
    print("Phase added.")
    return global_id
