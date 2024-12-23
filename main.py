# -*- coding: utf-8 -*-

from src.scripts.analyse import analyse
from src.utilities.config import Config
from src.utilities.utils import get_file_paths


CONFIG_PATH = "config.ini"


def main() -> None:
    config = Config(CONFIG_PATH)
    data_paths = get_file_paths(directory_path=config.project.data_dir, recursive=True, extension="csv")

    for data_path in data_paths:
        analyse(data_path, config)

    print()
    print("All analyses complete.")


if __name__ == "__main__":
    main()
