# -*- coding: utf-8 -*-

from src.scripts.analyse import analyse
from src.utilities.config import Config
from src.utilities.utilities import get_file_paths


def main() -> None:
    config = Config()
    data_paths = get_file_paths(directory_path=config.project.data_dir, recursive=True, extension="csv")

    for data_path in data_paths:
        analyse(data_path, config)

    print()
    print("All analyses complete.")
    input("Press ENTER to close: ")


if __name__ == "__main__":
    main()
