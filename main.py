# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from os import listdir
from tomllib import load as load_toml
from src.scripts.add_phase import add_phase
from src.scripts.analyse import analyse
from src.scripts.test import test
from src.utilities.config import Config
from src.utilities.utils import get_file_paths, parse_ids


CONFIG_PATH = "config.ini"
TOML_PATH = "pyproject.toml"


def main(args: Namespace, config: Config) -> None:
    match args.mode:
        case "analyse":
            if "path" in vars(args):
                data_paths = [args.path]
            else:
                data_paths = get_file_paths(directory_path=config.project.data_dir, recursive=True, extension="csv", prompt="Data files found:")

            for data_path in data_paths:
                analyse(data_path, config)

            print()
            print("All analyses complete.")

        case "add_phase":
            global_ids = parse_ids(input("Enter Pathfinder database IDs of phases to add separated by commas/hyphens: "))

            for global_id in global_ids:
                add_phase(global_id, config)

            print()
            print("All analyses complete.")

        case "test":
            if "path" in vars(args):
                data_refs = [args.path.split("/")[-1].split(".")[0]]
            else:
                data_refs = [path.split("/")[-1].split(".")[0] for path in listdir(config.test.data_dir)]

            for data_ref in data_refs:
                test(data_ref, config)

            print()
            print("All tests complete.")


if __name__ == "__main__":
    with open(TOML_PATH, "rb") as toml:
        parser = ArgumentParser("ebsd-utils")
        parser.add_argument("mode", metavar="program_mode", type=str, choices=["analyse", "add_phase", "test"], help="\"analyse\", \"add_phase\", or \"test\"")
        parser.add_argument("-f", "--file", metavar="path", type=str, help="path of EBSD data file to analyse or test")
        parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {load_toml(toml)["project"]["version"]}")

    args = parser.parse_args()
    config = Config(CONFIG_PATH)
    main(args, config)
