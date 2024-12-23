# -*- coding: utf-8 -*-

from os import listdir
from PIL.Image import open as open_image
from src.utilities.config import Config
from src.scripts.analyse import analyse


def test(data_path: str, analysis_path: str, map_dir: str, config: Config) -> str:
    config.project.analysis_dir = f"{config.project.test_cache_dir}/{config.project.analysis_dir}"
    config.project.map_dir = f"{config.project.test_cache_dir}/{config.project.map_dir}"
    analysis_ref = analyse(data_path, config)

    with (
        open(analysis_path) as control_analysis,
        open(f"{config.project.analysis_dir}/{analysis_ref}.csv") as test_analysis,
    ):
        control_line = None
        test_line = None

        while control_line != "" or test_line != "":
            control_line = control_analysis.readline()
            test_line = test_analysis.readline()

            if control_line != test_line:
                message = (
                    f"Lines in analysis files differ:\n"
                    f"Control: {control_line}\n"
                    f"Test: {test_line}"
                )

                raise AssertionError(message)

    map_refs = [path.split("/")[-1].split(".")[0] for path in listdir(map_dir)]

    for map_ref in map_refs:
        with (
            open_image(f"{map_dir}/{map_ref}.png") as control_map,
            open_image(f"{config.project.map_dir}/{analysis_ref}/{map_ref}.png") as test_map,
        ):
            if control_map.size != test_map.size:
                message = (
                    f"Map dimensions differ for {map_ref}.png:\n"
                    f"Control: {control_map.size}\n"
                    f"Test: {test_map.size}"
                )

                raise AssertionError(message)

            for y in range(control_map.height):
                for x in range(control_map.width):
                    if control_map.getpixel((x, y)) != test_map.getpixel((x, y)):
                        message = (
                            f"Map pixel {(x, y)} differs for {map_ref}.png\n"
                            f"Control: {control_map.getpixel((x, y))}\n"
                            f"Test: {test_map.getpixel((x, y))}\n"
                        )

                        raise AssertionError(message)

    return analysis_ref
