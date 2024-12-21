# -*- coding: utf-8 -*-

from datetime import datetime
from glob import glob
from sys import exit
from os.path import getsize
from math import floor, log10, degrees, radians
from copy import copy
from collections.abc import Callable


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def log_or_zero(value: float) -> float:
    return 0.0 if value == 0.0 else log10(value)


def float_degrees(angle: float) -> float:
    return degrees(angle)


def float_radians(angle: float) -> float:
    return radians(angle)


def tuple_degrees(angles: tuple[float, float, float]) -> tuple[float, float, float]:
    return float_degrees(angles[0]), float_degrees(angles[1]), float_degrees(angles[2])


def tuple_radians(angles: tuple[float, float, float]) -> tuple[float, float, float]:
    return float_radians(angles[0]), float_radians(angles[1]), float_radians(angles[2])


def highest_common_factor(numbers: list[int]) -> int:
    """
    Calculates the highest common factor of a list of integers.
    :param numbers: The list of integers.
    :return: The highest common factor.
    """

    numbers = copy(numbers)

    if len(numbers) == 2:
        x = numbers[0]
        y = numbers[1]

        while y:
            x, y = y, x % y

        return x
    else:
        z = numbers.pop()
        return highest_common_factor(list((z, highest_common_factor(numbers))))


def format_sig_figs(number: int | float, sig_figs: int) -> str:
    """
    Formats a number as a string, rounded to the significant figures specified.
    :param number: The number to be formatted.
    :param sig_figs: The number of significant figures.
    :return: The formatted string.
    """

    if float(number) == 0.0:
        return "0.0"
    else:
        return str(round(float(number), -int(floor(log10(abs(float(number)))) - sig_figs + 1)))


def format_sig_figs_or_int(number: int | float, sig_figs: int) -> str:
    """
    Formats a number as a string, rounded to the nearest integer or to the significant figures specified.
    The rounding method selected is the one that results in the least precision loss.
    :param number: The number to be formatted.
    :param sig_figs: The number of significant figures.
    :return: The formatted string.
    """

    if abs(number) >= 10 ** (sig_figs - 1):
        return str(int(round(number)))
    else:
        return str(format_sig_figs(number, sig_figs))


def format_file_size(size: int) -> str:
    """
    Formats a file size in bytes as a string, in units of B, kB, MB, or GB, rounded to three significant figures.
    The largest possible unit is used such that the numeric representation in those units is greater than one.
    :param size: The file size in bytes to be formatted.
    :return: The formatted string with units.
    """

    if size < 1024:
        return f"{size} B"

    size = float(size) / 1024.0

    if size < 1024.0:
        return f"{format_sig_figs(size, 3)} kB"

    size /= 1024.0

    if size < 1024.0:
        return f"{format_sig_figs(size, 3)} MB"

    size /= 1024.0
    return f"{format_sig_figs(size, 3)} GB"


def format_time_interval(interval: int | float) -> str:
    """
    Formats a time interval in seconds as a string in ``hh:mm:ss`` format, rounded to the nearest second.
    The number of hours can exceed two digits.
    :param interval: The time interval in seconds to be formatted.
    :return: The formatted string.
    """

    if type(interval) is float:
        interval = int(round(interval))

    seconds = interval % 60
    minutes = int(round((interval - seconds) / 60)) % 60
    hours = int(round((interval - 60 * minutes - seconds) / 3600))
    return str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2)


def parse_ids(id_string: str) -> list[int]:
    """
    Parses a specially formatted input string to produce an ordered list of unique integer IDs.
    The input string must take the following format: ``<START>[-<STOP>[-<STEP>]][,<START>[-<STOP>[-<STEP>]] ...]``
    Each comma-separated ``<START>[-<STOP>[-<STEP>]]`` tuple is evaluated using the built-in range function, but with ``<STOP>`` increased by one.
    For example, the input string ``"0-33-3,15-37-5,7,12-19"`` will produce the following list of integers:
    ``[0, 3, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 30, 33, 35]``
    :param id_string: The specially formatted input string.
    :return: The ordered list of unique integer IDs.
    """

    ids = set()
    domains = id_string.split(',')

    for domain in domains:
        domain_args = domain.split('-')

        if len(domain_args) == 1:
            ids.add(int(domain_args[0]))
        elif len(domain_args) == 2:
            for value in range(int(domain_args[0]), int(domain_args[1]) + 1):
                ids.add(value)
        else:
            for value in range(int(domain_args[0]), int(domain_args[1]) + 1, int(domain_args[2])):
                ids.add(value)

    return sorted(list(ids))


def _get_file_paths(directory_path: str, recursive: bool, extension: str | None, exclusions: list[str] | None, prompt: str, get_many: bool) -> list[str]:
    directory_path += "/**"
    files = list()
    sub_dirs = list(sub_dir.replace("\\", "/") for sub_dir in glob(f"{directory_path}/", recursive=recursive))
    print(prompt)

    for file in list(file.replace("\\", "/") for file in sorted(glob(directory_path, recursive=recursive))):
        if file[-1] == "/":
            continue
        elif f"{file}/" in sub_dirs:
            continue
        elif extension is not None and file.split("/")[-1].split(".")[-1] != extension:
            continue
        elif exclusions is not None and file.split("/")[-1] in exclusions:
            continue
        else:
            files.append(file)

    if not files:
        print(" None")
        input("Press ENTER to exit program: ")
        exit()
    else:
        for id_, file in enumerate(files):
            print(f" - ID: {id_}, Name: '{file.split("/")[-1]}', Size: {format_file_size(getsize(file))}")

        if get_many:
            return list(files[fileID] for fileID in parse_ids(input("Enter file IDs to read from separated by commas/hyphens: ")))
        else:
            return [files[int(input("Enter file ID to read from: "))]]


def get_file_paths(directory_path: str, recursive: bool = False, extension: str = None, exclusions: list[str] = None, prompt: str = "Files found:") -> list[str]:
    """
    Returns a list of absolute paths to files chosen by the user via an interactive CLI from a selection matched.
    :param directory_path: The absolute path of the directory to match files from.
    :param recursive: Match files in subdirectories.
    :param extension: Extension to match files on. If none specified, all extensions will be matched.
    :param exclusions: Paths of files to exclude from those matched.
    :param prompt: Prompt to display to user.
    :return: The list of absolute file paths.
    """

    return _get_file_paths(directory_path, recursive, extension, exclusions, prompt, True)


def get_file_path(directory_path: str, recursive: bool = False, extension: str = None, exclusions: list[str] = None, prompt: str = "Files found:") -> str:
    """
    Returns the absolute path to a file chosen by the user via an interactive CLI from a selection matched.
    :param directory_path: The absolute path of the directory to match files from.
    :param recursive: Match files in subdirectories.
    :param extension: Extension to match files on. If none specified, all extensions will be matched.
    :param exclusions: Paths of files to exclude from those matched.
    :param prompt: Prompt to display to user.
    :return: The absolute file path.
    """

    return _get_file_paths(directory_path, recursive, extension, exclusions, prompt, False)[0]


def colour_wheel(i: int, n: int) -> tuple[float, float, float]:
    """
    Returns the RGB values for a colour with maximum saturation and brightness, with hue determined by input arguments.
    Hue is determined by the ratio of ``i`` to ``n``, where the range ``[0,n)`` of ``i`` maps onto the hue described by the angle ``[0,360)`` degrees.
    :param i: Hue value numerator, in the interval ``[0,n)``.
    :param n: Hue value denominator.
    :return: The RGB values of the colour.
    """

    angle = 360.0 * i / n

    if 0.0 <= angle < 60.0:
        r = 1.0
        g = angle / 60.0
        b = 0.0
    elif 60.0 <= angle < 120.0:
        r = (120.0 - angle) / 60.0
        g = 1.0
        b = 0.0
    elif 120.0 <= angle < 180.0:
        r = 0.0
        g = 1.0
        b = (angle - 120.0) / 60.0
    elif 180.0 <= angle < 240.0:
        r = 0.0
        g = (240.0 - angle) / 60.0
        b = 1.0
    elif 240.0 <= angle < 300.0:
        r = (angle - 240.0) / 60.0
        g = 0.0
        b = 1.0
    elif 300.0 <= angle < 360.0:
        r = 1.0
        g = 0.0
        b = (360.0 - angle) / 60.0
    else:
        raise ValueError("Values of 'i' and 'n' must satisfy: 0 <= i < n")

    return r, g, b


def maximise_brightness(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Maximises the brightness of a colour with RGB values ``(r, g, b)`` while maintaining hue and saturation.
    :param rgb: The RGB values of the input colour.
    :return: The RGB values of the output colour.
    """

    r, g, b = rgb
    brightness = max(r, g, b)
    r, g, b = r / brightness, g / brightness, b / brightness
    return r, g, b


class ProgressBar:
    def __init__(self, total_steps: int, length: int = 50, sig_figs: int = 2, print_function: Callable = print):
        """
        Creates a simple progress bar that can be printed to CLIs to track progress of long-running processes.
        When printed, shows a bar graphic, the percentage of completed steps, the total elapsed time, and the predicted remaining time.
        Remaining time is predicted via a linear extrapolation based on the fraction of completed steps.
        The default print function can be overridden if needed.
        :param total_steps: The total number of steps required for the progress bar to be filled.
        :param length: The length of the bar graphic in monospaced characters.
        :param sig_figs: The number of significant figures for displaying percentage completion.
        :param print_function: Function for overriding default print function.
        """

        self.total_steps = total_steps
        self.length = length
        self.sigfig = sig_figs
        self.print_function = print_function
        self.current_step = 0
        self.last_print_length = 0
        self.start_time = datetime.now()
        self.current_time = self.start_time

    def increment(self) -> None:
        """
        Increments the current progress bar step by one.
        :return: None
        """

        self.current_step += 1

    def set_step(self, step: int) -> None:
        """
        Sets the current progress bar step to the supplied value.
        :param step: The step value.
        :return: None
        """

        self.current_step = step

    def terminate(self) -> None:
        """
        Sets the current progress bar step to the maximum value.
        :return: None
        """

        self.current_step = self.total_steps

    def print(self) -> None:
        """
        Prints the progress bar to the CLI.
        :return: None
        """

        self.current_time = datetime.now()
        bar_fill = int(self.length * self.current_step // self.total_steps)
        bar = "|" + "█" * bar_fill + "-" * (self.length - bar_fill) + "|"
        progress = format_sig_figs_or_int(100 * self.current_step / self.total_steps, self.sigfig) + "%"

        if self.current_step == 0:
            output = f"Progress: {bar} {progress}"
            padding = " " * max(0, self.last_print_length - len(output))
            self.print_function(output + padding, end='\r')
            print_length = max(self.last_print_length, len(output))
        else:
            elapsed_seconds = (self.current_time - self.start_time).total_seconds()
            remaining_seconds = elapsed_seconds * (self.total_steps - self.current_step) / self.current_step
            elapsed_time = format_time_interval(elapsed_seconds)
            remaining_time = format_time_interval(remaining_seconds)
            output = f"Progress: {bar} {progress}   Elapsed: {elapsed_time}   Remaining: {remaining_time}"
            padding = " " * max(0, self.last_print_length - len(output))
            self.print_function(output + padding, end='\r')
            print_length = max(self.last_print_length, len(output))

        self.last_print_length = print_length

    def increment_print(self) -> None:
        """
        Increments the current progress bar step by one and prints it to the CLI.
        :return:
        """

        self.increment()
        self.print()

    def terminate_print(self) -> None:
        """
        Sets the current progress bar step to the maximum value and prints it to the CLI.
        :return: None
        """

        self.terminate()
        self.print()
        self.print_function()
