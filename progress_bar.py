import datetime
import utilities
from typing import Self


class ProgressBar:
    def __init__(self, total_steps: int, length: int = 50, sigfig: int = 2):
        self.total_steps = total_steps
        self.length = length
        self.sigfig = sigfig
        self.current_step = 0
        self.last_print_length = 0
        self.start_time = datetime.datetime.now()
        self.current_time = self.start_time

    def increment(self) -> Self:
        self.current_step += 1
        return self

    def set_step(self, step: int) -> Self:
        self.current_step = step
        return self

    def terminate(self) -> Self:
        self.current_step = self.total_steps
        return self

    def print(self) -> Self:
        self.current_time = datetime.datetime.now()
        bar_fill = int(self.length * self.current_step // self.total_steps)
        bar = "|" + "█" * bar_fill + "-" * (self.length - bar_fill) + "|"
        progress = str(utilities.intSigFig(100 * self.current_step / self.total_steps, self.sigfig)) + "%"

        if self.current_time == self.start_time:
            output = f"Progress: {bar} {progress}"
            padding = " " * max(0, self.last_print_length - len(output))
            print(output + padding, end='\r')
            print_length = max(self.last_print_length, len(output))
        else:
            elapsed_seconds = (self.current_time - self.start_time).total_seconds()
            remaining_seconds = elapsed_seconds * (self.total_steps - self.current_step) / self.current_step
            elapsed_time = utilities.format_time(elapsed_seconds)
            remaining_time = utilities.format_time(remaining_seconds)
            output = f"Progress: {bar} {progress}   Elapsed: {elapsed_time}   Remaining: {remaining_time}"
            padding = " " * max(0, self.last_print_length - len(output))
            print(output + padding, end='\r')
            print_length = max(self.last_print_length, len(output))

        if self.current_step == self.total_steps:
            print()

        self.last_print_length = print_length
        return self

    def increment_print(self) -> Self:
        return self.increment().print()

    def terminate_print(self) -> Self:
        return self.terminate().print()
