# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from enum import Enum

from src.utilities.utils import timestamp


class LogLevel(Enum):
  DEBUG = 1
  INFO = 2
  WARNING = 3
  ERROR = 4


class Logger(ABC):
    @abstractmethod
    def debug(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def info(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def warn(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def error(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def input(self, prompt: object = "") -> str:
        ...


class PrintLogger(Logger):
    def __init__(self, level: LogLevel):
        self.level = level

    def debug(self, *args, **kwargs) -> None:
        if self.level.value <= LogLevel.DEBUG.value:
            print(timestamp(), "DEBUG", *args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        if self.level.value <= LogLevel.INFO.value:
            print(timestamp(), "INFO ", *args, **kwargs)

    def warn(self, *args, **kwargs) -> None:
        if self.level.value <= LogLevel.WARNING.value:
            print(timestamp(), "WARN ", *args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        if self.level.value <= LogLevel.ERROR.value:
            print(timestamp(), "ERROR", *args, **kwargs)

    def input(self, prompt: object = "") -> str:
        return input(timestamp() + "INPUT" + str(prompt) + ": ")

