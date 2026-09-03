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
    def warning(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def error(self, *args, **kwargs) -> None:
        ...


class PrintLogger(Logger):
    def __init__(self, level: LogLevel):
        self._level = level

    def debug(self, *args, **kwargs) -> None:
        if self._level.value <= LogLevel.DEBUG.value:
            print(timestamp(), "DEBUG", *args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        if self._level.value <= LogLevel.INFO.value:
            print(timestamp(), "INFO ", *args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        if self._level.value <= LogLevel.WARNING.value:
            print(timestamp(), "WARN ", *args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        if self._level.value <= LogLevel.ERROR.value:
            print(timestamp(), "ERROR", *args, **kwargs)
