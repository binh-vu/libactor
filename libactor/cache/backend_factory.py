from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Union

from libactor.actor import Actor
from libactor.cache.backend import MemBackend, SqliteBackend, wrap_backend
from libactor.storage import GlobalStorage
from libactor.typing import Compression
from typing_extensions import Self


class FuncSqliteBackendFactory:

    @staticmethod
    def pickle(
        dbdir: Path | Callable[[], Path],
        compression: Optional[Compression] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        filename: Optional[str] = None,
        log_serde_time: bool | str = False,
    ):
        def constructor(func, cache_args_helper):
            if callable(dbdir):
                dbdir_ = dbdir()
            else:
                dbdir_ = dbdir

            backend = SqliteBackend(
                dbfile=dbdir_ / (filename or f"{func.__name__}.sqlite"),
                ser=pickle.dumps,
                deser=pickle.loads,
                compression=compression,
            )
            return wrap_backend(backend, mem_persist, log_serde_time)

        return constructor


class ActorSqliteBackendFactory:

    @staticmethod
    def pickle(
        compression: Optional[Compression] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        filename: Optional[str] = None,
        log_serde_time: bool | str = False,
        get_dbdir: Optional[Callable[[Any], Path]] = None,
    ):
        def constructor(self: Actor, func, cache_args_helper):
            backend = SqliteBackend(
                dbfile=(self.actor_dir if get_dbdir is None else get_dbdir(self))
                / (filename or f"{func.__name__}.sqlite"),
                ser=pickle.dumps,
                deser=pickle.loads,
                compression=compression,
            )
            return wrap_backend(backend, mem_persist, log_serde_time)

        return constructor

    @staticmethod
    def serde(
        *,
        cls: type[DataSerdeMixin],
        compression: Optional[Compression] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        filename: Optional[str] = None,
        log_serde_time: bool | str = False,
        get_dbdir: Optional[Callable[[Any], Path]] = None,
    ):
        def constructor(self: Actor, func, cache_args_helper):
            backend = SqliteBackend(
                dbfile=(self.actor_dir if get_dbdir is None else get_dbdir(self))
                / (filename or f"{func.__name__}.sqlite"),
                ser=cls.ser,
                deser=cls.deser,
                compression=compression,
            )
            return wrap_backend(backend, mem_persist, log_serde_time)

        return constructor


def func_mem_backend_factory(func, cache_args_helper):
    return MemBackend()


def actor_mem_backend_factory(self, func, cache_args_helper):
    return MemBackend()


class FuncBackendFactory:
    sqlite = FuncSqliteBackendFactory
    mem = func_mem_backend_factory

    @staticmethod
    def workdir(func: str, version: int):
        """Get working directory for the function."""
        workdir = GlobalStorage.get_instance().workdir / "funcs" / f"{func}_{version}"
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir


class ActorBackendFactory:
    sqlite = ActorSqliteBackendFactory
    mem = actor_mem_backend_factory


class BackendFactory:
    func = FuncBackendFactory
    actor = ActorBackendFactory


class DataSerdeMixin(ABC):
    """Mixin for serializing and deserializing data to and from bytes. Compression should handle separately such as in the backend."""

    @abstractmethod
    def ser(self) -> bytes: ...

    @classmethod
    @abstractmethod
    def deser(cls, data: bytes) -> Self: ...
