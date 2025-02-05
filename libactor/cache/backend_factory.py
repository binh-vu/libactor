from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from libactor.actor import Actor
from libactor.cache.backend import MemBackend, SqliteBackend, wrap_backend
from libactor.typing import Compression
from typing_extensions import Self


class FuncSqliteBackendFactory:

    @staticmethod
    def pickle(
        dbdir: Path,
        compression: Optional[Compression] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        filename: Optional[str] = None,
        log_serde_time: bool | str = False,
    ):
        def constructor(func, cache_args_helper):
            backend = SqliteBackend(
                dbfile=dbdir / (filename or f"{func.__name__}.sqlite"),
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
    ):
        def constructor(self: Actor, func, cache_args_helper):
            backend = SqliteBackend(
                dbfile=self.actor_dir / (filename or f"{func.__name__}.sqlite"),
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
    ):
        def constructor(self: Actor, func, cache_args_helper):
            backend = SqliteBackend(
                dbfile=self.actor_dir / (filename or f"{func.__name__}.sqlite"),
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
