from __future__ import annotations

import bz2
import gzip
import pickle
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Union

from hugedict.sqlite import SqliteDict, SqliteDictFieldType
from loguru import logger
from timer import Timer

from libactor.actor.actor import Actor
from libactor.cache.cache_args import CacheArgsHelper
from libactor.misc import Chain2, identity
from libactor.storage.global_storage import GlobalStorage
from libactor.typing import Compression, T

try:
    import lz4.frame as lz4_frame  # type: ignore
except ImportError:
    lz4_frame = None


class Backend(Generic[T], ABC):
    def __init__(
        self,
        ser: Callable[[T], bytes],
        deser: Callable[[bytes], T],
        compression: Optional[Compression] = None,
    ):
        if compression == "gzip":
            origin_ser = ser
            origin_deser = deser
            ser = lambda x: gzip.compress(origin_ser(x), mtime=0)
            deser = lambda x: origin_deser(gzip.decompress(x))
        elif compression == "bz2":
            origin_ser = ser
            origin_deser = deser
            ser = lambda x: bz2.compress(origin_ser(x))
            deser = lambda x: origin_deser(bz2.decompress(x))
        elif compression == "lz4":
            if lz4_frame is None:
                raise ValueError("lz4 is not installed")
            # using lambda somehow terminate the program without raising an error
            ser = Chain2(lz4_frame.compress, ser)
            deser = Chain2(deser, lz4_frame.decompress)
        else:
            assert compression is None, compression

        self.compression = compression
        self.ser = ser
        self.deser = deser

    @abstractmethod
    def has_key(self, key: str) -> bool: ...

    @abstractmethod
    def get(self, key: str) -> T: ...

    @abstractmethod
    def set(self, key: str, value: T) -> None: ...


class SqliteBackend(Backend):
    def __init__(
        self,
        dbfile: Path,
        ser: Callable[[Any], bytes],
        deser: Callable[[bytes], Any],
        compression: Optional[Compression] = None,
    ):
        super().__init__(ser, deser, compression)
        self.dbfile = dbfile
        self.dbconn = SqliteDict(
            self.dbfile,
            keytype=SqliteDictFieldType.bytes,
            ser_value=identity,
            deser_value=identity,
            timeout=30,
        )

    def has_key(self, key: bytes) -> bool:
        return key in self.dbconn

    def get(self, key: bytes) -> Any:
        return self.deser(self.dbconn[key])

    def set(self, key: bytes, value: Any) -> None:
        self.dbconn[key] = self.ser(value)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            SqliteBackend,
            (self.dbfile, self.ser, self.deser, self.compression),
        )


class MemBackend(Backend):

    def __init__(self, cache_obj: Optional[dict[bytes, Any]] = None):
        self.cache_obj = cache_obj or {}

    def has_key(self, key: bytes) -> bool:
        return key in self.cache_obj

    def get(self, key: bytes) -> Any:
        return self.cache_obj[key]

    def set(self, key: bytes, value: Any) -> None:
        self.cache_obj[key] = value

    def clear(self):
        self.cache_obj.clear()


class LogSerdeTimeBackend(Backend):
    def __init__(self, backend: Backend, name: str = ""):
        self.backend = backend
        self.name = name + " " if len(name) > 0 else name

    def has_key(self, key: str) -> bool:
        return self.backend.has_key(key)

    def get(self, key: str) -> Any:
        with Timer().watch_and_report(
            f"{self.name}deserialize",
            logger.debug,
        ):
            return self.backend.get(key)

    def set(self, key: str, value: Any) -> None:
        with Timer().watch_and_report(
            f"{self.name}serialize",
            logger.debug,
        ):
            self.backend.set(key, value)


class ReplicatedBackends(Backend):
    """A composite backend that a backend (i) is a super set (key-value) of
    its the previous backend (i-1). Accessing to this composite backend will
    slowly build up the front backends to have the same key-value pairs as
    the last backend.

    This is useful for combining MemBackend and DiskBackend.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def has_key(self, key: str) -> bool:
        return any(backend.has_key(key) for backend in self.backends)

    def get(self, key: str) -> Any:
        for i, backend in enumerate(self.backends):
            if backend.has_key(key):
                value = backend.get(key)
                if i > 0:
                    # replicate the value to the previous backend
                    for j in range(i):
                        self.backends[j].set(key, value)
                return value

    def set(self, key: str, value: Any):
        for backend in self.backends:
            backend.set(key, value)


def wrap_backend(
    backend: Backend,
    mem_persist: Optional[Union[MemBackend, bool]],
    log_serde_time: str | bool,
):
    if log_serde_time:
        backend = LogSerdeTimeBackend(
            backend, name="" if isinstance(log_serde_time, bool) else log_serde_time
        )
    if mem_persist:
        if mem_persist is not None:
            mem_persist = MemBackend()
        backend = ReplicatedBackends([mem_persist, backend])
    return backend
