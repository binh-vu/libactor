from __future__ import annotations

import bz2
import gzip
import pickle
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Union

from hugedict.sqlite import SqliteDict, SqliteDictFieldType

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
    sqlite: SqliteBackendFactory

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

    def postinit(self, func: Callable, args_helper: CacheArgsHelper):
        if not hasattr(self, "_is_postinited"):
            self._is_postinited = True
        else:
            raise RuntimeError("Backend can only be postinited once")

    @abstractmethod
    def has_key(self, key: str) -> bool: ...

    @abstractmethod
    def get(self, key: str) -> T: ...

    @abstractmethod
    def set(self, key: str, value: T) -> None: ...


class SqliteBackend(Backend):
    def __init__(
        self,
        ser: Callable[[Any], bytes],
        deser: Callable[[bytes], Any],
        dbdir: Path,
        filename: Optional[str] = None,
        compression: Optional[Compression] = None,
    ):
        super().__init__(ser, deser, compression)
        self.filename = filename
        self.dbdir = dbdir

    def postinit(self, func: Callable, args_helper: CacheArgsHelper):
        super().postinit(func, args_helper)
        if self.filename is None:
            self.filename = f"{func.__name__}.sqlite"

        self.dbconn: SqliteDict = SqliteDict(
            self.dbdir / self.filename,
            keytype=SqliteDictFieldType.bytes,
            ser_value=identity,
            deser_value=identity,
        )

    def has_key(self, key: bytes) -> bool:
        return key in self.dbconn

    def get(self, key: bytes) -> Any:
        return self.deser(self.dbconn[key])

    def set(self, key: bytes, value: Any) -> None:
        self.dbconn[key] = self.ser(value)


class MemBackend(Backend):

    def __init__(self, cache_obj: Optional[dict[bytes, Any]] = None):
        self.cache_obj = cache_obj or {}

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        """"""
        oid = id(obj)
        if oid not in self.id2cache:
            self.id2cache[oid] = {}

            # create a weakref
            def cleanup(ref):
                del self.id2cache[oid]
                del self.id2ref[oid]

            self.id2ref[oid] = weakref.ref(obj, cleanup)

        self.current_id = oid
        yield None

    def has_key(self, key: bytes) -> bool:
        assert self.current_id is not None
        return key in self.id2cache[self.current_id]

    def get(self, key: bytes) -> Any:
        assert self.current_id is not None
        return self.id2cache[self.current_id][key]

    def set(self, key: bytes, value: Any) -> None:
        assert self.current_id is not None
        self.id2cache[self.current_id][key] = value

    def clear(self):
        self.id2cache.clear()
        self.id2ref.clear()


class SqliteBackendFactory:
    @staticmethod
    def pickle(
        actor: Actor,
        compression: Optional[Compression] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        filename: Optional[str] = None,
        log_serde_time: bool | str = False,
    ):
        backend = SqliteBackend(
            ser=pickle.dumps,
            deser=pickle.loads,
            dbdir=actor.actor_dir,
            filename=filename,
            compression=compression,
        )
        return wrap_backend(backend, mem_persist, log_serde_time)


class LogSerdeTimeBackend(Backend):
    def __init__(self, backend: Backend, name: str = ""):
        self.backend = backend
        self.logger: Logger = None  # type: ignore
        self.name = name + " " if len(name) > 0 else name

    def postinit(self, func: Callable, args_helper: CacheArgsHelper):
        self.backend.postinit(func, args_helper)
        self.logger = logger

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        if self.logger is None:
            self.logger = logger.bind(name=obj.__class__.__name__)
        with self.backend.context(obj, *args, **kwargs):
            yield None

    def has_key(self, key: bytes) -> bool:
        return self.backend.has_key(key)

    def get(self, key: bytes) -> Value:
        with Timer().watch_and_report(
            f"{self.name}deserialize",
            self.logger.debug,
        ):
            return self.backend.get(key)

    def set(self, key: bytes, value: Value) -> None:
        with Timer().watch_and_report(
            f"{self.name}serialize",
            self.logger.debug,
        ):
            self.backend.set(key, value)


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
