from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    overload,
)

import orjson
from joblib import Parallel, delayed
from libactor.typing import Compression, DataClassInstance, T
from serde.helper import AVAILABLE_COMPRESSIONS

TYPE_ALIASES = {"typing.List": "list", "typing.Dict": "dict", "typing.Set": "set"}
CB = TypeVar("CB")


def identity(x: T) -> T:
    return x


def orjson_dumps(obj, **kwargs):
    if "default" not in kwargs:
        kwargs["default"] = _orjson_default
    return orjson.dumps(obj, **kwargs)


def _orjson_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        # so that the order is deterministic, and we can compare actor state by its serialized JSON
        return sorted(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class Chain2:
    def __init__(self, g, f):
        self.g = g
        self.f = f

    def exec(self, args):
        return self.g(self.f(args))

    def __call__(self, args):
        return self.g(self.f(args))


def get_classpath(type: Type | Callable) -> str:
    if type.__module__ == "builtins":
        return type.__qualname__

    if hasattr(type, "__qualname__"):
        return type.__module__ + "." + type.__qualname__

    # typically a class from the typing module
    if hasattr(type, "_name") and type._name is not None:
        path = type.__module__ + "." + type._name
        if path in TYPE_ALIASES:
            path = TYPE_ALIASES[path]
    elif hasattr(type, "__origin__") and hasattr(type.__origin__, "_name"):
        # found one case which is typing.Union
        path = type.__module__ + "." + type.__origin__._name
    else:
        raise NotImplementedError(type)

    return path


def param_as_dict(param: DataClassInstance) -> dict:
    """Convert a dataclass to a dictionary"""
    if not is_dataclass(param):
        raise TypeError(
            f"Parameter must be an instance of a dataclass. Get: {type(param)}"
        )
    return _param_as_dict_inner(param, dict_factory=dict)  # type: ignore


def _param_as_dict_inner(obj, dict_factory):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if is_dataclass(obj):
        result = []
        for f in fields(obj):
            value = _param_as_dict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_param_as_dict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_param_as_dict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (
                _param_as_dict_inner(k, dict_factory),
                _param_as_dict_inner(v, dict_factory),
            )
            for k, v in obj.items()
        )
    # elif isinstance(obj, RelWorkdirPath):
    #     return str(ReamWorkspace.get_instance().get_rel_path(obj.absolute()))
    else:
        return deepcopy(obj)


@dataclass
class NoParams:
    """For an actor that has no parameter"""

    pass


@overload
def to_serde_compression(
    compression: Compression,
) -> AVAILABLE_COMPRESSIONS: ...


@overload
def to_serde_compression(
    compression: Literal[None],
) -> None: ...


def to_serde_compression(
    compression: Optional[Compression],
) -> Optional[AVAILABLE_COMPRESSIONS]:
    if compression is None:
        return None

    if compression == "gzip":
        return "gz"

    seq: Sequence[AVAILABLE_COMPRESSIONS] = ["lz4", "bz2"]
    if compression in seq:
        return compression

    raise Exception(f"Not supported compression: {compression}")


_parallel_executor = None
_parallel_executor_objects = {}


def get_parallel_executor(
    n_jobs: int = -1,
    return_as: Literal["generator_unordered", "generator"] = "generator_unordered",
) -> Callable[[Iterable[T]], Iterable[T]]:
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = Parallel(n_jobs=n_jobs, return_as=return_as)
    return _parallel_executor  # type: ignore


def typed_delayed(func: CB) -> CB:
    return delayed(func)  # type: ignore


def get_cache_object(id: int, obj):
    """Cache the object to avoid reinitializing the objects multiple time in joblib"""
    global _parallel_executor_objects
    if id not in _parallel_executor_objects:
        _parallel_executor_objects[id] = obj
    return _parallel_executor_objects[id]
