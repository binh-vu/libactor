from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Protocol, TypeVar, Union


T = TypeVar("T")
P = TypeVar("P")

DataClassInstance = Any
Compression = Literal["snappy", "gzip", "lz4", "zstd", "bz2"]
ArgSer = Callable[[Any], Optional[Union[str, int, bool]]]
NoneType = type(None)
CacheKeyFn = Callable[..., bytes]


class ActorFnTrait(Protocol):
    key: str
