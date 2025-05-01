from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Any, Generic, Optional, TypeGuard, Union, get_origin
from uuid import uuid4

from libactor.misc import get_optional_type, is_optional_type
from libactor.typing import T


@dataclass(slots=True)
class IdentObj(Generic[T]):
    key: str
    value: T

    @staticmethod
    def from_value(value: T) -> IdentObj[T]:
        return IdentObj(str(uuid4()).replace("-", "_"), value)


@dataclass(slots=True)
class LazyIdentObj(Generic[T]):
    key: str
    # way to obtain the value...

    @cached_property
    def value(self):
        raise NotImplementedError()


def is_ident_obj(x: Any) -> TypeGuard[Union[IdentObj, LazyIdentObj]]:
    return isinstance(x, (IdentObj, LazyIdentObj))


def is_ident_obj_cls(x: type, optional: bool = True) -> bool:
    if optional and is_optional_type(x):
        x = get_optional_type(x)
    uox = get_origin(x)
    if uox is None:
        uox = x
    return issubclass(uox, (IdentObj, LazyIdentObj))


def get_ident_obj_key(x: Optional[Union[IdentObj, LazyIdentObj]]) -> Optional[str]:
    if x is None:
        return None
    return x.key


def fmt_keys(*args: str, **kwargs: str | IdentObj) -> str:
    return ",".join(
        chain(
            args,
            (
                f"{k}={v.key}" if is_ident_obj(x=v) else f"{k}={v}"
                for k, v in kwargs.items()
            ),
        )
    )
