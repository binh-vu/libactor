from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, TypeGuard, Union
from uuid import uuid4

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


def is_ident_obj_cls(x: type) -> bool:
    return issubclass(x, (IdentObj, LazyIdentObj))


def get_ident_obj_key(x: Union[IdentObj, LazyIdentObj]) -> str:
    return x.key
