from __future__ import annotations

import collections.abc
from locale import normalize
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from libactor.misc._misc import identity

UnitTypeConversion = Annotated[
    Callable[[Any], Any], "A function that convert an object of type T1 to T2"
]
ComposeTypeConversion = Annotated[
    Callable[[Any, UnitTypeConversion], Any],
    "A function that convert a generic object of type G[T1] to G[T2]",
]


class TypeConversion:
    """Inspired by Rust type conversion traits. This class allows to derive a type conversion function from output of a pipe to input of another pipe."""

    class UnknownConversion(Exception):
        pass

    def __init__(
        self, type_casts: Sequence[UnitTypeConversion | ComposeTypeConversion]
    ):
        self.generic_single_type_conversion: dict[type, UnitTypeConversion] = {}
        self.unit_type_conversions: dict[tuple[type, type], UnitTypeConversion] = {}
        self.compose_type_conversion: dict[type, ComposeTypeConversion] = {}

        for fn in type_casts:
            sig = get_type_hints(fn)
            if len(sig) == 2:
                fn = cast(UnitTypeConversion, fn)

                intype = sig[[x for x in sig if x != "return"][0]]
                outtype = sig["return"]

                intype_origin = get_origin(intype)
                intype_args = get_args(intype)
                if (
                    intype_origin is not None
                    and len(intype_args) == 1
                    and intype_args[0] is outtype
                    and isinstance(outtype, TypeVar)
                ):
                    # this is a generic conversion G[T] => T
                    self.generic_single_type_conversion[intype_origin] = fn
                else:
                    self.unit_type_conversions[intype, outtype] = fn
            else:
                assert len(sig) == 3, "Invalid type conversion function"
                fn = cast(ComposeTypeConversion, fn)

                intype = sig[[x for x in sig if x != "return"][0]]
                outtype = sig["return"]
                intype_origin = get_origin(intype)
                assert intype_origin is not None
                self.compose_type_conversion[intype_origin] = fn

    def get_conversion(
        self, source_type: type, target_type: type
    ) -> UnitTypeConversion:
        # handle identity conversion
        # happen when source_type = target_type or target_type is Union[source_type, ...]
        if source_type == target_type:
            # source_type is target_type doesn't work with collections.abc.Sequence
            return identity
        if get_origin(target_type) in (Union, UnionType) and source_type in get_args(
            target_type
        ):
            return identity

        if (source_type, target_type) in self.unit_type_conversions:
            # we already have a unit type conversion function for these types
            return self.unit_type_conversions[source_type, target_type]

        # check if this is a generic conversion
        intype_origin = get_origin(source_type)
        intype_args = get_args(source_type)

        if intype_origin is None or len(intype_args) != 1:
            raise TypeConversion.UnknownConversion(
                f"Cannot find conversion from {source_type} to {target_type}"
            )

        outtype_origin = get_origin(target_type)
        outtype_args = get_args(target_type)

        if outtype_origin is None:
            # we are converting G[T] => T'
            if (
                target_type is not intype_args[0]
                or intype_origin not in self.generic_single_type_conversion
            ):
                # either T != T' or G is unkknown
                raise TypeConversion.UnknownConversion(
                    f"Cannot find conversion from {source_type} to {target_type}"
                )
            return self.generic_single_type_conversion[intype_origin]

        # we are converting G[T] => G'[T']
        if (
            outtype_origin is not intype_origin
            or intype_origin not in self.compose_type_conversion
        ):
            # either G != G' or G is unknown
            raise TypeConversion.UnknownConversion(
                f"Cannot find conversion from {source_type} to {target_type}"
            )
        # G == G' => T == T'
        compose_func = self.compose_type_conversion[intype_origin]
        func = self.get_conversion(intype_args[0], outtype_args[0])
        return lambda x: compose_func(x, func)


def patch_get_origin(t: type) -> Any:
    """The original get_origin(typing.Sequence) returns collections.abc.Sequence.
    Later comparing typing.Sequence[T] to collections.abc.Sequence[T] aren't equal.

    This function will return typing.Sequence instead.
    """
    origin = get_origin(t)
    if origin is None:
        return origin
    return {
        collections.abc.Mapping: Mapping,
        collections.abc.Sequence: Sequence,
        collections.abc.MutableSequence: MutableSequence,
        collections.abc.MutableMapping: MutableMapping,
        collections.abc.Set: Set,
        collections.abc.MutableSet: MutableSet,
    }.get(origin, origin)


def is_generic_type(t: type) -> bool:
    return isinstance(t, TypeVar) or any(is_generic_type(a) for a in get_args(t))


def align_generic_type(
    generic_type: type, target_type: type
) -> tuple[type, tuple[type, type]]:
    """Return the grounded outer type, and the mapping from the TypeVar to the concrete type"""
    if isinstance(generic_type, TypeVar):
        return target_type, (generic_type, target_type)

    origin = patch_get_origin(generic_type)
    assert origin is not None
    if origin != patch_get_origin(target_type):
        raise TypeConversion.UnknownConversion(
            f"Cannot ground generic type {generic_type} to {target_type}"
        )

    if len(get_args(generic_type)) != 1:
        raise NotImplementedError()

    gt = align_generic_type(get_args(generic_type)[0], get_args(target_type)[0])
    return origin[gt[0]], gt[1]


def ground_generic_type(generic_type: type, var2type: dict[TypeVar, type]) -> type:
    if isinstance(generic_type, TypeVar):
        return var2type[generic_type]

    origin = get_origin(generic_type)
    if origin is None:
        # nothing to ground
        return generic_type

    return origin[*(ground_generic_type(t, var2type) for t in get_args(generic_type))]
