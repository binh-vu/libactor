from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Callable,
    Sequence,
    TypeVar,
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

    def get_conversion(self, intype: type, outtype: type) -> UnitTypeConversion:
        if intype is outtype:
            return identity

        if (intype, outtype) in self.unit_type_conversions:
            # we already have a unit type conversion function for these types
            return self.unit_type_conversions[intype, outtype]

        # check if this is a generic conversion
        intype_origin = get_origin(intype)
        intype_args = get_args(intype)

        if intype_origin is None or len(intype_args) != 1:
            raise TypeConversion.UnknownConversion(
                f"Cannot find conversion from {intype} to {outtype}"
            )

        outtype_origin = get_origin(outtype)
        outtype_args = get_args(outtype)

        if outtype_origin is None:
            # we are converting G[T] => T'
            if (
                outtype is not intype_args[0]
                or intype_origin not in self.generic_single_type_conversion
            ):
                # either T != T' or G is unkknown
                raise TypeConversion.UnknownConversion(
                    f"Cannot find conversion from {intype} to {outtype}"
                )
            return self.generic_single_type_conversion[intype_origin]

        # we are converting G[T] => G'[T']
        if (
            outtype_origin is not intype_origin
            or intype_origin not in self.compose_type_conversion
        ):
            # either G != G' or G is unknown
            raise TypeConversion.UnknownConversion(
                f"Cannot find conversion from {intype} to {outtype}"
            )
        # G == G' => T == T'
        compose_func = self.compose_type_conversion[intype_origin]
        func = self.get_conversion(intype_args[0], outtype_args[0])
        return lambda x: compose_func(x, func)
