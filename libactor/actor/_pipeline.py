from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from inspect import isfunction
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    MutableSequence,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from libactor.actor._actor import Actor, P
from libactor.cache.identitied_object import IdentObj
from libactor.misc import get_parallel_executor, identity, typed_delayed
from tqdm import tqdm

InValue = TypeVar("InValue")
OutValue = TypeVar("OutValue")

"""Storing context needed for processing a job"""
Context = TypeVar("Context", bound=Mapping)
NewContext = TypeVar("NewContext", bound=Mapping)


class PipeObject(Actor[P], Generic[P, InValue, OutValue, Context, NewContext]):
    """
    PipeObject is a subclass of Actor that processes a job in a pipeline.
    """

    def forward(self, input: InValue, context: Context) -> tuple[OutValue, NewContext]:
        """
        Args:
            input: The job to be processed.
            context: The context needed for processing the job.

        Returns:
            tuple[OutValue, Context]: The processed job and the updated context
        """
        raise NotImplementedError()


class PipeContextObject(
    PipeObject[P, InValue, InValue, Context, NewContext],
):
    """
    A special PipeObject that does not change the job, but only updates the context.
    """


class Pipeline(Generic[InValue, OutValue, Context, NewContext]):
    """
    A special actor graph that is a linear chain.

    The Pipeline class represents a sequence of actors connected in a linear fashion,
    where each actor processes data and passes it to the next actor in the chain.
    This structure is useful for scenarios where data needs to be processed in a
    step-by-step manner, with each step being handled by a different actor.
    """

    def __init__(
        self,
        pipes: Sequence[PipeObject],
        type_conversions: Optional[
            Sequence[UnitTypeConversion | ComposeTypeConversion]
        ] = None,
    ):
        upd_type_conversions: list[UnitTypeConversion | ComposeTypeConversion] = list(
            type_conversions or []
        )
        upd_type_conversions.append(cast_ident_obj)

        self.pipes = pipes
        self.pipe_transitions = get_pipe_transitions(pipes, upd_type_conversions)

    @overload
    def process(
        self,
        inp: InValue,
        context: Optional[Context | Callable[[], Context]] = None,
        return_context: Literal[False] = False,
    ) -> OutValue: ...

    @overload
    def process(
        self,
        inp: InValue,
        context: Optional[Context | Callable[[], Context]] = None,
        return_context: Literal[True] = True,
    ) -> tuple[OutValue, NewContext]: ...

    def process(
        self,
        inp: InValue,
        context: Optional[Context | Callable[[], Context]] = None,
        return_context: bool = False,
    ) -> OutValue | tuple[OutValue, NewContext]:
        """Process the job through the pipeline."""
        if context is None:
            context = {}  # type: ignore
        elif callable(context):
            context = context()

        val: Any = inp
        for pi, pipe in enumerate(self.pipes):
            val, context = pipe.forward(val, context)
            if pi < len(self.pipe_transitions):
                val = self.pipe_transitions[pi](val)

        if return_context:
            return val, context  # type: ignore
        return val

    def par_process(
        self,
        lst: list[InValue],
        context: Optional[Context | Callable[[], Context]] = None,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        if n_jobs == 1:
            if context is not None:
                context = context() if callable(context) else context

            return list(
                tqdm(
                    (self.process(inp, context) for inp in lst),
                    total=len(lst),
                    disable=not verbose,
                    desc="pipeline processing",
                )
            )

        return list(
            tqdm(
                get_parallel_executor(n_jobs=n_jobs, return_as="generator")(
                    typed_delayed(self.process)(inp, context) for inp in lst
                ),
                total=len(lst),
                disable=not verbose,
                desc="pipeline processing",
            )
        )


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


def get_pipe_transitions(
    pipes: Sequence[PipeObject],
    type_casts: Sequence[UnitTypeConversion | ComposeTypeConversion],
) -> list[Callable]:
    if len(pipes) == 0:
        return []

    conversion = TypeConversion(type_casts)
    transformations = []

    _, prev_intype = get_input_output_type(pipes[0].__class__)
    for pipe in pipes[1:]:
        intype, outtype = get_input_output_type(pipe.__class__)
        transformations.append(conversion.get_conversion(prev_intype, intype))
        prev_intype = outtype

    return transformations


def get_input_output_type(cls: type[PipeObject]) -> tuple[type, type]:
    sig = get_type_hints(cls.forward)
    if get_origin(sig["return"]) is not tuple:
        raise Exception("Invalid return type" + str(get_origin(sig["return"])))

    input_type = sig["input"]
    output_type = get_args(sig["return"])[0]
    return input_type, output_type


UnitTypeConversion = Annotated[
    Callable[[Any], Any], "A function that convert an object of type T1 to T2"
]
ComposeTypeConversion = Annotated[
    Callable[[Any, UnitTypeConversion], Any],
    "A function that convert a generic object of type G[T1] to G[T2]",
]

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def cast_ident_obj(obj: IdentObj[T1], func: Callable[[T1], T2]) -> IdentObj[T2]:
    return IdentObj(obj.key, func(obj.value))
