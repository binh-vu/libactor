from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    Optional,
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
from libactor.misc import (
    TypeConversion,
    get_cache_object,
    get_parallel_executor,
    identity,
    typed_delayed,
)
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

        ppid = id(self)
        ppobj = self

        def invoke(inp, context):
            return get_cache_object(
                ppid,
                ppobj,
            ).process(inp, context)

        return list(
            tqdm(
                get_parallel_executor(n_jobs=n_jobs, return_as="generator")(
                    typed_delayed(invoke)(inp, context) for inp in lst
                ),
                total=len(lst),
                disable=not verbose,
                desc="pipeline processing",
            )
        )


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
