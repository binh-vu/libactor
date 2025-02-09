from __future__ import annotations

from typing import Any, Generic, Mapping, TypedDict, TypeVar, Union

from libactor.actor._actor import Actor, P

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


class Pipeline(Generic[InValue, OutValue]):
    """
    A special actor graph that is a linear chain.

    The Pipeline class represents a sequence of actors connected in a linear fashion,
    where each actor processes data and passes it to the next actor in the chain.
    This structure is useful for scenarios where data needs to be processed in a
    step-by-step manner, with each step being handled by a different actor.
    """

    def __init__(self, pipes: list[PipeObject]):
        self.pipes = pipes

    def process(self, inp: InValue) -> OutValue:
        """Process the job through the pipeline."""
        context = {}
        val: Any = inp
        for pipe in self.pipes:
            val, context = pipe.forward(val, context)
        return val
