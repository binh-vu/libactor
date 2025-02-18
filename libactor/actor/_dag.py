from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

from graph.interface import BaseEdge, BaseNode
from graph.retworkx import RetworkXStrDiGraph
from libactor.actor._actor import Actor
from libactor.misc import TypeConversion


class Cardinality(enum.Enum):
    ONE_TO_ONE = 1
    MANY_TO_ONE = 2
    ONE_TO_MANY = 3
    MANY_TO_MANY = 4


@dataclass
class InputStreamSpecs:
    nodes: list[str] = field(default_factory=list)
    cardinalities: list[Cardinality] = field(default_factory=list)

    def is_single_stream(self):
        return all(car == Cardinality.ONE_TO_ONE for car in self.cardinalities)

    def emit(self, *args):
        assert all(
            car == Cardinality.ONE_TO_ONE or car == Cardinality.ONE_TO_MANY
            for car in self.cardinalities
        )
        if self.is_single_stream():
            yield args
            return

        many_index = []
        remain_index = []
        manys = []
        for i, car in enumerate(self.cardinalities):
            if car == Cardinality.ONE_TO_MANY:
                manys.append(args[i])
                many_index.append(i)
            else:
                remain_index.append(i)

        output = [None] * len(self.nodes)
        for i in remain_index:
            output[i] = args[i]

        for cp in itertools.product(*manys):
            for i, c in zip(many_index, cp):
                output[i] = c
            yield output


class ActorNode(BaseNode):
    def __init__(self, id: str, actor: Actor, input_spec: InputStreamSpecs):
        self.id = id
        self.actor = actor
        self.input_spec = input_spec


class DAG:

    def __init__(
        self,
        graph: RetworkXStrDiGraph[int, ActorNode, BaseEdge],
        type_conversion: TypeConversion,
    ) -> None:
        self.graph = graph
        self.type_conversion = type_conversion

    @staticmethod
    def from_dictmap(dictmap: dict[str, Actor | tuple[InputStreamSpecs, Actor]]):
        """Create a DAG from a dictionary mapping.

        Args:
            nodes: A dictionary where the key is a string and the value is either an Actor or a tuple containing a list of upstream actor ids and an Actor.

        Returns:
            DAG: A directed acyclic graph (DAG) constructed from the provided dictionary mapping.
        """
        g: RetworkXStrDiGraph[int, ActorNode, BaseEdge] = RetworkXStrDiGraph(
            check_cycle=True, multigraph=False
        )

        # create a graph first
        for uid, uinfo in dictmap.items():
            if isinstance(uinfo, tuple):
                upstream_actors, actor = uinfo
            else:
                upstream_actors = InputStreamSpecs()
                actor = uinfo
            g.add_node(ActorNode(uid, actor, upstream_actors))
        for vid, vinfo in dictmap.items():
            if isinstance(vinfo, tuple):
                upstream_actors, actor = vinfo
            else:
                upstream_actors = InputStreamSpecs()
                actor = vinfo

            for i, uid in enumerate(upstream_actors.nodes):
                g.add_edge(BaseEdge(id=-1, source=uid, target=vid, key=i))

        # add typing conversions
        type_conversion = TypeConversion([])

        return DAG(g, type_conversion)

    def process(self, actor_inargs: dict[str, list], capture_actors: set[str]):
        stack = list(actor_inargs.keys())
        capture_output = {}

        while len(stack) > 0:
            actor_id = stack.pop()
            actor_node = self.graph.get_node(actor_id)
            inargs = actor_inargs.pop(actor_id)

            if len(actor_node.input_spec.nodes) == 0:
                # actor is the stream source, process it.
                output = actor_node.actor.forward(*inargs)
            else:
                # not a stream source
                if actor_node.input_spec.is_single_stream():
                    output = actor_node.actor.new_forward(*inargs)
                else:
                    output = [
                        actor_node.actor.new_forward(*x)
                        for x in actor_node.input_spec.emit(*inargs)
                    ]

            for outedge in self.graph.out_edges(actor_id):
                # store the output in the input args of the downstream actor.
                actor_inargs[outedge.target][outedge.key] = output

            if actor_id in capture_actors:
                capture_output[actor_id] = output

        return capture_output
