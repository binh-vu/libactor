from __future__ import annotations

import enum
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Callable,
    Optional,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from graph.interface import BaseEdge, BaseNode
from graph.retworkx import RetworkXStrDiGraph, topological_sort
from libactor.actor._actor import Actor
from libactor.cache import IdentObj
from libactor.misc import TypeConversion, identity
from libactor.misc._type_conversion import (
    ComposeTypeConversion,
    UnitTypeConversion,
    align_generic_type,
    ground_generic_type,
    is_generic_type,
)


class SupportCardinality(enum.Enum):
    ONE_TO_ONE = 1
    ONE_TO_MANY = 2


ComputeFnId = Annotated[str, "ComputeFn Identifier"]
ComputeFn = Actor | Callable


class Flow:
    def __init__(
        self,
        source: list[ComputeFnId] | ComputeFnId,
        target: ComputeFn,
        cardinality: SupportCardinality = SupportCardinality.ONE_TO_ONE,
    ):
        self.source = [source] if isinstance(source, str) else source
        self.cardinality = cardinality
        self.target = target

    def __post_init__(self):
        if (self.cardinality == SupportCardinality.ONE_TO_MANY) and len(
            self.source
        ) != 1:
            raise Exception("Can't have multiple sources for ONE_TO_MANY")


@dataclass
class FnSignature:
    return_type: type
    argnames: list[str]
    argtypes: list[type]


class ActorNode(BaseNode[ComputeFnId]):
    def __init__(
        self,
        id: ComputeFnId,
        actor: ComputeFn,
        sorted_outedges: Optional[Sequence[ActorEdge]] = None,
    ):
        self.id = id
        self.actor = actor
        self.signature = ActorNode.get_signature(self.actor)
        self.sorted_outedges: Annotated[
            Sequence[ActorEdge], "Outgoing edges sorted in topological order"
        ] = (sorted_outedges or [])
        self.type_conversions: list[UnitTypeConversion] = []
        self.topo_index: int = 0
        self.required_args: list[str] = []
        self.required_context: list[str] = []

    @staticmethod
    def get_signature(actor: ComputeFn) -> FnSignature:
        if isinstance(actor, Actor):
            sig = get_type_hints(actor.new_forward)
        else:
            sig = get_type_hints(actor)

        argnames = list(sig.keys())[:-1]
        try:
            return FnSignature(
                sig["return"],
                argnames,
                [sig[arg] for arg in argnames],
            )
        except:
            print("Cannot figure out the signature of", actor)
            print("The parsed signature is:", sig)
            raise

    def invoke(self, args: Sequence, context: Sequence):
        norm_args = (self.type_conversions[i](a) for i, a in enumerate(args))
        return self.get_func()(*norm_args, *context)

    def get_func(self):
        if isinstance(self.actor, Actor):
            return self.actor.new_forward
        else:
            return self.actor


class ActorEdge(BaseEdge[str, int]):

    def __init__(
        self,
        id: int,
        source: str,
        target: str,
        argindex: int,
        cardinality: SupportCardinality,
        type_conversion: UnitTypeConversion,
    ):
        super().__init__(id, source, target, argindex)
        self.argindex = argindex
        self.cardinality = cardinality
        self.type_conversion = type_conversion


class DAG:

    def __init__(
        self,
        graph: RetworkXStrDiGraph[int, ActorNode, ActorEdge],
        type_conversion: TypeConversion,
    ) -> None:
        self.graph = graph
        self.type_conversion = type_conversion

    @staticmethod
    def from_dictmap(
        dictmap: dict[ComputeFnId, Flow | ComputeFn],
        type_conversions: Optional[
            Sequence[UnitTypeConversion | ComposeTypeConversion]
        ] = None,
        strict: bool = True,
    ):
        """Create a DAG from a dictionary mapping.

        Args:
            dictmap: A dictionary mapping of actor identifiers to actors or tuples of upstream actors and the actor.
            type_conversions: A list of type conversions to be used for converting the input types.
            strict: If True, we do type checking.

        Returns:
            DAG: A directed acyclic graph (DAG) constructed from the provided dictionary mapping.
        """
        # add typing conversions
        upd_type_conversions: list[UnitTypeConversion | ComposeTypeConversion] = list(
            type_conversions or []
        )
        upd_type_conversions.append(cast_ident_obj)
        type_service = TypeConversion(upd_type_conversions)

        g: RetworkXStrDiGraph[int, ActorNode, ActorEdge] = RetworkXStrDiGraph(
            check_cycle=True, multigraph=False
        )

        # create a graph
        for uid, uinfo in dictmap.items():
            if isinstance(uinfo, Flow):
                actor = uinfo.target
            else:
                actor = uinfo
            g.add_node(ActorNode(uid, actor))

        # grounding function that has generic type input and output
        for uid, flow in dictmap.items():
            if not isinstance(flow, Flow):
                continue

            u = g.get_node(uid)
            usig = u.signature
            if is_generic_type(usig.return_type) or any(
                is_generic_type(t) for t in usig.argtypes
            ):
                var2type = {}
                for i, t in enumerate(usig.argtypes):
                    if is_generic_type(t):
                        # align the generic type with the previous return type
                        if len(flow.source) <= i and strict:
                            raise TypeConversion.UnknownConversion(
                                f"Cannot ground the generic type based on upstream actors for actor {uid}"
                            )

                        source_return_type = g.get_node(
                            flow.source[i]
                        ).signature.return_type
                        if flow.cardinality == SupportCardinality.ONE_TO_MANY:
                            source_return_type = get_args(source_return_type)[0]
                        usig.argtypes[i], (var, nt) = align_generic_type(
                            t, source_return_type
                        )
                        var2type[var] = nt
                if is_generic_type(usig.return_type):
                    usig.return_type = ground_generic_type(
                        usig.return_type,
                        var2type,
                    )

        for uid, flow in dictmap.items():
            if not isinstance(flow, Flow):
                continue

            u = g.get_node(uid)
            usig = u.signature
            if flow.cardinality == SupportCardinality.ONE_TO_MANY:
                # check if the return type is a generic type with a sequence as its origin (S[T])
                s = g.get_node(flow.source[0])
                ssig = s.signature

                ssig_return_origin = get_origin(ssig.return_type)
                ssig_return_args = get_args(ssig.return_type)
                cast_fn = identity
                if (
                    ssig_return_origin is None
                    or not issubclass(ssig_return_origin, Sequence)
                    or len(ssig_return_args) != 1
                ):
                    # we do not know how to convert this
                    if strict:
                        raise TypeConversion.UnknownConversion(
                            f"Cannot find conversion from {ssig.return_type} to {usig.argtypes[0]}"
                        )
                else:
                    try:
                        cast_fn = type_service.get_conversion(
                            ssig_return_args[0], usig.argtypes[0]
                        )
                    except:
                        if strict:
                            raise

                g.add_edge(
                    ActorEdge(
                        id=-1,
                        source=s.id,
                        target=u.id,
                        argindex=0,
                        cardinality=flow.cardinality,
                        type_conversion=cast_fn,
                    )
                )
            # elif flow.cardinality == SupportCardinality.MANY_TO_ONE:
            #     s = g.get_node(flow.source[0])
            #     ssig = s.signature

            #     usig_input_origin = get_origin(usig.argtypes[0])
            #     usig_input_args = get_args(usig.argtypes[0])
            #     cast_fn = identity
            #     if (
            #         usig_input_origin is None
            #         or not issubclass(usig_input_origin, Sequence)
            #         or len(usig_input_args) != 1
            #     ):
            #         # we do not know how to convert this
            #         if strict:
            #             raise TypeConversion.UnknownConversion(
            #                 f"Cannot find conversion from {ssig.return_type} to {usig.argtypes[0]}"
            #             )
            #     else:
            #         try:
            #             cast_fn = type_service.get_conversion(
            #                 ssig.return_type, usig_input_args[0]
            #             )
            #         except:
            #             if strict:
            #                 raise
            #     g.add_edge(
            #         ActorEdge(
            #             id=-1,
            #             source=s.id,
            #             target=u.id,
            #             argindex=0,
            #             cardinality=flow.cardinality,
            #             type_conversion=cast_fn,
            #         )
            #     )
            else:
                for idx, sid in enumerate(flow.source):
                    s = g.get_node(sid)
                    ssig = s.signature
                    cast_fn = identity
                    try:
                        cast_fn = type_service.get_conversion(
                            ssig.return_type, usig.argtypes[idx]
                        )
                    except Exception as e:
                        if strict:
                            raise TypeConversion.UnknownConversion(
                                f"Don't know how to convert output of {sid} to input of {uid}"
                            ) from e
                    g.add_edge(
                        ActorEdge(
                            id=-1,
                            source=sid,
                            target=uid,
                            argindex=idx,
                            cardinality=flow.cardinality,
                            type_conversion=cast_fn,
                        )
                    )

            # arguments of a compute function that are not provided by the upstream actors must be provided by the context.
            u.required_args = usig.argnames[: len(flow.source)]
            u.required_context = usig.argnames[len(flow.source) :]

        # sort the outedges of each node in topological order
        actor2topo = {uid: i for i, uid in enumerate(topological_sort(g))}
        for u in g.iter_nodes():
            u.topo_index = actor2topo[u.id]
            u.sorted_outedges = sorted(
                g.out_edges(u.id), key=lambda x: actor2topo[x.target]
            )
            inedges = g.in_edges(u.id)
            u.type_conversions = [identity] * len(u.signature.argnames)
            for inedge in inedges:
                u.type_conversions[inedge.argindex] = inedge.type_conversion

        return DAG(g, type_service)

    def process(
        self,
        actor_inargs: dict[ComputeFnId, tuple],
        context: dict[str, Callable | Any],
        capture_actors: set[str],
    ) -> dict[str, list]:
        context = {k: v() if callable(v) else v for k, v in context.items()}
        actor2context = {}
        actor2args: dict[ComputeFnId, list | deque[tuple]] = {}
        actor2incard: dict[ComputeFnId, SupportCardinality] = {}

        for u in self.graph.iter_nodes():
            actor2context[u.id] = tuple(context[name] for name in u.required_context)
            inedges = self.graph.in_edges(u.id)
            if (
                len(inedges) > 0
                and inedges[0].cardinality == SupportCardinality.ONE_TO_MANY
            ):
                actor2args[u.id] = []
                actor2incard[u.id] = SupportCardinality.ONE_TO_MANY
            else:
                actor2args[u.id] = [None] * len(u.required_args)
                actor2incard[u.id] = SupportCardinality.ONE_TO_ONE

        stack: list[ComputeFnId] = []
        capture_output: dict[ComputeFnId, Any] = defaultdict(list)

        for uid, args in sorted(
            actor_inargs.items(),
            key=lambda x: self.graph.get_node(x[0]).topo_index,
            reverse=True,
        ):
            u = self.graph.get_node(uid)
            result = u.invoke(args, actor2context[uid])
            if (
                len(u.sorted_outedges) == 1
                and u.sorted_outedges[0].cardinality != SupportCardinality.ONE_TO_ONE
            ):
                if u.sorted_outedges[0].cardinality == SupportCardinality.ONE_TO_MANY:
                    actor2args[u.sorted_outedges[0].target] = deque(
                        (x,) for x in reversed(result)
                    )
                else:
                    assert False, "Unreachable code"
                    # assert (
                    #     u.sorted_outedges[0].cardinality
                    #     == SupportCardinality.MANY_TO_ONE
                    # )
                    # result = [result]
                    # actor2args[outedge.target] = [(result,)]
                stack.append(u.sorted_outedges[0].target)
            else:
                for outedge in reversed(u.sorted_outedges):
                    intup = [None] * len(
                        self.graph.get_node(outedge.target).required_args
                    )
                    intup[outedge.argindex] = result
                    actor2args[outedge.target] = intup
                    stack.append(outedge.target)

        while len(stack) > 0:
            uid = stack[-1]
            u = self.graph.get_node(uid)

            # get next arguments to process
            if actor2incard[uid] == SupportCardinality.ONE_TO_MANY:
                u_lst_args: deque[tuple] = actor2args[uid]  # type: ignore
                u_args = u_lst_args.popleft()
                if len(u_lst_args) == 0:
                    # we do not have any more arguments to process for this actor
                    # we pop it out of the stack.
                    stack.pop()
            else:
                u_args = actor2args[uid]
                # done with this actor, we remove it from the stack.
                stack.pop()

            # invoke the actor
            result = u.invoke(u_args, actor2context[uid])

            # capture the output if needed
            if uid in capture_actors:
                capture_output[uid].append(result)

            # propagate the result to the downstream actors
            if (
                len(u.sorted_outedges) > 0
                and u.sorted_outedges[0].cardinality == SupportCardinality.ONE_TO_MANY
            ):
                actor2args[u.sorted_outedges[0].target].extend((x,) for x in result)
                stack.append(u.sorted_outedges[0].target)
            else:
                for outedge in reversed(u.sorted_outedges):
                    actor2args[outedge.target][outedge.argindex] = result
                    stack.append(outedge.target)

        return capture_output

    def par_process(
        self,
        actor_inargs: dict[ComputeFnId, tuple],
        context: dict[str, Callable | Any],
        capture_actors: set[str],
    ) -> dict[str, list]: ...


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def cast_ident_obj(obj: IdentObj[T1], func: Callable[[T1], T2]) -> IdentObj[T2]:
    return IdentObj(obj.key, func(obj.value))
