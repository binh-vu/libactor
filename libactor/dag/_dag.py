from __future__ import annotations

import enum
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Optional,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
)

from graph.interface import BaseEdge, BaseNode
from graph.retworkx import RetworkXStrDiGraph, topological_sort
from libactor.actor._actor import Actor
from libactor.cache import IdentObj
from libactor.misc import (
    ComposeTypeConversion,
    FnSignature,
    TypeConversion,
    UnitTypeConversion,
    align_generic_type,
    ground_generic_type,
    identity,
    is_generic_type,
)


class Cardinality(enum.Enum):
    ONE_TO_ONE = 1
    ONE_TO_MANY = 2


ComputeFnId = Annotated[str, "ComputeFn Identifier"]
ComputeFn = Actor | Callable


class Flow:
    def __init__(
        self,
        source: list[ComputeFnId] | ComputeFnId,
        target: ComputeFn,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        is_optional: bool = False,
    ):
        self.source = [source] if isinstance(source, str) else source
        self.cardinality = cardinality
        self.target = target
        self.is_optional = is_optional

        if (self.cardinality == Cardinality.ONE_TO_MANY) and len(self.source) != 1:
            raise Exception("Can't have multiple sources for ONE_TO_MANY")

        if self.cardinality == Cardinality.ONE_TO_ONE and self.is_optional:
            raise Exception("Can't have optional ONE_TO_ONE flow")


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
            return FnSignature.parse(actor.forward)
        else:
            return FnSignature.parse(actor)

    def invoke(self, args: Sequence, context: Sequence):
        norm_args = (self.type_conversions[i](a) for i, a in enumerate(args))
        return self.get_func()(*norm_args, *context)

    def get_func(self):
        if isinstance(self.actor, Actor):
            return self.actor.forward
        else:
            return self.actor


class ActorEdge(BaseEdge[str, int]):

    def __init__(
        self,
        id: int,
        source: str,
        target: str,
        argindex: int,
        cardinality: Cardinality,
        is_optional: bool,
        type_conversion: UnitTypeConversion,
    ):
        super().__init__(id, source, target, argindex)
        self.argindex = argindex
        self.cardinality = cardinality
        self.is_optional = is_optional
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
        dictmap: dict[
            ComputeFnId,
            Flow
            | ComputeFn
            | Sequence[Flow | ComputeFn | tuple[ComputeFnId, ComputeFn | Flow]],
        ],
        type_conversions: Optional[
            Sequence[UnitTypeConversion | ComposeTypeConversion]
        ] = None,
        strict: bool = True,
    ):
        """Create a DAG from a dictionary mapping.

        Args:
            dictmap: A dictionary mapping identifier to:
                1. an actor
                2. a flow specifying the upstream actors and the actor.
                3. a linear sequence (pipeline) of flows and actors. If a sequence is provided, the output of an actor will be the input
                    of the next actor. The identifier of each actor in the pipeline will be generated automatically (Flow | ComputeFn) unless is provided
                    in the tuple[ComputeFnId, ComputeFn | Flow]
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

        # normalize dictmap to remove pipeline
        # to remove the pipeline, we need to rewire the start actor and end actor
        # because the other actor think the pipeline as a single actor
        assert (
            "" not in dictmap
        ), "Empty key is not allowed as it's a reserved key for placeholder in pipeline"
        norm_dictmap: dict[ComputeFnId, Flow | ComputeFn] = {}
        pipeline_idmap: dict[ComputeFnId, ComputeFnId] = {}
        for uid, flow in dictmap.items():
            if isinstance(flow, Sequence):
                pipe_ids = []
                for uof_i, uof_tup in enumerate(flow):
                    if isinstance(uof_tup, tuple):
                        uof_id, uof = uof_tup
                    else:
                        if uof_i > 0:
                            uof_id = f"{uid}:{uof_i}"
                        else:
                            uof_id = uid
                        uof = uof_tup

                    if isinstance(uof, Flow):
                        if any(s == "" for s in uof.source) and uof_i == 0:
                            raise ValueError(
                                "Trying to use the input of the previous object in the pipeline at the start of the pipeline"
                            )
                        new_uof = Flow(
                            source=[
                                s if s != "" else pipe_ids[uof_i - 1]
                                for s in uof.source
                            ],
                            target=uof.target,
                            cardinality=uof.cardinality,
                            is_optional=uof.is_optional,
                        )
                    else:
                        new_uof = Flow(
                            [] if uof_i == 0 else [pipe_ids[uof_i - 1]], target=uof
                        )
                    norm_dictmap[uof_id] = new_uof
                    pipe_ids.append(uof_id)
                pipeline_idmap[uid] = pipe_ids[len(flow) - 1]
            else:
                norm_dictmap[uid] = flow
        for uid, flow in dictmap.items():
            if isinstance(flow, Flow):
                flow.source = [pipeline_idmap.get(s, s) for s in flow.source]

        # create a graph
        for uid, uinfo in norm_dictmap.items():
            if isinstance(uinfo, Flow):
                actor = uinfo.target
            else:
                actor = uinfo
            g.add_node(ActorNode(uid, actor))

        # grounding function that has generic type input and output
        for uid, flow in norm_dictmap.items():
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
                        if flow.cardinality == Cardinality.ONE_TO_MANY:
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

        for uid, flow in norm_dictmap.items():
            if not isinstance(flow, Flow):
                continue

            u = g.get_node(uid)
            usig = u.signature
            if flow.cardinality == Cardinality.ONE_TO_MANY:
                # check if the return type is a generic type with a sequence as its origin (S[T])
                s = g.get_node(flow.source[0])
                ssig = s.signature

                ssig_return_origin = get_origin(tp=ssig.return_type)
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
                        is_optional=flow.is_optional,
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
                            is_optional=flow.is_optional,
                            type_conversion=cast_fn,
                        )
                    )

            u.required_args = usig.argnames[: len(flow.source)]
            # arguments of a compute function that are not provided by the upstream actors must be provided by the context.
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
        input: dict[ComputeFnId, tuple],
        output: set[str],
        context: dict[str, Callable | Any],
    ) -> dict[str, list]:
        context = {k: v() if callable(v) else v for k, v in context.items()}
        actor2context = {}
        actor2args: dict[ComputeFnId, list | deque[tuple]] = {}
        actor2incard: dict[ComputeFnId, tuple[Cardinality, bool]] = {}

        for u in self.graph.iter_nodes():
            if u.id in input:
                # user provided input should supersede the context
                n_provided_args = len(input[u.id])
                n_consumed_context = n_provided_args - len(u.required_args)
            else:
                n_consumed_context = 0

            actor2context[u.id] = tuple(
                context[name] for name in u.required_context[n_consumed_context:]
            )
            inedges = self.graph.in_edges(u.id)
            if len(inedges) > 0 and inedges[0].cardinality == Cardinality.ONE_TO_MANY:
                actor2args[u.id] = deque()
                actor2incard[u.id] = (Cardinality.ONE_TO_MANY, inedges[0].is_optional)
            else:
                actor2args[u.id] = [None] * len(u.required_args)
                actor2incard[u.id] = (Cardinality.ONE_TO_ONE, False)

        stack: list[ComputeFnId] = []
        capture_output: dict[ComputeFnId, Any] = defaultdict(list)

        for uid, args in sorted(
            input.items(),
            key=lambda x: self.graph.get_node(x[0]).topo_index,
            reverse=True,
        ):
            u = self.graph.get_node(uid)
            result = u.invoke(args, actor2context[uid])
            if (
                len(u.sorted_outedges) == 1
                and u.sorted_outedges[0].cardinality != Cardinality.ONE_TO_ONE
            ):
                if u.sorted_outedges[0].cardinality == Cardinality.ONE_TO_MANY:
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
            if actor2incard[uid][0] == Cardinality.ONE_TO_MANY:
                u_lst_args: deque[tuple] = actor2args[uid]  # type: ignore
                if len(u_lst_args) == 0:
                    if not actor2incard[uid][1]:  # is optional
                        raise RuntimeError(
                            f"Actor `{uid}` requires some data but the upstream actor doesn't produce any"
                        )
                    stack.pop()
                    continue

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
            if uid in output:
                capture_output[uid].append(result)

            # propagate the result to the downstream actors
            if (
                len(u.sorted_outedges) > 0
                and u.sorted_outedges[0].cardinality == Cardinality.ONE_TO_MANY
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
