from __future__ import annotations

from graph.interface import BaseEdge, BaseNode
from graph.retworkx.digraph import RetworkXDiGraph


@dataclass
class ActorNode: ...


@dataclass
class ActorEdge: ...


class DAG:
    graph: RetworkXDiGraph[int, ActorNode, ActorEdge]
