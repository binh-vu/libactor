from __future__ import annotations

from functools import cached_property
from typing import Generic, Optional, Sequence

import orjson

from libactor.actor.state import ActorState
from libactor.storage.global_storage import GlobalStorage
from libactor.typing import P


class Actor(Generic[P]):

    VERSION = 100

    def __init__(
        self,
        params: P,
        dep_actors: Optional[Sequence[Actor]] = None,
    ):
        self.params = params
        self.dep_actors: Sequence[Actor] = dep_actors or []
        self._cache_obj = {}

    def get_actor_state(self) -> ActorState:
        """Get the state of this actor"""
        deps = [actor.get_actor_state() for actor in self.dep_actors]
        return ActorState.create(
            self.__class__,
            self.params,
            dependencies=deps,
        )

    @cached_property
    def key(self):
        full_key = orjson.dumps(self.get_actor_state().to_dict()).decode()
        return GlobalStorage.get_instance().shorten_key(full_key)

    @cached_property
    def actor_dir(self):
        actor_dir = (
            GlobalStorage.get_instance().workdir
            / f"{self.__class__.__name__}_{self.VERSION}"
        )
        actor_dir.mkdir(exist_ok=True, parents=True)
        return actor_dir
