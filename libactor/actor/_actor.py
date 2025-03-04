from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Callable, Generic, Optional, Sequence

from libactor.actor._state import ActorState
from libactor.misc import get_classpath, orjson_dumps
from libactor.storage._global_storage import GlobalStorage
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

    def forward(self, *args, **kwargs):
        # This method should be implemented by subclasses to define the forward pass logic
        raise NotImplementedError("Subclasses should implement this method")

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
        full_key = orjson_dumps(self.get_actor_state().to_dict()).decode()
        return GlobalStorage.get_instance().shorten_key(full_key)

    @cached_property
    def actor_dir(self):
        actor_dir = (
            GlobalStorage.get_instance().workdir
            / f"{self.__class__.__name__}_{self.VERSION}"
            / self.key
        )
        actor_dir.mkdir(exist_ok=True, parents=True)
        return actor_dir

    def __reduce__(self):
        if len(self.dep_actors) == 0:
            return self.__class__, (self.params,)
        return self.__class__, (self.params, self.dep_actors)


@lru_cache()
def make_key(func: Callable, version: int | str, ser_args: str):
    full_key = ":".join((get_classpath(func), str(version), ser_args))
    return GlobalStorage.get_instance().shorten_key(full_key)
