from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from libactor.cache.backend import Backend
from libactor.cache.cache_args import CacheArgsHelper
from libactor.cache.identitied_object import is_ident_obj
from libactor.misc import identity, orjson_dumps


def cache_actor_call(
    backend: Backend,
    key_tracker: dict[str, str],
    cache_args: Optional[list[str]] = None,
    disable: bool | Callable[[Any], bool] = False,
):
    if isinstance(disable, bool) and disable:
        return identity

    def wrapper_fn(func: Callable):
        cache_args_helper = CacheArgsHelper.from_actor_func(func)
        if cache_args is not None:
            cache_args_helper.keep_args(cache_args)

        cache_args_helper.ensure_auto_cache_key_friendly()
        keyfn = lambda self, *args, **kwargs: orjson_dumps(
            cache_args_helper.get_args(self, *args, **kwargs)
        ).decode()

        backend.postinit(func, cache_args_helper)

        @wraps(func)
        def fn(self, *args, **kwargs):
            if not isinstance(disable, bool) and disable(self):
                return func(self, *args, **kwargs)

            key = keyfn(self, *args, **kwargs)
            if backend.has_key(key):
                return backend.get(key)

            val = func(self, *args, **kwargs)
            if is_ident_obj(val):
                key_tracker[val.key] = key
            backend.set(key, val)

            return val

        return fn

    return wrapper_fn
