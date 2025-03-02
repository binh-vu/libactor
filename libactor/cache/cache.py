from __future__ import annotations

import inspect
import types
from functools import partial, wraps
from typing import Any, Callable, Optional, TypeVar, cast

from libactor.cache.backend import Backend
from libactor.cache.cache_args import CacheArgsHelper, get_func_type
from libactor.misc import identity, orjson_dumps
from libactor.typing import ArgSer

C = TypeVar("C", bound=Callable)


def cache(
    backend: (
        Backend
        | Callable[[Callable, CacheArgsHelper], Backend]
        | Callable[[Any, Callable, CacheArgsHelper], Backend]
    ),
    cache_args: Optional[list[str]] = None,
    cache_ser_args: Optional[dict[str, ArgSer]] = None,
    disable: bool = False,
    cache_attr: str = "__cache_backends__",
) -> Callable[[C], C]:
    """A single cache decorator that can be used for both functions and instance methods.

    Args:
        backend: a backend constructor if the cache function is applied on an instance method, otherwise
            it can be both backend or backend constructor. The backend constructor will be called once.
        cache_args: a list of arguments to be used to compute cache key. If None, all arguments will be used.
        cache_ser_args: a dictionary of argument name to serialization function to serialize the argument.
        disable: if True, the cache will be disabled.
        cache_attr: the attribute name to store the cache backend instances (only applicable when caching methods).
    """
    if disable:
        return identity

    backend_factory = backend

    def wrapper_fn(func: C) -> C:
        cache_args_helper = CacheArgsHelper.from_func(
            func, cache_ser_args=cache_ser_args
        )
        if cache_args is not None:
            cache_args_helper.keep_args(cache_args)

        cache_args_helper.ensure_auto_cache_key_friendly()

        func_type = get_func_type(func)
        if func_type == "function":
            # we are caching a function
            keyfn = lambda *args, **kwargs: orjson_dumps(
                cache_args_helper.get_func_args(*args, **kwargs)
            ).decode()

            if callable(backend_factory):
                store = cast(
                    Callable[[Callable, CacheArgsHelper], Backend], backend_factory
                )(func, cache_args_helper)
            else:
                assert isinstance(backend_factory, Backend)
                store = backend_factory

            @wraps(func)
            def fn(*args, **kwargs):
                key = keyfn(*args, **kwargs)
                if store.has_key(key):
                    return store.get(key)

                val = func(*args, **kwargs)
                store.set(key, val)

                return val

            return fn  # type: ignore
        else:
            assert func_type == "instancemethod"

            keyfn = lambda self, *args, **kwargs: orjson_dumps(
                cache_args_helper.get_method_args(self, *args, **kwargs)
            ).decode()

            assert callable(backend_factory) and not isinstance(
                backend_factory, Backend
            ), f"Backend factory must be a constructor function to create one for each instance. Got {backend_factory}"

            @wraps(func)
            def method(self, *args, **kwargs):
                if cache_attr not in self.__dict__:
                    self.__dict__[cache_attr] = {}
                assert not (
                    func.__name__.startswith("__") and func.__name__.endswith("__")
                ), "Current implementation does not support caching magic methods as we cannot replace them with a new method."
                assert func.__name__ not in self.__dict__[cache_attr], (
                    self,
                    cache_attr,
                    func.__name__,
                    self.__dict__[cache_attr],
                )
                self.__dict__[cache_attr][func.__name__] = cast(
                    Callable[[Any, Callable, CacheArgsHelper], Backend], backend_factory
                )(self, func, cache_args_helper)

                def update_method(self, *args, **kwargs):
                    store = self.__dict__[cache_attr][func.__name__]
                    key = keyfn(self, *args, **kwargs)
                    if store.has_key(key):
                        return store.get(key)

                    val = func(self, *args, **kwargs)
                    store.set(key, val)
                    return val

                setattr(self, func.__name__, types.MethodType(update_method, self))
                return update_method(self, *args, **kwargs)

            return method  # type: ignore

    return wrapper_fn
