from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from loguru import logger

from libactor.cache.identitied_object import get_ident_obj_key, is_ident_obj_cls
from libactor.typing import ActorFnTrait, ArgSer, NoneType


class CacheArgsHelper:
    """Helper to working with arguments of a function. This class ensures
    that we can select a subset of arguments to use for the cache key, and
    to always put the calling arguments in the same declared order.
    """

    def __init__(
        self,
        args: dict[str, Parameter],
        argtypes: dict[str, Optional[Type]],
        self_args: Optional[Callable[..., dict]] = None,
        cache_ser_args: Optional[dict[str, ArgSer]] = None,
    ):
        self.args = args
        self.argtypes = argtypes
        self.argnames: list[str] = list(self.args.keys())
        self.cache_args = self.argnames
        self.cache_ser_args: dict[str, ArgSer] = cache_ser_args or {}
        self.cache_self_args = self_args or None

        # add cache ser args that serialize IdentObj
        for arg, argtype in argtypes.items():
            if argtype is not None and is_ident_obj_cls(argtype):
                assert (
                    arg not in self.cache_ser_args
                ), f"{arg} has a default serialization. You should not attempt to set it yourself"
                self.cache_ser_args[arg] = get_ident_obj_key

    @staticmethod
    def from_func(
        func: Callable,
        self_args: Optional[Callable[..., dict]] = None,
        cache_ser_args: Optional[dict[str, ArgSer]] = None,
    ):
        args: dict[str, Parameter] = {}
        try:
            argtypes: dict[str, Optional[Type]] = get_type_hints(func)
            if "return" in argtypes:
                argtypes.pop("return")
        except TypeError:
            logger.error(
                "Cannot get type hints for function {}. "
                "If this is due to eval function, it's mean that the type is incorrect (i.e., incorrect Python's code). "
                "For example, we have a hugedict.prelude.RocksDBDict class, which is a class built from Rust (Python's extension module), "
                "the class is not a generic class, but we have a .pyi file that declare it as a generic class (cheating). This works fine"
                "for pylance and mypy checker, but it will cause error when we try to get type hints because the class is not subscriptable.",
                func,
            )
            raise
        for name, param in signature(func).parameters.items():
            args[name] = param
            if name not in argtypes:
                argtypes[name] = None

        if next(iter(args)) == "self":
            args.pop("self")

        return CacheArgsHelper(args, argtypes, self_args, cache_ser_args)

    def keep_args(self, names: Iterable[str]) -> None:
        self.cache_args = list(names)

    def get_cache_argtypes(self) -> dict[str, Optional[Type]]:
        return {name: self.argtypes[name] for name in self.cache_args}

    def ensure_auto_cache_key_friendly(self):
        for name in self.cache_args:
            param = self.args[name]
            if (
                param.kind == Parameter.VAR_KEYWORD
                or param.kind == Parameter.VAR_POSITIONAL
            ):
                raise TypeError(
                    f"Variable arguments are not supported for automatically generating caching key to cache function call. Found one with name: {name}"
                )

            if name in self.cache_ser_args:
                # the users provide a function to serialize the argument manually, so we trust the user.
                continue

            argtype = self.argtypes[name]
            if argtype is None:
                raise TypeError(
                    f"Automatically generating caching key to cache a function call requires all arguments to be annotated. Found one without annotation: {name}"
                )
            origin = get_origin(argtype)
            if origin is None:
                if (
                    not issubclass(argtype, (str, int, bool, Path))
                    and not is_ident_obj_cls(argtype)
                    and argtype is not NoneType
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, Path, or None. Found {name} with type {argtype}"
                    )
            elif origin is Union:
                args = get_args(argtype)
                if any(
                    a is not NoneType
                    and get_origin(a) is not Literal
                    and not issubclass(a, (str, int, bool, Path))
                    for a in args
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, Path, or None. Found {name} with type {argtype}"
                    )
            elif origin is Literal:
                args = get_args(argtype)
                if any(
                    not isinstance(a, (str, int, bool)) and a is not NoneType
                    for a in args
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, None, or Literal with values of those types. Found {name} with type {argtype}"
                    )
            elif not is_ident_obj_cls(origin):
                raise TypeError(
                    f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, or IdentObj or LazyIdentObj None. Found {name} with type {argtype}"
                )

    def get_func_args(self, *args, **kwargs) -> dict:
        # TODO: improve me!
        out = {name: value for name, value in zip(self.args, args)}
        out.update(
            [
                (name, kwargs.get(name, self.args[name].default))
                for name in self.argnames[len(args) :]
            ]
        )
        if len(self.cache_args) != len(self.argnames):
            out = {name: out[name] for name in self.cache_args}

        for name, ser_fn in self.cache_ser_args.items():
            out[name] = ser_fn(out[name])
        return out

    def get_method_args(self, obj: ActorFnTrait, *args, **kwargs) -> dict:
        # TODO: improve me!
        out = {name: value for name, value in zip(self.args, args)}
        out.update(
            [
                (name, kwargs.get(name, self.args[name].default))
                for name in self.argnames[len(args) :]
            ]
        )
        if len(self.cache_args) != len(self.argnames):
            out = {name: out[name] for name in self.cache_args}

        if self.cache_self_args is not None:
            out.update(self.cache_self_args(obj))

        for name, ser_fn in self.cache_ser_args.items():
            out[name] = ser_fn(out[name])
        return out

    def get_args_as_tuple(self, obj: ActorFnTrait, *args, **kwargs) -> tuple:
        # TODO: improve me!
        return tuple(self.get_method_args(obj, *args, **kwargs).values())

    @staticmethod
    def gen_cache_self_args(
        *attrs: Union[str, Callable[[Any], Union[str, bool, int, None]]]
    ):
        """Generate a function that returns a dictionary of arguments extracted from self.

        Args:
            *attrs: a list of attributes of self to be extracted.
                - If an attribute is a string, it is property of self, and the value is obtained by getattr(self, attr).
                - If an attribute is a callable, it is a no-argument method of self, and the value is obtained by
                  attr(self). To specify a method of self in the decorator, just use `method_a` instead of `Class.method_a`,
                  and the method must be defined before the decorator is called.
        """
        props = [attr for attr in attrs if isinstance(attr, str)]
        funcs = [attr for attr in attrs if callable(attr)]

        def get_self_args(self):
            args = {name: getattr(self, name) for name in props}
            args.update({func.__name__: func(self) for func in funcs})
            return args

        return get_self_args

    def get_string_template_func(self, template: str):
        def interpolate(obj, *args, **kwargs):
            out = {name: value for name, value in zip(self.args, args)}
            out.update(
                [
                    (name, kwargs.get(name, self.args[name].default))
                    for name in self.argnames[len(args) :]
                ]
            )
            return template.format(**out)

        return interpolate


def get_func_type(
    func: Callable,
) -> Literal["instancemethod", "function", "classmethod"]:
    args: dict[str, Parameter] = {}
    try:
        argtypes: dict[str, Optional[Type]] = get_type_hints(func)
        if "return" in argtypes:
            argtypes.pop("return")
    except TypeError:
        logger.error(
            "Cannot get type hints for function {}. "
            "If this is due to eval function, it's mean that the type is incorrect (i.e., incorrect Python's code). "
            "For example, we have a hugedict.prelude.RocksDBDict class, which is a class built from Rust (Python's extension module), "
            "the class is not a generic class, but we have a .pyi file that declare it as a generic class (cheating). This works fine"
            "for pylance and mypy checker, but it will cause error when we try to get type hints because the class is not subscriptable.",
            func,
        )
        raise
    for name, param in signature(func).parameters.items():
        args[name] = param
        if name not in argtypes:
            argtypes[name] = None

    first_argname = next(iter(args))
    if first_argname == "self":
        return "instancemethod"
    if first_argname == "cls":
        return "classmethod"
    return "function"
