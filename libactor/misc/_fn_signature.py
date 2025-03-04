from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin, get_type_hints

from libactor.misc._misc import get_classpath


@dataclass
class FnSignature:
    return_type: type
    argnames: list[str]
    argtypes: list[type]
    default_args: dict[str, Any]  # Added this field to store default values

    @staticmethod
    def parse(func: Callable) -> FnSignature:
        sig = get_type_hints(func)
        argnames = list(sig.keys())[:-1]

        # Get the default values using inspect.signature
        inspect_sig = inspect.signature(func)
        defaults = {}
        for name, param in inspect_sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default

        try:
            return FnSignature(
                sig["return"],
                argnames,
                [sig[arg] for arg in argnames],
                defaults,  # Add the default values to the signature
            )
        except:
            print("Cannot figure out the signature of", func)
            print("The parsed signature is:", sig)
            raise


def type_to_string(_type: type) -> str:
    """Return a fully qualified type name"""
    origin = get_origin(_type)
    if origin is None:
        return get_classpath(_type)
    return (
        get_classpath(origin)
        + "["
        + ", ".join([get_classpath(arg) for arg in get_args(_type)])
        + "]"
    )
