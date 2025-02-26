from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, get_args, get_origin, get_type_hints

from libactor.misc._misc import get_classpath


@dataclass
class FnSignature:
    return_type: type
    argnames: list[str]
    argtypes: list[type]

    @staticmethod
    def parse(func: Callable) -> FnSignature:
        sig = get_type_hints(func)
        argnames = list(sig.keys())[:-1]
        try:
            return FnSignature(
                sig["return"],
                argnames,
                [sig[arg] for arg in argnames],
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
