from typing import Optional

from libactor.misc._fn_signature import FnSignature


# Test functions with various signatures
def function_no_defaults(a: int, b: str) -> bool:
    return True


def function_with_defaults(a: int, b: str = "default", c: float = 3.14) -> bool:
    return True


def function_mixed_defaults(
    a: int, b: str, c: float = 3.14, d: Optional[list[int]] = None
) -> dict[str, int]:
    return {"result": 42}


def test_parse_no_defaults():
    signature = FnSignature.parse(function_no_defaults)

    assert signature.return_type == bool
    assert signature.argnames == ["a", "b"]
    assert signature.argtypes == [int, str]
    assert signature.default_args == {}


def test_parse_with_defaults():
    signature = FnSignature.parse(function_with_defaults)

    assert signature.return_type == bool
    assert signature.argnames == ["a", "b", "c"]
    assert signature.argtypes == [int, str, float]
    assert signature.default_args == {"b": "default", "c": 3.14}


def test_parse_mixed_defaults():
    signature = FnSignature.parse(function_mixed_defaults)

    assert signature.return_type == dict[str, int]
    assert signature.argnames == ["a", "b", "c", "d"]
    assert signature.argtypes == [int, str, float, Optional[list[int]]]
    assert signature.default_args == {"c": 3.14, "d": None}
