from __future__ import annotations

from libactor.cache import IdentObj, is_ident_obj_cls


def test_is_ident_obj_cls():
    assert is_ident_obj_cls(IdentObj)
    assert is_ident_obj_cls(IdentObj[int])
    assert is_ident_obj_cls(IdentObj[list[str]])

    assert not is_ident_obj_cls(int)
    assert not is_ident_obj_cls(list[int])
