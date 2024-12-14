from __future__ import annotations

import pytest

from libactor.cache import BackendFactory, IdentObj, MemBackend, cache


def test_cache_func():
    backend = MemBackend()

    @cache(backend=backend)
    def square_sum(arr: IdentObj[list[int]]):
        return sum([x**2 for x in arr.value])

    # first time they call, the value will be cached
    arr = IdentObj(key="abc", value=[1, 2, 3])
    assert square_sum(arr) == 14

    # next call with the same argument (key), the value will be returned from cache
    arr.value = [4, 5, 6]
    assert square_sum(arr) == 14

    # next call with the different argument (key), the value will be recalculated
    arr.key = "def"
    assert square_sum(arr) == 77

    # clear the cache and the value will be recalculated
    arr.value = [1, 2, 3]
    assert square_sum(arr) == 77
    backend.clear()
    assert square_sum(arr) == 14


def test_cache_method():
    class Actor:
        def __init__(self, coeff: int):
            self.coeff = coeff

        @cache(backend=BackendFactory.actor.mem)
        def square_sum(self, arr: IdentObj[list[int]]):
            return self.coeff * sum([x**2 for x in arr.value])

    actor1 = Actor(2)
    actor2 = Actor(3)

    # first time they call, the value will be cached
    arr = IdentObj(key="abc", value=[1, 2, 3])
    assert actor1.square_sum(arr) == 28

    # next call with the same argument (key), the value will be returned from cache
    arr.value = [4, 5, 6]
    assert actor1.square_sum(arr) == 28

    # cache between actors are independent
    assert actor2.square_sum(arr) == 231

    # next call with the different argument (key), the value will be recalculated
    arr.key = "def"
    assert actor1.square_sum(arr) == 154


def test_cache_magic_method():
    class Actor:
        def __init__(self, coeff: int):
            self.coeff = coeff

        @cache(backend=BackendFactory.actor.mem)
        def __call__(self, arr: IdentObj[list[int]]):
            return self.coeff * sum([x**2 for x in arr.value])

    actor = Actor(2)
    # first time they call, an error will be thrown to tell users to use proper method name
    with pytest.raises(AssertionError):
        actor(IdentObj(key="abc", value=[1, 2, 3]))
