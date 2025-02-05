from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from libactor.actor._version import ActorVersion
from libactor.misc import get_classpath, param_as_dict
from libactor.typing import DataClassInstance


@dataclass
class ActorState:
    """Represent a state of an actor, including its class, versions, and parameters"""

    classpath: str
    classversion: Union[str, int, ActorVersion]
    params: Union[
        DataClassInstance, list[DataClassInstance], dict[str, DataClassInstance]
    ]
    dependencies: list[ActorState]

    @staticmethod
    def create(
        CLS: type,
        args: Union[
            DataClassInstance, list[DataClassInstance], dict[str, DataClassInstance]
        ],
        version: Optional[Union[int, str, ActorVersion]] = None,
        dependencies: Optional[list[ActorState]] = None,
    ) -> ActorState:
        """Compute a unique cache id"""
        if version is None:
            assert hasattr(CLS, "VERSION"), f"Class {CLS} must have a VERSION attribute"
            version = getattr(CLS, "VERSION")

        assert version is not None and isinstance(
            version, (int, str, ActorVersion)
        ), "Version must be a string, a number, or an ActorVersion"

        return ActorState(
            classpath=get_classpath(CLS),
            classversion=version,
            params=args,
            dependencies=dependencies or [],
        )

    def get_classname(self) -> str:
        return self.classpath.split(".")[-1]

    def to_dict(self) -> dict:
        """Return the state in dictionary form, mainly used for comparing the state"""
        return {
            "classpath": self.classpath,
            "classversion": self.classversion,
            "params": self.convert_params_to_collection(),
            "dependencies": [d.to_dict() for d in self.dependencies],
        }

    def convert_params_to_collection(self) -> list | dict:
        if isinstance(self.params, list):
            params = [param_as_dict(p) for p in self.params]
        elif isinstance(self.params, dict):
            params = {
                k: v if isinstance(v, (str, int, bool)) else param_as_dict(v)
                for k, v in self.params.items()
            }
        elif hasattr(self.params, "to_dict"):
            params = self.params.to_dict()
        else:
            params = param_as_dict(self.params)
        return params
