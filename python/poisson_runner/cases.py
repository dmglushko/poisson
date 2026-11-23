"""Verification cases: a problem with a known exact solution, named and parametrised.

A `Case` only carries the case name and its parameters. Everything else (the geometry, the
materials and charges, the exact formula, the validation of the parameters) lives in C++
(`src/verification/`), so the mesh and the formula always come from the same definition.

    case = Case.of("planar_layers",
                   layers=[{"name": "oxide", "thickness": 1, "eps_r": 3.9},
                           {"name": "si", "thickness": 2, "eps_r": 11.7, "screening_length": 0.5}],
                   left=dirichlet(0), right=dirichlet(1))

The case names and their parameters are listed in `src/verification/architecture.md`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .hashing import normalize_integral_floats
from .records import register_record

__all__ = ["Case", "dirichlet", "neumann", "robin"]


def _plain(value: Any, where: str) -> Any:
    """A JSON-ready copy: dicts with string keys, lists (from lists or tuples), numbers, strings, bools, None."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: parameter names must be strings, got {k!r}")
            out[k] = _plain(v, f"{where}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(value)]
    raise TypeError(f"{where}: {type(value).__name__} is not a valid case parameter")


def _frozen(value: Any) -> Any:
    """A read-only copy of plain parameters: mappings as MappingProxyType, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def _canonical(value: Any) -> str:
    """Canonical JSON for equality and hashing: integral floats (also -0.0) as ints, bools kept apart."""
    return json.dumps(normalize_integral_floats(value), sort_keys=True, separators=(",", ":"))


@register_record("verification_case")
@dataclass(frozen=True, eq=False)
class Case:
    """A verification case: its C++ name (e.g. `"planar_layers"`) and its parameters.

    `params` is stored read-only (nested mappings and tuples); `spec()` gives the plain JSON form.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Case needs a non-empty name")
        # Read-only, so the hash cannot change after the case is put in a set or dict.
        object.__setattr__(self, "params", _frozen(_plain(self.params, "params")))

    @classmethod
    def of(cls, name: str, **params: Any) -> Case:
        return cls(name, params)

    def spec(self) -> dict[str, Any]:
        """The JSON description the C++ stages read: `{"name": ..., "params": {...}}`."""
        return {"name": self.name, "params": _plain(self.params, "params")}

    @property
    def screened(self) -> bool:
        """Whether the mesh must be imported with screening on (the `screened` parameter, default on).

        As in C++, a missing or null `screened` means on.
        """
        value = self.params.get("screened")
        return True if value is None else bool(value)

    def __eq__(self, other: object) -> bool:
        # Like the C++ stages: 1 and 1.0 are the same number, true is not 1.
        if not isinstance(other, Case):
            return NotImplemented
        return _canonical(self.spec()) == _canonical(other.spec())

    def __hash__(self) -> int:
        return hash(_canonical(self.spec()))

    def __str__(self) -> str:
        return f"{self.name}{json.dumps(self.spec()['params'], sort_keys=True, separators=(',', ':'))}"

    def to_dict(self) -> dict[str, Any]:
        """Saved form (`register_record`): plain JSON parameters."""
        return self.spec()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Case:
        return cls(data["name"], data.get("params", {}))


def dirichlet(value: float) -> dict[str, Any]:
    """A Dirichlet condition u = value [V]."""
    return {"type": "dirichlet", "value": value}


def neumann(value: float = 0.0) -> dict[str, Any]:
    """A Neumann condition eps_r du/dn = value [V/nm]."""
    return {"type": "neumann", "value": value}


def robin(alpha: float, value: float = 0.0) -> dict[str, Any]:
    """A Robin condition eps_r du/dn + alpha u = value (alpha in 1/nm, value in V/nm)."""
    return {"type": "robin", "alpha": alpha, "value": value}
