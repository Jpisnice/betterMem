from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Tuple

from bettermem.core.nodes import NodeId

TripleIndex = Tuple[NodeId, NodeId]


@dataclass
class SparseTensor3D:
    """Minimal sparse 3D tensor view for (i, j, k) transitions.

    This is primarily a structural helper around the underlying dict-of-dicts
    representation used by TransitionModel.
    """

    data: Dict[TripleIndex, Dict[NodeId, float]] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, MutableMapping]:
        return {
            f"{i}|{j}": dict(row) for (i, j), row in self.data.items()
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Mapping[NodeId, float]]) -> "SparseTensor3D":
        data: Dict[TripleIndex, Dict[NodeId, float]] = {}
        for key, row in payload.items():
            i, j = key.split("|", 1)
            data[(i, j)] = dict(row)
        return cls(data=data)

