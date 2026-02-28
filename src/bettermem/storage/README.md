Storage and persistence
=======================

The `bettermem.storage` package provides **serialization** utilities for
indices and sparse transition data.

Modules
-------

- `sparse_tensor.py`:
  - Defines `SparseTensor3D`, a minimal wrapper for the dict-of-dicts
    representation used for second-order transitions:
    \(\mathcal{T}_{ijk}\) stored as `data[(i,j)][k] = value`.
  - Provides `to_dict` / `from_dict` helpers for JSON-friendly encoding.
- `persistence.py`:
  - `save_index(path, graph, transition_model, config=None)` writes:
    - `graph.json`: serialized `Graph` with nodes and edges.
    - `transition.json`: serialized `TransitionModel` counts/probabilities.
    - `config.json`: `BetterMemConfig` settings (optional).
  - `load_index(path)` reconstructs:
    - `Graph` via `Graph.from_dict`.
    - `TransitionModel` via `TransitionModel.from_dict`.
    - Optional `BetterMemConfig`.

How it fits into the system
---------------------------

- The `BetterMem.save` and `BetterMem.load` API methods delegate to this
  package to persist and restore indices.
- Keeping graph structure, transition statistics, and configuration
  versioned and decoupled allows BetterMem to:
  - Run offline without re-indexing.
  - Share indices across environments or processes.

