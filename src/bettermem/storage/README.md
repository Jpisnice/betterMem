# Storage

Serialization and persistence for indices.

## Modules

- **`persistence.py`**:
  - **save_index(path, graph, transition_model, config=None)**: writes `graph.joblib` (Graph with nodes/edges), `transition.joblib` (TransitionModel), and optional `config.json`.
  - **load_index(path, mmap_mode=None)**: loads graph and transition model from joblib; parses optional `config.json` into BetterMemConfig. Graph is rebuilt via `Graph.from_dict` (including topic indexes). Topic model is **not** persisted; a loaded client runs without it (query prior can fall back to uniform over topics).
- **`sparse_tensor.py`**: Optional sparse representation for transition counts; used by TransitionModel serialization if applicable.

## How it fits

- **BetterMem.save** and **BetterMem.load** delegate here. Enables offline use and sharing indices without re-indexing.
