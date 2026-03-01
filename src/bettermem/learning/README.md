# Learning

Utilities used for **smoothing and shaping** of transition distributions. The main query path uses the transition model only when blending with the policy (configurable); the transition model itself uses smoothing when normalizing counts.

## Modules

- **`smoothing.py`**: Additive (Laplace) and Dirichlet-style smoothing for count-based distributions. Used by **TransitionModel** in `bettermem.core.transition_model` when building P(k|i,j) and P(k|j) from topic sequences.
- **`entropy_control.py`**: Entropy and temperature rescaling of distributions; available for experimentation or future use (e.g. controlling exploration at query time).
- **`reinforcement.py`**: Optional reinforcement learner for reweighting transitions from feedback; integration with the traversal is minimal and pluggable.

## How it fits

- **TransitionModel** imports and uses `bettermem.learning.smoothing` when fitting and normalizing second- and first-order counts. The rest of the learning package is optional/experimental.
