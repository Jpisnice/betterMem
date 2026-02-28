Learning and smoothing
======================

The `bettermem.learning` package contains **probabilistic shaping** tools
and reinforcement-learning hooks for the transition model.

Modules
-------

- `smoothing.py`:
  - `additive_smoothing(counts, alpha)` (Laplace/add-\(\alpha\)) and
    `dirichlet_smoothing(counts, prior_mass, vocab_size)`.
  - These convert raw counts into smoothed distributions, e.g.
    \((\text{count} + \alpha) / (\text{total} + \alpha V)\).
- `entropy_control.py`:
  - `entropy(P) = -\sum p \log p`, used as a measure of uncertainty.
  - `temperature_rescale(P, T)` to sharpen or flatten a distribution via a
    softmax over \(\log p / T\).
- `reinforcement.py`:
  - `SimpleReinforcementLearner` maintains average rewards for triplets
    \((i, j, k)\) in second-order transitions, and returns multiplicative
    adjustment factors for successors of \((i,j)\).

How it fits into the system
---------------------------

- `TransitionModel` in `bettermem.core.transition_model` can call into
  smoothing functions when normalizing counts, so that the learned
  \(P_2(k\mid i,j)\) and \(P_1(k\mid j)\) distributions remain robust in
  sparse regimes.
- Entropy control and temperature rescaling can be used at query time to
  adjust exploration vs. exploitation in traversal distributions.
- The reinforcement learner conceptually modulates the base
  \(P(k\mid i,j)\) based on feedback from downstream answer quality,
  although integration remains minimal and pluggable.

