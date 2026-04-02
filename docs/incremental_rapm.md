# Incremental RAPM: Design Notes

## Context

Rolling RAPM fits one ridge regression per date — e.g. "what are player estimates
as of 2025-03-01, using the last N days of possessions?" The naive implementation
rebuilds the normal equations from scratch at every date:

```
G_t = X_t' W_t X_t          (2P × 2P Gram matrix)
b_t = X_t' W_t y_t          (2P right-hand side)
beta_t = (G_t + λI)^{-1} b_t
```

For ~1 M possessions and ~1,500 players this is expensive enough that fitting
across a full season of dates takes minutes.

---

## Why Not Rust?

The natural question is whether rewriting the solver in Rust would help.
Short answer: **probably not dramatically**.

The current bottleneck in `solve_ridge` ([rapm.py](../deep_rapm/rapm.py)) is:

1. `Xw.T @ Xw` — sparse → dense matrix multiply
2. `np.linalg.solve` — dense linear system on the resulting `(2P × 2P)` matrix

Both already delegate to optimized BLAS/LAPACK. A Rust rewrite using `ndarray` +
`blas-src` + `lapack` calls the same kernels. The overhead reduction is real but
modest — typically 1.2–1.5× — not transformative.

Rust *would* help more if the wall time is dominated by Python-side orchestration
(preprocessing, indexing, sparse assembly, per-row weighting) rather than the
matrix ops themselves. That is worth profiling. But the fundamental distinction is:

- **Language optimization** gives constant-factor gains.
- **Algorithmic optimization** changes the scaling.

For rolling RAPM, the algorithmic path is far more valuable.

**Verdict:** Build Rust if you want a clean systems artifact or production binary.
Build the incremental solver if you want real speed. Profile first; only then
consider Rust for remaining bottlenecks.

---

## Incremental Update Approach

When the window moves forward one day, most possessions are unchanged. Instead of
recomputing G and b from scratch, maintain them incrementally:

$$
G_{t+1} = G_t + X_{\text{add}}' W_{\text{add}} X_{\text{add}} - X_{\text{drop}}' W_{\text{drop}} X_{\text{drop}}
$$

$$
b_{t+1} = b_t + X_{\text{add}}' W_{\text{add}} y_{\text{add}} - X_{\text{drop}}' W_{\text{drop}} y_{\text{drop}}
$$

Each date then pays only for the possessions entering and leaving the window, not
the full history.

### Complexity

| Approach | Cost per date |
|----------|---------------|
| Full recompute | O(n · P²) where n ≈ 1 M, P ≈ 1,500 |
| Incremental | O(k · P²) where k ≈ 1,500 possessions/day |

Rough ratio: n/k ≈ 650, so **~650× fewer floating-point operations per date**.

### Note on "rank" terminology

A single possession row has 10 nonzeros (5 offense + 5 defense), so it contributes
a rank-1 outer product of a sparse 10-hot vector to G. A full day's possessions
contribute a sum of ~1,500 such rank-1 updates — not literally a "rank-10 update"
at the day level, but the practical point holds: the update is much cheaper than
a full recompute.

---

## Further Optimization: Cholesky Rank Updates

For a sequence of nearby ridge systems `(G_t + λI) β_t = b_t`, maintaining a
Cholesky factorization and applying rank-1 updates/downdates (via `scipy.linalg.cho_rank1_update`
or similar) avoids re-factorizing the full `(2P × 2P)` matrix at each step. This
is numerically stable for moderate condition numbers and is the natural next step
after incremental G/b updates are in place.

---

## Implementation Plan

1. In `rolling_rapm.py` (or a new `deep_rapm/rolling.py`), maintain `G` and `b`
   as running state keyed by date.
2. On each date step:
   - Add rows for possessions in new games (`X_add`, `w_add`, `y_add`).
   - Subtract rows for possessions falling outside the window (`X_drop`, `w_drop`, `y_drop`).
3. Solve `(G + λI) β = b` with `np.linalg.solve` (or a maintained Cholesky factor).
4. Profile: if the solve dominates, add Cholesky update/downdate.
5. Only then consider Rust for any remaining Python-overhead bottlenecks.
