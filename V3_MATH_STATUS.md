# Leesin V3 mathematical validation — candidate stage

This patch deliberately **does not change the V2 production output**.
It adds a parallel mathematical validation module for V3 candidates.

## What is implemented

- Exact V2-compatible right-eCDF Specificity.
- Mid-rank eCDF candidate for ties.
- Exact finite-grid one-row replacement sensitivity:

  \[
  \Delta_1^\Gamma
  =
  \max_{c'\in\mathcal N_1^\Gamma(c)}
  |S_\Gamma(c',t)-S_\Gamma(c,t)|.
  \]

- A display-direction value `1 - delta`, explicitly labelled as a stability
  transform rather than a probability or established Sample Sufficiency.
- Raw Coverage.
- Pielou and Simpson/order-2 effective-evenness candidates.
- An iid multinomial expected total-variation upper-bound candidate:

  \[
  E[TV(\hat P,P)]\le \frac12\sqrt{\frac{A-1}{N}}.
  \]

- Brute-force and reduced exact algorithms, cross-checked in randomized tests.
- A reproducible counterexample showing that the current rank-based eCDF's
  one-row sensitivity does **not** shrink when all counts are scaled up.

## Why V2 is not changed yet

For a uniform four-bin reference

\[
(m,m,m,m),
\]

moving one row from the target bin to another bin changes current right-eCDF
Specificity by `0.75`, regardless of whether `m=10` or `m=10000`.  Mid-rank
handling reduces the jump to `0.375`, but it still does not vanish.

Therefore exact one-row sensitivity is already a useful and mathematically
correct perturbation certificate, but `Z = 1 - delta_1` has **not yet passed**
the requirements for a Sample Sufficiency score.  The validation layer must
be used to compare Specificity and Z candidates before app/UI integration.

## Apply

Copy these paths into the repository root:

- `v3_metrics.py`
- `tests/test_v3_metrics.py`
- `experiments/__init__.py`
- `experiments/v3_math_validation.py`

Then run:

```bash
python -m unittest tests.test_v3_metrics -v
python -m experiments.v3_math_validation
```

The existing V2 test suite should remain unchanged and continue to pass.
