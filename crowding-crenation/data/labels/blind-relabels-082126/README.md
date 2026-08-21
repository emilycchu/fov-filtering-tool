# blind re-labels, 2026-08-21

A second, independent pass over 50 FOVs that were already annotated, to measure how often the
annotator agrees with **themselves**. Every accuracy number this project reports is bounded by
that rate, and until this pass it had never been measured — so it was unanswerable whether
v2.2's 55% density exact-match is a mediocre model or a model at the ceiling.

`blind-relabels-annotations.txt` is the worksheet as filled in: one line per blind id, free-text
tags in the v3 vocabulary (7-level density, so it measures vocabulary drift *plus* noise). All 50
lines parse against `_v3_common.parse_tanzania_tags`.

This is annotation input, so it lives here rather than in `data/results/`. The set was built by
`scripts/combined/combined-v3/build_blind_relabels.py`, and the un-blinding key stays next to
that script's other output, in `data/results/combined-v3/blind-relabels-KEY.csv` — it is
generated, and reproducible from the seed. The 203 MB image zip is gitignored for the same
reason.

## First read

Scored after `collapse_to_v2_density` folds the two new rungs back to `sparser`, so a vocabulary
change is not counted as disagreement:

| axis | exact | weighted | off-by-one | further |
|---|---|---|---|---|
| density | 41/50 (82%) | 82% | 8 | 1 (`slightly dense` -> `sparser`) |
| overlap | 41/50 (82%) | 88% | 7 | 2 |

Both axes exactly right on 35/50. **82% density self-agreement against v2.2's 55% means the
model is not at the ceiling**, so further feature work is justified — which was the question this
set was built to answer.

Three things the disagreements are not evenly spread over:

- **`slightly dense` is the unstable rung.** 6 of its 7 FOVs moved (5 down, 1 up), and pass 2
  used it once. It is doing the work of a boundary rather than naming a regime — five of the nine
  density disagreements are here.
- **`rouleaux` (the 4th overlap level) vanished.** All 3 moved, 2 down to `some` and 1 up to
  `heavy`; pass 2 used it zero times. The overlap ordinal may really have four usable rungs.
- **The two slides are not equally noisy**: KTR-72502946 scores 30/33 density, KTR-72502948
  11/17. The draw is 33/17 in favour of the cleaner slide, so the pooled 82% is, if anything,
  optimistic.

One FOV disagrees hard rather than by a rung: **fov-02** (`tanzania-073026/dpc-203`) went
`very dense, no rouleaux` -> `dense, heavy rouleaux`. Under the overlap definition this project
settled on — any cell-cell overlap, not linear chains — a very dense field can hardly have none,
so pass 1 looks like the error. Worth eyeballing before it is averaged into anything.

Neither new rung (`no cells`, `few cells`) was used, so all 5 old `sparser` FOVs stayed
`sparser`. That is a null result for the redistribution question, not evidence against the new
levels: proportional sampling drew from a pool annotated before they existed, and only 3 of the
661 existing FOVs fire the old empty-field gate at all. The 648-FOV v3 worklist is where the
bottom rungs actually get exercised.

Not yet done: kappa, per-cell confusion, and a scoring script. The numbers above are a direct
join against the key.
