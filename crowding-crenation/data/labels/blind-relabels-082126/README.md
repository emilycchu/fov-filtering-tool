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
| overlap | 41/47 (87%) | 92% | 5 | 1 |

**Density 82% against v2.2's 55% exact-match means the model is not at the ceiling**, so further
feature work is justified — which was the question this set was built to answer.

The overlap denominator is 47, not 50: **the annotator reports having forgotten that `rouleaux`
(the 4th rung) was an available label during pass 2**, so on the 3 FOVs whose pass-1 label was
`rouleaux` — fov-18, fov-34, fov-47 — the instrument could not agree no matter what was on the
slide. Those are excluded rather than scored, because counting them as disagreements measures the
worksheet and not the annotator. As measured they give 41/50 (82%); assuming all 3 would have
been re-picked they give 44/50 (88%); excluding them, 41/47 (87%, 92% weighted) is the honest
figure and the one to quote. Both axes right on 35/47 (74%).

Three things the disagreements are not evenly spread over:

- **`slightly dense` is the unstable rung.** 6 of its 7 FOVs moved (5 down, 1 up), and pass 2
  used it once. It is doing the work of a boundary rather than naming a regime — five of the nine
  density disagreements are here.
- **`rouleaux` vanishing was an instrument artifact, not a finding.** All 3 moved and pass 2
  used the rung zero times, which reads exactly like a level the ordinal does not need — but the
  cause was the annotator losing track of the vocabulary mid-worksheet, not the level failing to
  describe anything. **It says nothing about whether the overlap axis needs 4 rungs or 5**, and
  the design fix belongs in the worksheet (see below).
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

Distinguishing the two: `slightly dense` was in the annotator's working vocabulary and
`rouleaux` was not — pass 2 does use `slightly dense`, on fov-18, which is *also* one of the
three FOVs where `rouleaux` went missing. Same line, one rung remembered and one forgotten. So
the `slightly dense` instability is a real judgement boundary and the `rouleaux` gap is not.

## The worksheet needs fixing before the 648

The vocabulary appears once, in a header 50 lines above where it is needed, and that was enough
to lose a level. The same worksheet shape is about to be used for the 648-FOV v3 worklist, where
a silently-missing rung would bias the labels the model is actually fitted on rather than a
ceiling estimate. `build_blind_relabels.py` should repeat the allowed tags inline — every row, or
every tenth — before that set is generated.

Not yet done: kappa, per-cell confusion, and a scoring script. The numbers above are a direct
join against the key, with the 3 FOVs above excluded by hand.
