# Annotator self-agreement: the label-noise ceiling

Every accuracy number this project reports is bounded by how often the annotator agrees with
themselves, and until 2026-08-21 that bound had never been measured. v2.2's headline is **55%
density exact-match** under slide-grouped CV; whether that is a mediocre model or a model already
at the ceiling was unanswerable, and the answer decides whether more feature work is worth doing
at all.

It is worth doing. **Density self-agreement is 82%**, so v2.2 leaves ~27 points on the table.

- Set built by `build_blind_relabels.py`; protocol and sampling rationale are in its docstring.
- Filled-in worksheet: `data/labels/blind-relabels-082126/blind-relabels-annotations.txt`.
- Un-blinding key: `data/results/combined-v3/blind-relabels-KEY.csv`.
- Numbers below are a direct join of those two files, not a committed scorer. Reproducing them is
  still a to-do; see [Open work](#open-work).

## Method

50 already-labelled FOVs, drawn from the 646 annotated under the 5-level vocabulary, re-presented
as `fov-01.png` .. `fov-50.png` under a seeded shuffle with both slides interleaved. Anonymous
filenames, shuffled order, and a key held outside the zip -- so this measures independent
judgement rather than recall of the first pass, which is what a second trip through the annotation
tool would have measured.

Two adjustments before scoring, both of which change what the number means:

1. **Collapse to v2.2's vocabulary.** The worksheet ships the 7-level v3 density ordinal, so a raw
   comparison would score the vocabulary change as annotator disagreement.
   `_v3_common.collapse_to_v2_density` folds `no cells` / `few cells` back to `sparser`. Nearly
   lossless: only 3 of the 661 existing FOVs fire v2.2's empty-field gate.
2. **Exclude 3 FOVs from the overlap axis.** The annotator reports having lost track of `rouleaux`
   (the 4th rung) being an available tag partway through pass 2, so on the 3 FOVs whose pass-1
   label was `rouleaux` the second pass could not have agreed whatever was on the slide. Scoring
   them as disagreements would measure the worksheet, not the annotator. See
   [The rouleaux gap](#the-rouleaux-gap-is-not-a-finding).

## Headline

| axis | n | exact | weighted | off-by-one | further | kappa (unw / lin / quad) |
|---|---|---|---|---|---|---|
| density | 50 | 41 (82%) | 82% | 8 | 1 | 0.686 / 0.804 / 0.895 |
| overlap | 47 | 41 (87%) | 92% | 4 | 2 | 0.751 / 0.804 / 0.833 |

Both axes right on 35/47 (74%). Weighted columns use the key's `sampling_weight` (cell frequency /
cell sampling rate) to undo the floor-of-2 that guaranteed every bucket an appearance; the
unweighted columns are the per-bucket view.

The high quadratic kappas against modest unweighted ones say the disagreements are mostly
adjacent-rung: 12 of the 15 are one level. The three that are not are fov-27 on density
(`slightly dense` -> `sparser`, -2), fov-45 on overlap (`some` -> `no rouleaux`, -2) and fov-02 on
overlap (`no` -> `heavy rouleaux`, +4), the last discussed below.

## The pooled rate is the wrong summary

Density agreement is not spread across the ordinal. It is one bad rung:

| pass-1 level | n | reproduced | rate |
|---|---|---|---|
| `sparser` | 5 | 5 | 100% |
| `monolayer` | 30 | 29 | 97% |
| `slightly dense` | 7 | 1 | **14%** |
| `dense` | 4 | 3 | 75% |
| `very dense` | 4 | 3 | 75% |

`slightly dense` lost 6 of its 7 FOVs -- 4 to `monolayer`, 1 to `dense`, 1 two rungs down to
`sparser` -- and pass 2 used the rung once in 50 FOVs. It is behaving like a boundary between
`monolayer` and `dense` rather than a regime anyone can name twice.

**This is the finding with consequences.** 30 of the 50 FOVs are `monolayer`, so the pooled 82% is
mostly a measurement of how reliably a monolayer is recognised (97%). Comparing a model's pooled
accuracy against a pooled ceiling therefore hides that the headroom is enormous at the ends and
that `slightly dense` is close to unscoreable -- a model cannot be asked to hit a target the
annotator reproduces 1 time in 7. Per-level ceilings, not the pooled figure, are what v3's
evaluation should quote.

Overlap is better behaved, with the middle rungs mildly soft:

| pass-1 level | n | reproduced | rate |
|---|---|---|---|
| `no rouleaux` | 31 | 29 | 94% |
| `slight rouleaux` | 7 | 5 | 71% |
| `some rouleaux` | 5 | 3 | 60% |
| `heavy rouleaux` | 4 | 4 | 100% |
| `rouleaux` | 3 | -- | excluded |

## The rouleaux gap is not a finding

All 3 `rouleaux` FOVs moved (2 down to `some`, 1 up to `heavy`) and pass 2 used the rung zero
times. That reads exactly like a level the ordinal does not need, and it was initially recorded as
one. It is not: the cause was the annotator losing track of the vocabulary mid-worksheet, so it
**says nothing about whether the overlap axis needs 4 rungs or 5.**

The treatment barely matters -- as measured 41/50 (82%), excluding the 3 41/47 (87%), counting
them as agreements 44/50 (88%) -- so there is no reason to re-annotate them. Scoring them as
misses was the only wrong option.

What separates this from `slightly dense` is direct: pass 2 *does* use `slightly dense`, on
fov-18, which is also one of the three FOVs where `rouleaux` went missing. One rung remembered and
one forgotten on the same line. So the `slightly dense` instability is a real judgement boundary
and this gap is an instrument defect.

**The instrument needs fixing before the 648-FOV v3 worklist reuses it.** The worksheet states its
vocabulary once, in a header 50 lines above where it is used. On a set 13x larger a silently
missing rung would bias the labels the model is *fitted* on rather than one ceiling estimate.
`build_blind_relabels.py` should repeat the allowed tags inline -- every row, or every tenth.

## The two slides are not equally noisy

| slide | dataset | n | density | overlap |
|---|---|---|---|---|
| KTR-72502946 | tanzania-080526 | 33 | 30 (91%) | 29/30 (97%) |
| KTR-72502948 | tanzania-073026 | 17 | 11 (65%) | 12/17 (71%) |

Proportional allocation drew 33/17 in favour of the *cleaner* slide, so **the pooled figures are
optimistic**, and a ceiling quoted per-slide would be 65-91% on density rather than a single 82%.
Two slides is too few to say why they differ; KTR-72502948 is the denser and more rouleaux-heavy of
the pair, which is consistent with the per-level table but not established by it.

## One hard disagreement

fov-02 (`tanzania-073026/dpc-203`) went `very dense, no rouleaux` -> `dense, heavy rouleaux`. The
density half of that is an ordinary one-rung slip; the overlap half spans the entire axis, 4 rungs
end to end, and is the largest single disagreement in the set. Under the overlap
definition this project settled on -- any cell-cell overlap, not linear chains -- a very dense
field can hardly have none, so **pass 1 looks like the error**. It is one FOV and it is left
unresolved, but it should be re-checked before it is averaged into anything, and it is a reason to
audit `very dense` + `no rouleaux` in the wider pool.

## Null result on the new bottom rungs

Neither `no cells` nor `few cells` was used, so all 5 old `sparser` FOVs stayed `sparser`. This is
**not** evidence against the two new levels: proportional sampling drew from a pool annotated
before they existed, and only 3 of the 661 existing FOVs fire the old empty-field gate at all, so
the set almost certainly contains nothing a `no cells` label would fit. The redistribution question
-- where the `no cells` / `few cells` / `sparser` boundaries actually fall -- is still open, and
the 648-FOV worklist is where the bottom rungs get exercised.

## What this means for v3

1. **Feature work is justified.** 55% measured against an 82% pooled ceiling, and against a 97%
   ceiling on the majority class, is not a model that has run out of signal.
2. **Quote per-level ceilings, not the pooled rate.** `evaluate_v3.py` comparing pooled accuracy to
   82% would be comparing against a number dominated by the easiest rung.
3. **Treat `slightly dense` as suspect.** At 1/7 self-agreement it is either a rung to merge or one
   to re-specify with an explicit criterion. Deciding that on n=7 would be overreach -- the 648-FOV
   worklist gives it a real sample, and it should be checked there first.
4. **Fix the worksheet before generating the worklist**, per above.

## Caveats

n=50, one annotator, two slides, one sitting -- this is a usable bound and not a precise one. The
95% binomial interval on 41/50 is roughly 69-91%, so "82%" should not be read to two digits. It
measures *intra*-annotator agreement, which bounds this project's labels but says nothing about
whether a second annotator would draw the boundaries in the same places; both slides here were
labelled by the same person who re-labelled them.

## Open work

- A committed scorer. These numbers came from a hand-join against the key; the 3-FOV exclusion in
  particular should be a documented list in code rather than a step in a shell one-liner.
- Per-cell confusion beyond the adjacent-rung counts above, once n justifies it.
- Inter-annotator agreement, if a second annotator is ever available. Unblocked but unscheduled.
