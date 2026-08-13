# nigeria-081226 labels

8 FOVs across 2 slides (HP231668, HP245487 R), hand-annotated on the same two axes as the
other label sets in `data/labels/`: `density` (5 levels, `sparser` kept distinct from
`monolayer`) and `overlap` (5 levels, displayed as "Rouleaux").

Two FOVs carry a `_sparse` suffix in their filenames. **That suffix is not parsed anywhere.**
It is the only label material the raw dataset shipped with, and this CSV exists so the
annotation is recorded data rather than an inference from a filename — the same reason
`tanzania-073026` has an annotated CSV instead of the pipeline reading slide tags at runtime.

Column layout matches `initial-dataset-071626/fovs.csv` (`filename,…,density,overlap`) so
`merge_labels_v2.py::load_initial_dataset()` can consume it unchanged when this set is pooled
into a future calibration run.

| density | n | overlap | n |
|---|---|---|---|
| very dense | 6 | heavy rouleaux | 6 |
| sparser | 2 | no rouleaux | 2 |

Against these labels, v2.2 scores 6/8 correct on both axes; with the empty-field gate
(`empty_field_override` in `density_overlap_v2.2_params.json`) it is 8/8. See
`data/results/nigeria-081226/README.md`.
