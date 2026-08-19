# UMAP hyperparameter/seed robustness sweep

Supplementary results for the real-vs-generated UMAP analysis in the paper
(Section "Global morphological similarity via UMAP"). The main text and
figures use a single primary configuration (`random_state=42`,
`n_neighbors=15`, `min_dist=0.6`); the files here are the additional sweep
referenced in the paper's hyperparameter table (Appendix,
"Hyperparameters for t-SNE and UMAP") and cited as evidence that the
real/synthetic overlap pattern is not an artifact of that one embedding.

## Sweep

9 combinations total: `random_state in {0, 1, 2}` x
`(n_neighbors, min_dist) in {(15, 0.6), (30, 0.3), (50, 0.1)}`, all with
`n_components=3`, `metric=correlation`, fit jointly on 2,000 real + 2,000
synthetic samples per class (14,000 + 14,000 total), same as the primary
embedding.

## File naming

Each combination is tagged `seed{S}_nn{N}_md{M}` in the filename, e.g.
`seed1_nn30_md0.3` = `random_state=1`, `n_neighbors=30`, `min_dist=0.3`.

- `embedding_seed{S}_nn{N}_md{M}.npz` — raw 3D UMAP coordinates and class
  labels for that configuration (real and synthetic samples).
- `umap_two_views_corr_2k_seed{S}_nn{N}_md{M}.pdf` — combined real+synthetic
  scatter, two viewpoints (paper-style summary plot, analogous to
  Figure "Three perspectives of 2D UMAP embeddings...").
- `umap_three_views_2d_corr_2k_seed{S}_nn{N}_md{M}.pdf` — same, three
  viewpoints.
- `umap_per_class_corr_2k_seed{S}_nn{N}_md{M}[_view2].pdf` — per-class
  real-vs-synthetic panels, 3D, one or two rotated viewpoints.
- `umap_per_class_2d_corr_2k_seed{S}_nn{N}_md{M}_v12/_v13/_v23.pdf` —
  per-class panels projected onto each pair of the three UMAP dimensions
  (dims 1-2, 1-3, 2-3 respectively).

## Takeaway

The same qualitative pattern reported for the primary configuration holds
across all 9 sweep configurations: strong real/synthetic overlap for most
classes, with Whistle showing the most consistent partial separation.
