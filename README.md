# wafer-map-eda-python
My implementation of EDA on WM-811K wafer map dataset and defect pattern analysis for semiconductor yield engineering

# WM-811K Wafer Map EDA — Defect Pattern Analysis with Python

## Dataset
- Sumber: Kaggle (qingyi/wm811k-wafer-map), asli dari MIR Lab
- 811,457 wafer map, ~20% berlabel pola defect
- Sample awal 1000 baris (`random_state=42`) dipakai untuk eksplorasi first-look di Day 1. Sejak Day 2, seluruh analisis dilakukan di dataset penuh (811,457 baris raw → 809,424 baris setelah filter reliability, lihat bagian Feature Engineering)

## Kolom dan tipe datanya
waferMap         object
dieSize           float64
lotName           object
waferIndex        float64
trianTestLabel    object
failureType       object

## Status
### Day 1/5 — Setup & data loading selesai
### Day 2/5 — Cleaning & feature engineering selesai
### Day 3/5 — Group operations & statistical analysis selesai

---

## Data Quality Deep-Dive: The "Double None" Problem

*Analysis performed on the full 811,457-row dataset (`wm811k_v2.pkl`), not a sample.*

### The Misleading Observation
Running `df.isnull().sum()` on the raw dataset returns zero missing values across every column. At face value, this looks like a perfectly clean dataset.

### Why That's Wrong
For two columns — `failureType` and `trainTestLabel` (stored as `trianTestLabel` in the raw data, a typo native to the original dataset, not introduced here) — "missing" is not encoded as `NaN`. It's encoded as an **empty nested array**. Pandas' `.isnull()` has no way to detect this, so it silently reports a clean dataset that isn't clean in the way it appears.

After flattening both columns with a custom `extract_label()` function, a cross-tabulation exposes what's actually going on:

### The Real Structure

| `trainTestLabel_clean` | Center | Donut | Edge-Loc | Edge-Ring | Loc | Near-full | Random | Scratch | none | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Test | 832 | 146 | 2,772 | 1,126 | 1,973 | 95 | 257 | 693 | 110,701 | **118,595** |
| Training | 3,462 | 409 | 2,417 | 8,554 | 1,620 | 54 | 609 | 500 | 36,730 | **54,355** |
| none (never reviewed) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 638,507 | **638,507** |
| **Total** | **4,294** | **555** | **5,189** | **9,680** | **3,593** | **149** | **866** | **1,193** | **785,938** | **811,457** |

What looks like one homogeneous "none" category in `failureType` is actually **three structurally different populations**:

| Population | Count | Share |
|---|---:|---:|
| Never reviewed (unlabeled) | 638,507 | 78.7% |
| Reviewed, no defect found | 147,431 | 18.2% |
| Reviewed, with a defect pattern | 25,519 | 3.1% |

The first two groups both show up as `failureType = 'none'`, but one has simply never been inspected, while the other has been inspected and passed. Treating them as the same thing erases a meaningful distinction in the data.

### Design Decision: Sampling Strategy
This distinction is the direct basis for the `stratify_key` used in this project's sampling approach (see `01_project_setup.ipynb`). Rather than lumping every `failureType == 'none'` row into one stratum, `unlabeled` and `reviewed_no_defect` are kept as two separate strata — preserving the inspection-status information that the raw label alone throws away. In practice, the per-category cap ended up higher than any single category's size, so no category was actually downsampled — every row carries forward, and the composite key serves as stratification bookkeeping rather than a mechanism to shrink the dataset.

### Design Decision: Defect Rate Calculation
The choice of denominator matters more than it looks:

- **Naive calculation** (using all 811,457 wafers as the denominator):
  `25,519 / 811,457 = 3.14%`
- **Correct calculation** (using only the 172,950 wafers that were actually reviewed):
  `25,519 / 172,950 = 14.76%`

The naive version understates the true defect rate by nearly **5x**, because it silently treats "never inspected" wafers as if they were "inspected and clean." Any downstream analysis of defect rates in this project uses the reviewed-only denominator.

### Sanity Check
Every per-category count above (Center: 4,294, Donut: 555, Edge-Loc: 5,189, Edge-Ring: 9,680, Loc: 3,593, Near-full: 149, Random: 866, Scratch: 1,193) matches the publicly documented class distribution for WM-811K exactly, confirming the raw file was loaded without corruption or unintended modification.

---

## Feature Engineering & Pattern Validation (Day 2/5)

To transition from visual inspection to quantitative analysis, defect density features were engineered directly from the 2D `waferMap` arrays across all 811,457 rows.

### 1. Global Defect Density
Calculated as the ratio of defect pixels (value `2`) to total valid die pixels (value `1` + `2`).
- **Validation**: The `Near-full` category showed a mean density of `0.877`, while the `none` (reviewed, no defect) category showed a mean of `0.094`. The baseline for `none` is not zero because "no pattern" does not strictly mean "zero defects"—a key domain insight.

### 2. Regional Defect Density (Radial Approach)
To distinguish edge patterns from center patterns, a radial distance approach was implemented:
1. **Centroid Calculation**: Found the geometric center `(row_center, col_center)` of all valid dies for each wafer.
2. **Euclidean Distance**: Calculated the distance of every valid die to the centroid.
3. **Normalization**: Divided distances by the maximum distance to scale all wafers to a standard `0.0 - 1.0` radius.
4. **Zonal Split**: Dies with normalized distance `<= 0.5` were classified as `Center`, while `> 0.5` were classified as `Edge`.

**Validation**: 
- `Edge-Ring` showed a distinct separation: `edge_density (0.184) >> center_density (0.061)`.
- `Random` showed near-identical densities across both zones (`center: 0.490`, `edge: 0.476`), validating the approach.

**Design Limitation - The "Donut" Asymmetry**:
A geometric asymmetry was discovered. A linear threshold of `0.5` effectively assigns ~25% of the wafer's area to the "Center" zone and ~75% to the "Edge" zone (due to $Area = \pi r^2$). Because a "Donut" defect sits at a median radius, its defects fall largely into the smaller "Center" zone, making it statistically resemble a "Center" defect. This is documented as a limitation of the 2-zone binary split and would require a 3-zone approach (Inner Core, Middle Ring, Outer Edge) to resolve perfectly.

### 3. Pattern Flagging (Data Driven Thresholds)
Rather than using arbitrary round numbers, natural breakpoints in the data were identified by calculating the gaps between category means and finding the midpoint between the 3rd Quartile (Q3) of the lower group and the 1st Quartile (Q1) of the upper group.

| Flag | Threshold | Mathematical Basis | Agreement Rate |
| :--- | :--- | :--- | :--- |
| `is_edge_heavy` | `>= 0.0807` | Midpoint of Q3 `none` (0.072) and Q1 `Edge-Loc` (0.088) | **Edge-Ring: 97.7%**, Edge-Loc: 79.6% |
| `is_center_heavy` | `<= -0.1073` | Midpoint of Q3 `Donut` (-0.102) and Q1 `Near-full` (-0.111) | Donut: 73.0%, Near-full: 27.5% |
| `is_globally_dense` | `>= 0.6752` | Midpoint of Q3 `Random` (0.59) and Q1 `Near-full` (0.76) | **Near-full: 100%** |

### 4. Visual Validation
An 8-panel gallery was generated using `seaborn.heatmap` to visually confirm that the numerical flags align with the physical defect patterns on the wafer maps.

A reliability filter (`dieSize >= 80` and `fill_ratio >= 0.3`) was also applied at this stage, producing `df_reliable` — 809,424 of the 811,457 rows (2,033 wafers excluded for having too few valid dies to compute density meaningfully). All Day 3 analysis below builds on `df_reliable`, not the raw 811,457-row set, so figures in this section and the next aren't always directly comparable.

---

## Group Operations & Statistical Analysis (Day 3/5)

Building on `df_reliable` (809,424 rows), this phase applied `groupby`-based aggregation, custom split-apply-combine functions, and correlation analysis to consolidate and extend the Day 2 findings.

### 1. Consolidated Pivot Table
A single `pd.pivot_table()` (`index='failureType_clean'`, four density metrics as `values`, `aggfunc=['mean','count']`, `margins=True`) consolidated metrics that were previously scattered across several cells. Every value cross-validated exactly against its structural definition (`edge_center_diff = density_edge - density_center` held precisely for every category).

| Category | edge_center_diff | Interpretation |
|---|---:|---|
| Donut | -0.195 | Most center-heavy |
| Edge-Ring | +0.199 | Most edge-heavy |
| Near-full | -0.046 | Near-zero — uniformly dense, not lopsided to either zone |
| none | +0.046 | Baseline, near-neutral |

**dieSize interaction (new limitation found)**: A second pivot (`columns='dieSize_binned'`) showed `density_global` decreasing systematically as `dieSize` increases, holding for 6 of 9 categories with reliable sample sizes — most convincingly for `none` (n=196K–373K per bin). This means part of the density difference observed between defect categories is entangled with wafer die-grid size, not purely the defect pattern itself.

### 2. Custom Group-Wise Functions (`groupby().apply()`)
Two custom functions were applied via `.groupby(..., observed=True).apply(fn, include_groups=False)`:

- **`top_n_density`** — top-3 highest-density wafers per category, as candidates for the Day 4 visual gallery.
- **`quantile_summary` + `find_split_threshold`** — a generalized, reusable reimplementation of the manual Q1/Q3 threshold derivation from Day 2. All three original thresholds (`THRESHOLD_CENTER`, `THRESHOLD_EDGE`, `THRESHOLD_DENSE`) were reproduced to 4 decimal places exactly, confirming the manual derivation and making the method reusable for future metrics.

**Anomaly found**: the top-3 `density_global` wafers in the `none` category all showed `density_global = 1.0`. Investigation traced these to the `unlabeled` stratum (never manually reviewed) rather than `reviewed_no_defect` — consistent with the "Double None" finding above, not a data quality bug.

**Design note**: `.groupby().apply()` was deliberately avoided for the Day 1 stratified sampling step (a manual loop was used instead) because newer pandas versions exclude the grouping column from what's passed into the applied function when the function body needs it. The two functions above were designed around that limitation by never referencing the grouping column internally.

### 3. Feature Correlation Analysis
A Pearson correlation matrix was computed across 8 numeric features (`dieSize`, `fill_ratio`, `density_global`, `density_center`, `density_edge`, `edge_center_diff`, `center_row`, `center_col`); `waferIndex` (non-informative ID) and `total_cells` (exact duplicate of `dieSize`, r=1.00) were excluded.

- **`density_global` is largely redundant** with `density_center` (r=0.97) and `density_edge` (r=0.97) — it carries little independent information beyond the two regional metrics.
- **`dieSize` vs `density_global`: r=-0.23** — quantifies the confounding pattern found in the pivot table above.
- **`density_edge` vs `edge_center_diff`: r=0.54** — partially structural, since `edge_center_diff` is derived from `density_edge` by definition, not a purely empirical relationship.

**Implication for Day 4+**: given the redundancy, feature selection for visualization/modeling should likely prioritize `density_center` + `density_edge` (or `edge_center_diff`) over `density_global`, and should treat `dieSize` as a covariate when comparing density across categories.

**Output**: `03_groupby_aggregation.ipynb`