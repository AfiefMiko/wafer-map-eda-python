# wafer-map-eda-python
My implementation of EDA on WM-811K wafer map dataset and defect pattern analysis for semiconductor yield engineering

# WM-811K Wafer Map EDA — Defect Pattern Analysis with Python

## Dataset
- Sumber: Kaggle (qingyi/wm811k-wafer-map), asli dari MIR Lab
- 811,457 wafer map, ~20% berlabel pola defect
- Sample kerja: 1000 baris (random_state=42)

## Kolom dan tipe datanya
waferMap         object
dieSize           float64
lotName           object
waferIndex        float64
trianTestLabel    object
failureType       object

## Status
### Day 1/5 — Setup & data loading selesai
### Day 2/5 — Cleaning selesai
### Day 12/13 — Feature Engineering & Pattern Validation selesai

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
This distinction is the direct basis for the `stratify_key` used in this project's sampling approach (see `day12_stratified_sampling.py`). Rather than lumping every `failureType == 'none'` row into one stratum, `unlabeled` and `reviewed_no_defect` are kept as two separate strata — preserving the inspection-status information that the raw label alone throws away.

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

## Feature Engineering & Pattern Validation (Day 12)

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

### 3. Pattern Flagging (Data-Driven Thresholds)
Rather than using arbitrary round numbers, natural breakpoints in the data were identified by calculating the gaps between category means and finding the midpoint between the 3rd Quartile (Q3) of the lower group and the 1st Quartile (Q1) of the upper group.

| Flag | Threshold | Mathematical Basis | Agreement Rate |
| :--- | :--- | :--- | :--- |
| `is_edge_heavy` | `>= 0.0807` | Midpoint of Q3 `none` (0.072) and Q1 `Edge-Loc` (0.088) | **Edge-Ring: 97.7%**, Edge-Loc: 79.6% |
| `is_center_heavy` | `<= -0.1073` | Midpoint of Q3 `Donut` (-0.102) and Q1 `Near-full` (-0.111) | Donut: 73.0%, Near-full: 27.5% |
| `is_globally_dense` | `>= 0.6752` | Midpoint of Q3 `Random` (0.59) and Q1 `Near-full` (0.76) | **Near-full: 100%** |

### 4. Visual Validation
An 8-panel gallery was generated using `seaborn.heatmap` to visually confirm that the numerical flags align with the physical defect patterns on the wafer maps. 

*(See `defect_gallery_validation.png` in the `/results` directory for the visual proof).*