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
Day 1/5— setup & data loading selesai