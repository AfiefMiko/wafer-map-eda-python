"""
src/features.py

Feature engineering pipeline (Day 12-13): dari dataframe v2 (output
data_loader.load_stratified_sample()) menjadi df_reliable (v3) --
lengkap dengan density metrics, reliability flags, dan pattern flags.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Utilitas kolom / memori
# ---------------------------------------------------------------

_KODE_SINGKAT = {
    'unlabeled': 'UNL', 'reviewed_no_defect': 'RND',
    'Center': 'C', 'Donut': 'D', 'Edge-Loc': 'EL', 'Edge-Ring': 'ER',
    'Loc': 'L', 'Near-full': 'NF', 'Random': 'R', 'Scratch': 'S',
}


def _optimize_dtypes(df):
    """Cast kolom kategorikal ke 'category', downcast dieSize & waferIndex."""
    kolom_kategorikal = ['stratify_key', 'failureType_clean', 'trainTestLabel_clean', 'lotName']
    for col in kolom_kategorikal:
        if col in df.columns:
            df[col] = df[col].astype('category')
    df['dieSize'] = pd.to_numeric(df['dieSize'], downcast='integer')
    df['waferIndex'] = pd.to_numeric(df['waferIndex'], downcast='integer')
    return df


# ---------------------------------------------------------------
# Fungsi density per wafer map
# ---------------------------------------------------------------

def calculate_density_global(wafer_map):
    """
    Density global sebuah wafer map (utilitas standalone untuk quick-check --
    pipeline utama pakai get_regional_densities() yang menghitung global
    SEKALIAN center/edge dalam satu pass, jadi tidak double-hitung).
    """
    if wafer_map is None or not isinstance(wafer_map, np.ndarray):
        return np.nan
    if wafer_map.size == 0:
        return np.nan
    count_valid = (wafer_map != 0).sum()
    if count_valid == 0:
        return np.nan
    count_defect = (wafer_map == 2).sum()
    return count_defect / count_valid


def get_wafer_centroid(wafer_map):
    """Titik pusat (centroid) die yang valid (nilai 1 dan 2). Return (row_center, col_center)."""
    if wafer_map is None or not hasattr(wafer_map, 'shape') or wafer_map.size == 0:
        return (np.nan, np.nan)
    valid_mask = wafer_map > 0
    if not np.any(valid_mask):
        return (np.nan, np.nan)
    rows, cols = np.where(valid_mask)
    return (np.mean(rows), np.mean(cols))


def get_regional_densities(wafer_map, rc=None, cc=None, threshold=np.sqrt(0.5)):
    """
    Density global, center, dan edge dari sebuah wafer map.
    threshold = sqrt(0.5) (~0.707) supaya zona center & edge seimbang
    secara LUAS (bukan jarak linear) -- luas lingkaran kuadratik
    terhadap radius (Area = pi * r^2).
    """
    if wafer_map is None or not hasattr(wafer_map, 'shape') or wafer_map.size == 0:
        return (np.nan, np.nan, np.nan)
    valid_mask = wafer_map > 0
    if not np.any(valid_mask):
        return (np.nan, np.nan, np.nan)

    rows, cols = np.where(valid_mask)
    vals = wafer_map[valid_mask]

    if rc is None or cc is None:
        rc = np.mean(rows)
        cc = np.mean(cols)

    dist = np.sqrt((rows - rc) ** 2 + (cols - cc) ** 2)
    max_dist = np.max(dist)

    if max_dist == 0:
        d = 1.0 if vals[0] == 2 else 0.0
        return (d, d, d)

    norm_dist = dist / max_dist
    center_mask = norm_dist <= threshold
    edge_mask = norm_dist > threshold

    global_density = np.sum(vals == 2) / len(vals)
    center_vals = vals[center_mask]
    center_density = np.sum(center_vals == 2) / len(center_vals) if len(center_vals) > 0 else np.nan
    edge_vals = vals[edge_mask]
    edge_density = np.sum(edge_vals == 2) / len(edge_vals) if len(edge_vals) > 0 else np.nan

    return (global_density, center_density, edge_density)


# ---------------------------------------------------------------
# Orchestrator utama
# ---------------------------------------------------------------

def engineer_features(df, output_path='../data/wm811k_v3_reliable.pkl', force_reload=False,
                       min_valid_die=80, min_fill_ratio=0.3):
    """
    Jalankan seluruh pipeline Day 12-13 di atas dataframe v2, cache
    hasilnya (df_reliable / v3) ke pickle.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force_reload:
        print(f"Cache ditemukan, load langsung dari {output_path}")
        return pd.read_pickle(output_path)

    print("Cache tidak ditemukan (atau force_reload=True) -- menjalankan pipeline Day 12-13...")
    df = df.copy()

    # 1. Optimasi dtype
    df = _optimize_dtypes(df)
    df['stratify_key_short'] = df['stratify_key'].map(_KODE_SINGKAT)

    # 2. Binning dieSize
    bins = [0, 710, 1902, np.inf]
    labels = ['kecil', 'sedang', 'besar']
    df['dieSize_binned'] = pd.cut(df['dieSize'], bins=bins, labels=labels)

    # 3. fill_ratio
    df['total_cells'] = df['waferMap'].apply(lambda x: x.shape[0] * x.shape[1])
    df['fill_ratio'] = df['dieSize'] / df['total_cells']

    # 4. Reliability flags
    df['density_global_reliable'] = df['dieSize'] >= min_valid_die
    df['density_regional_reliable'] = (df['dieSize'] >= min_valid_die) & (df['fill_ratio'] >= min_fill_ratio)

    # 5. Centroid + regional densities (list-comprehension, bukan .apply()
    #    -- jauh lebih cepat untuk 800rb+ baris)
    centroids = [get_wafer_centroid(wm) for wm in df['waferMap']]
    df['center_row'] = [c[0] for c in centroids]
    df['center_col'] = [c[1] for c in centroids]

    densities = [
        get_regional_densities(wm, rc, cc)
        for wm, rc, cc in zip(df['waferMap'], df['center_row'], df['center_col'])
    ]
    df[['density_global', 'density_center', 'density_edge']] = densities

    # 6. Filter ke baris reliable untuk analisis regional
    df_reliable = df[df['density_regional_reliable']].copy()
    df_reliable = df_reliable.dropna(subset=['density_center', 'density_edge'])
    print(f"Baris reliable: {len(df_reliable)} / {len(df)}")

    # 7. edge_center_diff
    df_reliable['edge_center_diff'] = df_reliable['density_edge'] - df_reliable['density_center']

    # 8. Threshold derivation -- titik tengah antar-gap quantile,
    #    BUKAN angka manual, biar reproducible kalau data berubah.
    q3_donut = df_reliable.loc[df_reliable['failureType_clean'] == 'Donut', 'edge_center_diff'].quantile(0.75)
    q1_nearfull = df_reliable.loc[df_reliable['failureType_clean'] == 'Near-full', 'edge_center_diff'].quantile(0.25)
    threshold_center = (q3_donut + q1_nearfull) / 2

    q3_none = df_reliable.loc[df_reliable['failureType_clean'] == 'none', 'edge_center_diff'].quantile(0.75)
    q1_edgeloc = df_reliable.loc[df_reliable['failureType_clean'] == 'Edge-Loc', 'edge_center_diff'].quantile(0.25)
    threshold_edge = (q3_none + q1_edgeloc) / 2

    q3_random_global = df_reliable.loc[df_reliable['failureType_clean'] == 'Random', 'density_global'].quantile(0.75)
    q1_nearfull_global = df_reliable.loc[df_reliable['failureType_clean'] == 'Near-full', 'density_global'].quantile(0.25)
    threshold_dense = (q3_random_global + q1_nearfull_global) / 2

    print(f"THRESHOLD_CENTER = {threshold_center:.4f}")
    print(f"THRESHOLD_EDGE   = {threshold_edge:.4f}")
    print(f"THRESHOLD_DENSE  = {threshold_dense:.4f}")

    # 9. Terapkan pattern flags
    df_reliable['is_center_heavy'] = df_reliable['edge_center_diff'] <= threshold_center
    df_reliable['is_edge_heavy'] = df_reliable['edge_center_diff'] >= threshold_edge
    df_reliable['is_globally_dense'] = df_reliable['density_global'] >= threshold_dense

    # 10. Cache ke pickle
    df_reliable.to_pickle(output_path)
    print(f"\nv3 (df_reliable) disimpan ke {output_path}")

    return df_reliable