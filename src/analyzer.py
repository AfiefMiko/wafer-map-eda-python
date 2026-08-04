"""
src/analyzer.py

Utilitas ringkasan & seleksi sample dari df_reliable (Day 13):
pivot table ringkasan, interaksi dieSize, dan top-N sample per
kategori defect (bahan galeri Day 14).
"""

import pandas as pd


def top_n_density(group, n=3):
    """
    Ambil n baris dengan density_global tertinggi dari satu grup.
    Aman lewat groupby().apply() karena tidak pernah mereferensikan
    kolom grouping (failureType_clean) di dalam body fungsi.
    """
    return group.nlargest(n, 'density_global')


def get_top_n_per_category(df_reliable, n=3, cols_needed=None):
    """
    Top-n sample (berdasarkan density_global tertinggi) per
    failureType_clean. waferMap ikut disertakan supaya bisa langsung
    diplot tanpa query ulang ke df_reliable.
    """
    if cols_needed is None:
        cols_needed = [
            'lotName', 'waferIndex', 'waferMap',
            'density_global', 'density_center', 'density_edge',
            'edge_center_diff',
        ]

    top_n = df_reliable[cols_needed].groupby(
        df_reliable['failureType_clean'], observed=True
    ).apply(top_n_density, n=n, include_groups=False)

    # Validasi cepat: tiap kategori harus punya tepat n baris
    counts = top_n.index.get_level_values(0).value_counts()
    if not (counts == n).all():
        print(f"WARNING: ada kategori dengan jumlah baris != {n}:")
        print(counts[counts != n])

    return top_n


def summary_pivot(df_reliable):
    """Pivot ringkasan mean & count untuk 4 metrik density, per kategori defect."""
    return pd.pivot_table(
        df_reliable,
        index='failureType_clean',
        values=['density_global', 'density_center', 'density_edge', 'edge_center_diff'],
        aggfunc=['mean', 'count'],
        margins=True,
        margins_name='All',
    )


def size_interaction(df_reliable):
    """Rata-rata density_global per kategori defect x dieSize_binned, plus count tiap sel."""
    means = pd.pivot_table(
        df_reliable, index='failureType_clean', columns='dieSize_binned',
        values='density_global', aggfunc='mean', observed=True,
    )
    counts = pd.pivot_table(
        df_reliable, index='failureType_clean', columns='dieSize_binned',
        values='density_global', aggfunc='count', observed=True,
    )
    return means, counts