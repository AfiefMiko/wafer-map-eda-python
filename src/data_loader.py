"""
src/data_loader.py

Loader dataset WM-811K dengan stratified sampling berbasis composite key
(Day 11-12). Hasilnya di-cache ke pickle -- notebook baru tidak perlu
reload & reproses 811K baris raw tiap kali dibuka.
"""

import pickle
import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd


def _extract_label(val):
    """Ekstrak label bersih dari kolom nested array (failureType / trainTestLabel)."""
    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
        inner = val[0]
        if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
            return inner[0]
    return 'none'


def load_stratified_sample(
    raw_path='../data/LSWMD.pkl',
    output_path='../data/wm811k_v2.pkl',
    cap=638507,
    random_state=42,
    force_reload=False,
):
    """
    Load dataset WM-811K dengan stratified sampling per composite key.

    Kalau `output_path` sudah ada dan force_reload=False, langsung load
    dari cache -- instan, tidak perlu proses ulang 811K baris raw.

    Returns
    -------
    pd.DataFrame dengan kolom tambahan: failureType_clean,
    trainTestLabel_clean, stratify_key.
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_reload:
        print(f"Cache ditemukan, load langsung dari {output_path}")
        return pd.read_pickle(output_path)

    print("Cache tidak ditemukan (atau force_reload=True) -- loading raw dataset...")

    # Fix kompatibilitas pandas versi lama/baru
    sys.modules['pandas.indexes'] = pd.core.indexes

    with open(raw_path, 'rb') as f:
        df_full = pickle.load(f, encoding='latin1')
    print(f"Data berhasil di-load, bentuk data: {df_full.shape}")

    print("Extracting labels...")
    df_full['failureType_clean'] = df_full['failureType'].apply(_extract_label)
    df_full['trainTestLabel_clean'] = df_full['trianTestLabel'].apply(_extract_label)

    # Composite stratify key: pisahkan "belum pernah direview" (unlabeled)
    # vs "sudah direview, hasilnya bersih" (reviewed_no_defect), biar tidak
    # tercampur jadi satu stratum 'none' gado-gado.
    df_full['stratify_key'] = np.where(
        df_full['trainTestLabel_clean'] == 'none',
        'unlabeled',
        np.where(
            df_full['failureType_clean'] == 'none',
            'reviewed_no_defect',
            df_full['failureType_clean'],
        ),
    )
    print("\nDistribusi stratify_key di FULL dataset:")
    print(df_full['stratify_key'].value_counts())

    # Sampling per grup, loop eksplisit (bukan groupby().apply()) --
    # versi pandas terbaru drop kolom grouping di dalam .apply().
    sampled_groups = []
    for key, group in df_full.groupby('stratify_key'):
        n = min(len(group), cap)
        sampled_groups.append(group.sample(n=n, random_state=random_state))

    df = pd.concat(sampled_groups, ignore_index=True)
    print("\nDistribusi stratify_key setelah stratified sampling:")
    print(df['stratify_key'].value_counts())
    print(f"\nShape akhir: {df.shape}")

    df.to_pickle(output_path)
    print(f"\nDisimpan ke {output_path}")

    del df_full
    gc.collect()

    return df