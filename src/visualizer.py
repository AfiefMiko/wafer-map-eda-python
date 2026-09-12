"""
src/visualizer.py

Fungsi-fungsi plotting untuk EDA WM-811K (Day 14-15). Menerima
df_reliable / df beserta hasil dari analyzer.py sebagai argumen, biar
notebook cukup panggil satu fungsi per grafik tanpa menyalin ulang
puluhan baris kode plotting.

Semua fungsi menyimpan hasilnya ke ../assets/<nama>.png (dpi=150)
selain menampilkan lewat plt.show(), dan mengembalikan `fig` supaya
masih bisa dimodifikasi lebih lanjut dari notebook kalau perlu.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


ASSETS_DIR = Path('../assets')


def _save(fig, filename):
    """Simpan figure ke ASSETS_DIR, buat foldernya kalau belum ada."""
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / filename, dpi=150, bbox_inches='tight')


# Day 14
# Defect gallery
#

def plot_defect_gallery(top_n_per_category, categories=None, filename='defect_gallery.png'):
    """
    Galeri 8-panel: satu wafer (density tertinggi) per kategori defect.
    `top_n_per_category` = output dari analyzer.get_top_n_per_category().
    """
    if categories is None:
        categories = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

    plt.close('all')

    cmap = ListedColormap(["#0B0B0B", '#08DA94', '#EEFF05'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        sample = top_n_per_category.loc[cat].iloc[0]  # Opsi A: density tertinggi
        wm = sample['waferMap']

        sns.heatmap(
            wm, cmap=cmap, norm=norm, cbar=False,
            square=True, xticklabels=False, yticklabels=False, ax=axes[i],
        )

        d_center = sample['density_center']
        d_edge = sample['density_edge']
        diff = sample['edge_center_diff']
        axes[i].set_title(
            f"{cat}\nC:{d_center:.2f}  E:{d_edge:.2f}  (\u0394:{diff:+.2f})",
            fontsize=11,
        )

    legend_elements = [
        Patch(facecolor="#0B0B0B", edgecolor='gray', label='Empty'),
        Patch(facecolor="#08DA94", label='Valid Die'),
        Patch(facecolor="#EEFF05", label='Defect'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=11)

    fig.suptitle('Wafer Map Defect Gallery \u2014 8 Kategori (WM-811K)', fontsize=16, y=1.02)
    fig.tight_layout()

    _save(fig, filename)
    plt.show()
    return fig

# Size interaction

def plot_size_interaction(means, counts, min_n=30, filename='size_interaction_heatmap.png'):
    """
    Heatmap density_global rata-rata per kategori x dieSize_binned.
    Sel dengan n < min_n di-mask (mean dari sample kecil kurang reliable).
    `means`, `counts` = output dari analyzer.size_interaction().
    """
    mask = counts < min_n

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        means, mask=mask, annot=True, fmt=".3f", cmap='coolwarm',
        linewidths=.5, cbar_kws={'label': 'density_global (mean)'}, ax=ax,
    )
    ax.set_title('density_global rata-rata: Kategori Defect x Ukuran Wafer', fontsize=13, pad=12)
    ax.set_xlabel('dieSize_binned')
    ax.set_ylabel('failureType_clean')
    fig.tight_layout()

    _save(fig, filename)
    plt.show()

    # pandas 3.0: .stack() tidak lagi otomatis buang NaN (lihat catatan
    # di notebook) -- .dropna() eksplisit di sini wajib, bukan opsional.
    masked_cells = counts.where(mask).stack().dropna()
    print(f"Sel di-mask (n < {min_n}):")
    print(masked_cells.astype(int) if len(masked_cells) > 0 else "Tidak ada, semua sel sample-nya cukup.")

    return fig

# Distribusi per kategori (edge_center_diff, density_global, dll)

def plot_distribution_by_category(df_reliable, col, threshold_lines=None, filename=None):
    """
    Boxplot distribusi `col` per failureType_clean, diurutkan berdasarkan
    median (gradasi rendah -> tinggi).

    threshold_lines: dict opsional {label: (nilai, warna)} untuk garis
    threshold horizontal, misal:
        {'THRESHOLD_EDGE': (0.081, 'red')}
    """
    order = (
        df_reliable.groupby('failureType_clean', observed=True)[col]
        .median()
        .sort_values()
        .index
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=df_reliable, x='failureType_clean', y=col,
        order=order, showfliers=False, ax=ax,
    )

    if threshold_lines:
        for label, (value, color) in threshold_lines.items():
            ax.axhline(value, color=color, linestyle='--', linewidth=1, label=f'{label} ({value:.3f})')
        ax.legend()

    ax.set_title(f'Distribusi {col} per Kategori Defect', fontsize=13, pad=12)
    ax.set_xlabel('failureType_clean')
    ax.set_ylabel(col)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()

    if filename:
        _save(fig, filename)
    plt.show()
    return fig

# Defect rate (naive vs corrected denominator)

def plot_defect_rate_comparison(df, filename='defect_rate_comparison.png'):
    """
    Bar chart naive vs corrected defect rate (salah vs benar pilih
    denominator). `df` = dataframe v2 (sebelum reliability filter),
    harus punya kolom stratify_key.
    """
    n_total = len(df)
    n_with_defect = (~df['stratify_key'].isin(['unlabeled', 'reviewed_no_defect'])).sum()
    n_no_defect = (df['stratify_key'] == 'reviewed_no_defect').sum()
    n_reviewed = n_no_defect + n_with_defect

    rate_naive = n_with_defect / n_total * 100
    rate_correct = n_with_defect / n_reviewed * 100

    print(f"Reviewed with defect : {n_with_defect}")
    print(f"Naive denominator    : {n_total}     -> {rate_naive:.2f}%")
    print(f"Correct denominator  : {n_reviewed}  -> {rate_correct:.2f}%")

    fig, ax = plt.subplots(figsize=(5, 6))
    bars = ax.bar(
        ['Naive\n(/ semua wafer)', 'Correct\n(/ wafer reviewed)'],
        [rate_naive, rate_correct],
        color=['#a6a6a6', '#d62728'],
    )
    for bar, val in zip(bars, [rate_naive, rate_correct]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Defect rate (%)')
    ax.set_title('Defect Rate: Denominator Salah Meremehkan ~5x', fontsize=12, pad=12)
    ax.set_ylim(0, max(rate_naive, rate_correct) * 1.25)
    fig.tight_layout()

    _save(fig, filename)
    plt.show()
    return fig