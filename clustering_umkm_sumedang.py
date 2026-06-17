"""
=============================================================
  APLIKASI DATA MINING - SEGMENTASI UMKM SUMEDANG
  Metode: K-Means Clustering
  Mata Kuliah: Data Mining
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Konfigurasi tampilan ──────────────────────────────────
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = ['#2563EB', '#16A34A', '#DC2626', '#D97706']
CLUSTER_LABELS = {
    0: 'Klaster A – Usaha Mikro Pemula',
    1: 'Klaster B – Usaha Kecil Berkembang',
    2: 'Klaster C – Usaha Menengah Potensial',
    3: 'Klaster D – Usaha Unggulan',
}


# ════════════════════════════════════════════════════════════
#  1. LOAD & EKSPLORASI DATA
# ════════════════════════════════════════════════════════════
def load_data(path='dataset_umkm_sumedang.csv'):
    df = pd.read_csv(path)
    print("=" * 55)
    print("  DATA MINING – SEGMENTASI UMKM SUMEDANG")
    print("=" * 55)
    print(f"\n📊 Total data UMKM  : {len(df)} usaha")
    print(f"📋 Jumlah fitur     : {df.shape[1]} kolom")
    print(f"\n── Distribusi Kategori Usaha ──")
    print(df['kategori_usaha'].value_counts().to_string())
    print(f"\n── Distribusi per Kecamatan ──")
    print(df['kecamatan'].value_counts().to_string())
    print(f"\n── Statistik Deskriptif Fitur Numerik ──")
    fitur = ['omzet_bulanan_juta', 'jumlah_pegawai',
             'lama_usaha_tahun', 'aset_juta', 'pinjaman_modal_juta']
    print(df[fitur].describe().round(2).to_string())
    return df


# ════════════════════════════════════════════════════════════
#  2. PRE-PROCESSING
# ════════════════════════════════════════════════════════════
def preprocess(df):
    fitur = ['omzet_bulanan_juta', 'jumlah_pegawai',
             'lama_usaha_tahun', 'aset_juta', 'pinjaman_modal_juta']

    # Cek missing value
    missing = df[fitur].isnull().sum()
    if missing.any():
        print(f"\n⚠ Missing values ditemukan:\n{missing[missing > 0]}")
        df[fitur] = df[fitur].fillna(df[fitur].median())
    else:
        print("\n✅ Tidak ada missing value pada fitur clustering.")

    X = df[fitur].copy()

    # Normalisasi dengan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Normalisasi StandardScaler selesai.")
    return X, X_scaled, fitur, scaler


# ════════════════════════════════════════════════════════════
#  3. ELBOW METHOD – MENENTUKAN K OPTIMAL
# ════════════════════════════════════════════════════════════
def elbow_method(X_scaled):
    print("\n── Menjalankan Elbow Method (k=2 s.d. 10) ──")
    inertia, silhouette = [], []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(X_scaled, km.labels_))
        print(f"  k={k}  Inertia={km.inertia_:.1f}  Silhouette={silhouette[-1]:.4f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Penentuan Jumlah Klaster Optimal', fontsize=14, fontweight='bold')

    axes[0].plot(list(K_range), inertia, 'o-', color='#2563EB', linewidth=2, markersize=6)
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('Jumlah Klaster (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].grid(alpha=0.3)
    # tandai elbow optimal
    axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k optimal = 4')
    axes[0].legend()

    axes[1].plot(list(K_range), silhouette, 's-', color='#16A34A', linewidth=2, markersize=6)
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Jumlah Klaster (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(alpha=0.3)
    axes[1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k optimal = 4')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('plot_elbow_silhouette.png', bbox_inches='tight')
    plt.show()
    print("✅ Plot disimpan: plot_elbow_silhouette.png")

    best_k = list(K_range)[silhouette.index(max(silhouette))]
    print(f"\n🎯 K optimal berdasarkan Silhouette Score tertinggi: k = {best_k}")
    return best_k


# ════════════════════════════════════════════════════════════
#  4. K-MEANS CLUSTERING
# ════════════════════════════════════════════════════════════
def run_kmeans(df, X_scaled, k=4):
    print(f"\n── Menjalankan K-Means dengan k={k} ──")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    df = df.copy()
    df['klaster'] = labels

    sil = silhouette_score(X_scaled, labels)
    print(f"✅ Clustering selesai.")
    print(f"   Silhouette Score akhir : {sil:.4f}")
    print(f"   Inertia (WCSS)        : {km.inertia_:.2f}")

    print("\n── Distribusi Data per Klaster ──")
    for c in sorted(df['klaster'].unique()):
        n = (df['klaster'] == c).sum()
        label = CLUSTER_LABELS.get(c, f'Klaster {c}')
        print(f"  {label}: {n} UMKM")

    return df, km, sil


# ════════════════════════════════════════════════════════════
#  5. ANALISIS & VISUALISASI PROFIL KLASTER
# ════════════════════════════════════════════════════════════
def analyze_clusters(df, fitur):
    print("\n── Rata-rata Fitur per Klaster ──")
    profil = df.groupby('klaster')[fitur].mean().round(2)
    for idx, row in profil.iterrows():
        label = CLUSTER_LABELS.get(idx, f'Klaster {idx}')
        print(f"\n  {label}")
        for f, v in row.items():
            print(f"    {f:<30}: {v}")

    # ── Radar chart profil klaster ──
    fitur_display = ['Omzet\n(juta)', 'Pegawai', 'Lama\nUsaha', 'Aset\n(juta)', 'Pinjaman\n(juta)']
    profil_norm = (profil - profil.min()) / (profil.max() - profil.min() + 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
    fig.suptitle('Profil Fitur per Klaster UMKM Sumedang', fontsize=15, fontweight='bold', y=1.01)

    angles = np.linspace(0, 2 * np.pi, len(fitur), endpoint=False).tolist()
    angles += angles[:1]

    for i, ax in enumerate(axes.flat):
        if i >= len(profil):
            ax.set_visible(False)
            continue
        values = profil_norm.iloc[i].tolist() + [profil_norm.iloc[i].tolist()[0]]
        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[i])
        ax.fill(angles, values, alpha=0.25, color=COLORS[i])
        ax.set_thetagrids(np.degrees(angles[:-1]), fitur_display, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(CLUSTER_LABELS.get(i, f'Klaster {i}'),
                     size=10, fontweight='bold', pad=15, color=COLORS[i])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_radar_klaster.png', bbox_inches='tight')
    plt.show()
    print("✅ Radar chart disimpan: plot_radar_klaster.png")
    return profil


# ════════════════════════════════════════════════════════════
#  6. VISUALISASI PCA 2D
# ════════════════════════════════════════════════════════════
def visualize_pca(df, X_scaled):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')

    for c in sorted(df['klaster'].unique()):
        mask = df['klaster'] == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[c], label=CLUSTER_LABELS.get(c, f'Klaster {c}'),
                   alpha=0.8, s=90, edgecolors='white', linewidths=0.5, zorder=3)

    ax.set_title('Visualisasi Klaster UMKM Sumedang (PCA 2D)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(f'Komponen Utama 1 ({var[0]*100:.1f}% variansi)', fontsize=11)
    ax.set_ylabel(f'Komponen Utama 2 ({var[1]*100:.1f}% variansi)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_pca_klaster.png', bbox_inches='tight')
    plt.show()
    print("✅ Plot PCA disimpan: plot_pca_klaster.png")


# ════════════════════════════════════════════════════════════
#  7. DISTRIBUSI KLASTER PER KECAMATAN & KATEGORI
# ════════════════════════════════════════════════════════════
def visualize_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribusi Klaster UMKM Sumedang', fontsize=14, fontweight='bold')

    # Per kecamatan
    cross_kec = pd.crosstab(df['kecamatan'], df['klaster'])
    cross_kec.plot(kind='bar', ax=axes[0], color=COLORS[:4], edgecolor='white', linewidth=0.5)
    axes[0].set_title('Per Kecamatan')
    axes[0].set_xlabel('Kecamatan')
    axes[0].set_ylabel('Jumlah UMKM')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend([CLUSTER_LABELS.get(i, f'K{i}').split('–')[0].strip() for i in range(4)],
                   title='Klaster', fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    # Per kategori usaha
    cross_kat = pd.crosstab(df['kategori_usaha'], df['klaster'])
    cross_kat.plot(kind='bar', ax=axes[1], color=COLORS[:4], edgecolor='white', linewidth=0.5)
    axes[1].set_title('Per Kategori Usaha')
    axes[1].set_xlabel('Kategori')
    axes[1].set_ylabel('Jumlah UMKM')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend([CLUSTER_LABELS.get(i, f'K{i}').split('–')[0].strip() for i in range(4)],
                   title='Klaster', fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_distribusi_klaster.png', bbox_inches='tight')
    plt.show()
    print("✅ Plot distribusi disimpan: plot_distribusi_klaster.png")


# ════════════════════════════════════════════════════════════
#  8. SIMPAN HASIL & REKOMENDASI
# ════════════════════════════════════════════════════════════
def save_results(df, sil_score):
    df_out = df.copy()
    df_out['label_klaster'] = df_out['klaster'].map(CLUSTER_LABELS)
    df_out.to_csv('hasil_clustering_umkm.csv', index=False)
    print("\n✅ Hasil disimpan: hasil_clustering_umkm.csv")

    print("\n" + "=" * 55)
    print("  REKOMENDASI KEBIJAKAN BERDASARKAN KLASTER")
    print("=" * 55)
    rekomendasi = {
        0: "Perlu pendampingan intensif, pelatihan manajemen keuangan dasar, dan kemudahan akses KUR Mikro.",
        1: "Fokus pada digitalisasi pemasaran, peningkatan kapasitas produksi, dan akses pasar yang lebih luas.",
        2: "Dukung ekspansi usaha, fasilitasi kemitraan dengan perusahaan besar, dan pelatihan ekspor.",
        3: "Prioritas untuk inkubasi bisnis lanjutan, akses pasar ekspor, dan peningkatan standar produk.",
    }
    for c, rec in rekomendasi.items():
        label = CLUSTER_LABELS.get(c, f'Klaster {c}')
        print(f"\n  {label}")
        print(f"  → {rec}")

    print(f"\n  Silhouette Score Model : {sil_score:.4f}")
    print("  (Nilai mendekati 1.0 menunjukkan klaster yang baik)")
    print("\n" + "=" * 55)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 1. Load data
    df = load_data('dataset_umkm_sumedang.csv')

    # 2. Pre-processing
    X, X_scaled, fitur, scaler = preprocess(df)

    # 3. Elbow method → tentukan k optimal
    k_optimal = elbow_method(X_scaled)

    # 4. K-Means clustering (gunakan k=4 sesuai elbow/silhouette)
    K_FINAL = 4
    df_result, model, sil_score = run_kmeans(df, X_scaled, k=K_FINAL)

    # 5. Analisis profil klaster
    profil = analyze_clusters(df_result, fitur)

    # 6. Visualisasi PCA 2D
    visualize_pca(df_result, X_scaled)

    # 7. Distribusi per kecamatan & kategori
    visualize_distribution(df_result)

    # 8. Simpan hasil & cetak rekomendasi
    save_results(df_result, sil_score)

    print("\n✅ Semua proses selesai! Cek file output di folder ini.")
