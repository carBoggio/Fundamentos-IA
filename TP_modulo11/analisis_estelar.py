"""
TP Bloque 11: Aprendizaje No Supervisado
Dataset: Datos estelares GAIA (vía astroquery, con fallback sintético)

Cubre:
  1. Carga de datos estelares GAIA
  2. Preprocesamiento y exploración
  3. PCA: reducción de dimensionalidad y visualización
  4. K-Means: método del codo y coeficiente de silueta
  5. Clustering final e interpretación astronómica
  6. DBSCAN como alternativa
  7. Isolation Forest: detección de anomalías estelares
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS GAIA
# ─────────────────────────────────────────────

print("=" * 62)
print("  BLOQUE 11 — Aprendizaje No Supervisado: Datos GAIA")
print("=" * 62)

def cargar_gaia():
    """Intenta bajar datos reales de GAIA. Si falla, genera sintéticos."""
    try:
        from astroquery.gaia import Gaia
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        query = """
            SELECT TOP 4000
                bp_rp, phot_g_mean_mag, parallax, pmra, pmdec
            FROM gaiadr3.gaia_source
            WHERE parallax > 5
              AND parallax < 100
              AND phot_g_mean_mag < 15
              AND bp_rp IS NOT NULL
              AND pmra IS NOT NULL
        """
        print("\n  Conectando con el archivo GAIA...")
        job    = Gaia.launch_job(query)
        tabla  = job.get_results().to_pandas()
        tabla  = tabla.dropna()
        # Magnitud absoluta: M = m - 5*log10(1000/parallax) + 5
        tabla["abs_mag"] = tabla["phot_g_mean_mag"] - 5 * np.log10(1000 / tabla["parallax"]) + 5
        print(f"  Datos GAIA reales descargados: {len(tabla)} estrellas")
        return tabla[["bp_rp", "abs_mag", "parallax", "pmra", "pmdec"]], "real"
    except Exception as e:
        print(f"\n  GAIA no disponible ({type(e).__name__}). Usando datos sintéticos.")
        return generar_datos_sinteticos(), "sintetico"

def generar_datos_sinteticos():
    """
    Genera estrellas sintéticas que replican las poblaciones del
    diagrama de Hertzsprung-Russell:
      - Secuencia principal (la mayoría de las estrellas)
      - Gigantes rojas
      - Enanas blancas
      - Supergigantes azules
      - Outliers / anomalías
    """
    rng = np.random.default_rng(42)
    grupos = []

    # Secuencia principal: franja diagonal del HR
    n = 2500
    color = rng.uniform(-0.3, 2.0, n)
    mag   = 4.5 * color + rng.normal(0, 0.4, n)
    pmra  = rng.normal(0, 15, n)
    pmdec = rng.normal(0, 15, n)
    par   = rng.uniform(5, 80, n)
    grupos.append(pd.DataFrame({"bp_rp": color, "abs_mag": mag,
                                "parallax": par, "pmra": pmra, "pmdec": pmdec}))

    # Gigantes rojas: esquina superior derecha
    n = 700
    color = rng.uniform(1.2, 3.5, n)
    mag   = rng.uniform(-3, 1.5, n)
    pmra  = rng.normal(-5, 8, n)
    pmdec = rng.normal(-5, 8, n)
    par   = rng.uniform(5, 40, n)
    grupos.append(pd.DataFrame({"bp_rp": color, "abs_mag": mag,
                                "parallax": par, "pmra": pmra, "pmdec": pmdec}))

    # Enanas blancas: esquina inferior izquierda
    n = 400
    color = rng.uniform(-0.4, 0.8, n)
    mag   = rng.uniform(10, 16, n)
    pmra  = rng.normal(0, 30, n)
    pmdec = rng.normal(0, 30, n)
    par   = rng.uniform(20, 100, n)
    grupos.append(pd.DataFrame({"bp_rp": color, "abs_mag": mag,
                                "parallax": par, "pmra": pmra, "pmdec": pmdec}))

    # Supergigantes azules: esquina superior izquierda
    n = 150
    color = rng.uniform(-0.4, 0.3, n)
    mag   = rng.uniform(-8, -2, n)
    pmra  = rng.normal(0, 5, n)
    pmdec = rng.normal(0, 5, n)
    par   = rng.uniform(5, 15, n)
    grupos.append(pd.DataFrame({"bp_rp": color, "abs_mag": mag,
                                "parallax": par, "pmra": pmra, "pmdec": pmdec}))

    # Anomalías: puntos que no encajan en ninguna población
    n = 80
    color = rng.uniform(-1.0, 5.0, n)
    mag   = rng.uniform(-10, 20, n)
    pmra  = rng.normal(0, 80, n)
    pmdec = rng.normal(0, 80, n)
    par   = rng.uniform(5, 100, n)
    grupos.append(pd.DataFrame({"bp_rp": color, "abs_mag": mag,
                                "parallax": par, "pmra": pmra, "pmdec": pmdec}))

    df = pd.concat(grupos, ignore_index=True)
    print(f"  Dataset sintético generado: {len(df)} estrellas")
    return df

df, origen = cargar_gaia()

print(f"\n  Origen datos  : {origen}")
print(f"  Total estrellas: {len(df)}")
print(f"\n  Estadísticas descriptivas:")
print(df.describe().round(3).to_string())

# ─────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  2. PREPROCESAMIENTO")
print("─" * 62)

df = df.dropna()
features = ["bp_rp", "abs_mag", "parallax", "pmra", "pmdec"]
X = df[features].values

# Recortar outliers extremos antes de escalar
for i in range(X.shape[1]):
    p1, p99 = np.percentile(X[:, i], [1, 99])
    X[:, i] = np.clip(X[:, i], p1, p99)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Features      : {features}")
print(f"  Escalado      : StandardScaler (media=0, std=1)")
print(f"  Estrellas     : {len(X)} tras limpieza")

# ─────────────────────────────────────────────
# 3. PCA
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  3. PCA — Reducción de Dimensionalidad")
print("─" * 62)

pca_full = PCA()
pca_full.fit(X_scaled)
var_exp   = pca_full.explained_variance_ratio_
var_acum  = np.cumsum(var_exp)

print(f"\n  Varianza explicada por componente:")
for i, (v, va) in enumerate(zip(var_exp, var_acum), 1):
    barra = "#" * int(v * 50)
    print(f"    PC{i}: {v:.3f} ({va:.3f} acumulado)  {barra}")

n_comp = np.argmax(var_acum >= 0.90) + 1
print(f"\n  Componentes para explicar 90% de varianza: {n_comp}")

pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
print(f"  Proyección 2D explica: {pca2.explained_variance_ratio_.sum():.1%} de la varianza")

# ─────────────────────────────────────────────
# 4. K-MEANS: MÉTODO DEL CODO Y SILUETA
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  4. K-MEANS — Selección de K")
print("─" * 62)

inercias   = []
siluetas   = []
rango_k    = range(2, 10)

print(f"\n  {'K':<5} {'Inercia':>12} {'Silueta':>10}")
print("  " + "-" * 30)

for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=1000)
    siluetas.append(sil)
    print(f"  {k:<5} {km.inertia_:>12.1f} {sil:>10.4f}")

k_optimo = list(rango_k)[np.argmax(siluetas)]
print(f"\n  K óptimo por silueta: {k_optimo}")

# ─────────────────────────────────────────────
# 5. CLUSTERING FINAL E INTERPRETACIÓN
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  5. CLUSTERING FINAL")
print("─" * 62)

km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
labels_km = km_final.fit_predict(X_scaled)

print(f"\n  Clusters encontrados: {k_optimo}")
print(f"\n  {'Cluster':<10} {'N estrellas':>12} {'bp_rp medio':>14} {'mag_abs media':>15} {'parallax media':>16}")
print("  " + "-" * 70)

nombres_astro = {
    0: "Secuencia principal",
    1: "Gigantes rojas",
    2: "Enanas blancas",
    3: "Supergigantes",
    4: "Mixto/transición",
}

for c in range(k_optimo):
    mask  = labels_km == c
    n     = mask.sum()
    color = df["bp_rp"].values[mask].mean()
    mag   = df["abs_mag"].values[mask].mean()
    par   = df["parallax"].values[mask].mean()
    print(f"  {c:<10} {n:>12} {color:>14.3f} {mag:>15.3f} {par:>16.3f}")

print("""
  Interpretación astronómica:
    bp_rp alto (>1.5) + mag_abs alta  → Gigantes rojas (frías y grandes)
    bp_rp bajo (<0.3) + mag_abs baja  → Supergigantes azules (muy luminosas)
    bp_rp bajo        + mag_abs alta  → Enanas blancas (pequeñas y densas)
    franja diagonal                   → Secuencia principal (como el Sol)
""")

# ─────────────────────────────────────────────
# 6. DBSCAN
# ─────────────────────────────────────────────

print("─" * 62)
print("  6. DBSCAN — Clustering por densidad")
print("─" * 62)

db = DBSCAN(eps=0.5, min_samples=15)
labels_db = db.fit_predict(X_scaled)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_ruido       = (labels_db == -1).sum()

print(f"\n  Clusters encontrados : {n_clusters_db}")
print(f"  Puntos marcados ruido: {n_ruido} ({n_ruido/len(labels_db):.1%})")
print(f"\n  Diferencia clave vs K-Means:")
print(f"    K-Means asigna cada punto a un cluster, sin excepciones")
print(f"    DBSCAN marca {n_ruido} estrellas como ruido — candidatas a anomalías")

# ─────────────────────────────────────────────
# 7. ISOLATION FOREST — DETECCIÓN DE ANOMALÍAS
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  7. ISOLATION FOREST — Anomalías estelares")
print("─" * 62)

iso = IsolationForest(contamination=0.03, random_state=42)
anomalias = iso.fit_predict(X_scaled)
n_anomalias = (anomalias == -1).sum()

print(f"\n  Estrellas anómalas detectadas: {n_anomalias} ({n_anomalias/len(X):.1%})")
print(f"\n  Propiedades promedio de las anomalías:")
mask_ano = anomalias == -1
for feat in features:
    val = df[feat].values[mask_ano].mean()
    print(f"    {feat:<12}: {val:.3f}")

print(f"\n  Interpretación: estas estrellas tienen combinaciones inusuales de")
print(f"  color, magnitud y movimiento propio. Pueden ser objetos exóticos,")
print(f"  errores de medición, o estrellas en fases evolutivas raras.")

# ─────────────────────────────────────────────
# 8. GRÁFICOS
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  8. GRÁFICOS")
print("─" * 62)

colores_cluster = plt.cm.tab10(np.linspace(0, 0.9, k_optimo))
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Diagrama HR coloreado por cluster
ax = axes[0, 0]
for c in range(k_optimo):
    mask = labels_km == c
    ax.scatter(df["bp_rp"].values[mask], df["abs_mag"].values[mask],
               s=3, alpha=0.4, color=colores_cluster[c], label=f"Cluster {c}")
ax.invert_yaxis()
ax.set_xlabel("Color (bp_rp)")
ax.set_ylabel("Magnitud absoluta")
ax.set_title("Diagrama HR — K-Means")
ax.legend(markerscale=4, fontsize=7)
ax.grid(True, alpha=0.2)

# 2. Proyección PCA
ax = axes[0, 1]
for c in range(k_optimo):
    mask = labels_km == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=3, alpha=0.4, color=colores_cluster[c], label=f"Cluster {c}")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} varianza)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} varianza)")
ax.set_title("Proyección PCA — K-Means")
ax.legend(markerscale=4, fontsize=7)
ax.grid(True, alpha=0.2)

# 3. Método del codo
ax = axes[0, 2]
ax.plot(list(rango_k), inercias, "o-", color="steelblue", lw=2)
ax.axvline(k_optimo, color="tomato", linestyle="--", label=f"K óptimo = {k_optimo}")
ax.set_xlabel("Número de clusters K")
ax.set_ylabel("Inercia (WCSS)")
ax.set_title("Método del Codo")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Coeficiente de silueta
ax = axes[1, 0]
ax.plot(list(rango_k), siluetas, "o-", color="seagreen", lw=2)
ax.axvline(k_optimo, color="tomato", linestyle="--", label=f"K óptimo = {k_optimo}")
ax.set_xlabel("Número de clusters K")
ax.set_ylabel("Coeficiente de silueta")
ax.set_title("Silueta por K")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. DBSCAN en HR
ax = axes[1, 1]
mask_ruido = labels_db == -1
ax.scatter(df["bp_rp"].values[~mask_ruido], df["abs_mag"].values[~mask_ruido],
           c=labels_db[~mask_ruido], cmap="tab10", s=3, alpha=0.4, label="Clusters")
ax.scatter(df["bp_rp"].values[mask_ruido], df["abs_mag"].values[mask_ruido],
           color="black", s=6, alpha=0.6, label=f"Ruido ({n_ruido})")
ax.invert_yaxis()
ax.set_xlabel("Color (bp_rp)")
ax.set_ylabel("Magnitud absoluta")
ax.set_title(f"Diagrama HR — DBSCAN ({n_clusters_db} clusters)")
ax.legend(markerscale=3, fontsize=7)
ax.grid(True, alpha=0.2)

# 6. Anomalías en HR
ax = axes[1, 2]
mask_norm = anomalias == 1
ax.scatter(df["bp_rp"].values[mask_norm], df["abs_mag"].values[mask_norm],
           color="steelblue", s=3, alpha=0.3, label=f"Normales ({mask_norm.sum()})")
ax.scatter(df["bp_rp"].values[mask_ano], df["abs_mag"].values[mask_ano],
           color="tomato", s=15, alpha=0.8, label=f"Anomalías ({n_anomalias})", zorder=5)
ax.invert_yaxis()
ax.set_xlabel("Color (bp_rp)")
ax.set_ylabel("Magnitud absoluta")
ax.set_title("Isolation Forest — Anomalías")
ax.legend(markerscale=3, fontsize=7)
ax.grid(True, alpha=0.2)

plt.suptitle("Análisis No Supervisado — Datos Estelares GAIA", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("analisis_estelar.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Guardado: analisis_estelar.png")

# Varianza explicada por PCA
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, 6), var_exp, color="steelblue", alpha=0.7, label="Individual")
ax.plot(range(1, 6), var_acum, "o-", color="tomato", label="Acumulada")
ax.axhline(0.90, color="gray", linestyle="--", label="90%")
ax.set_xlabel("Componente principal")
ax.set_ylabel("Varianza explicada")
ax.set_title("PCA — Varianza Explicada")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_varianza.png", dpi=100)
plt.close()
print("  Guardado: pca_varianza.png")

print("\n" + "=" * 62)
print("  RESUMEN FINAL")
print("=" * 62)
print(f"  Dataset            : {len(X)} estrellas GAIA ({origen})")
print(f"  Features           : {features}")
print(f"  PCA (2 componentes): {pca2.explained_variance_ratio_.sum():.1%} varianza retenida")
print(f"  K-Means            : {k_optimo} clusters (por silueta)")
print(f"  DBSCAN             : {n_clusters_db} clusters + {n_ruido} puntos ruido")
print(f"  Isolation Forest   : {n_anomalias} estrellas anómalas ({n_anomalias/len(X):.1%})")
print()
print("  Interpretación crítica:")
print("  Los clusters de K-Means recuperan naturalmente las poblaciones")
print("  estelares del diagrama HR sin haber visto ninguna etiqueta.")
print("  DBSCAN identifica las mismas regiones densas y además marca")
print("  como ruido los objetos en zonas de baja densidad poblacional.")
print("  Isolation Forest detecta estrellas con propiedades físicas")
print("  atípicas que merecen inspección individual.")
print("=" * 62)
