"""
TP Bloque 9: Aprendizaje Supervisado — Regresión
Dataset: California Housing (precios de viviendas)

Cubre:
  1. Carga y exploración del dataset
  2. Split Train / Validation / Test
  3. Regresión lineal base
  4. Diagnóstico: underfitting, buen ajuste, overfitting (polinómico)
  5. Curvas de aprendizaje
  6. K-Fold Cross Validation
  7. Regularización: Ridge (L2) y Lasso (L1)
  8. Evaluación final sobre Test
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # sin pantalla, guarda en archivo
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. CARGA Y EXPLORACIÓN
# ─────────────────────────────────────────────

print("=" * 62)
print("  BLOQUE 9 — Regresión: California Housing")
print("=" * 62)

data    = fetch_california_housing(as_frame=True)
df      = data.frame
X_full  = data.data.values
y_full  = data.target.values   # precio en cientos de miles de USD

print(f"\n  Ejemplos  : {X_full.shape[0]:,}")
print(f"  Features  : {X_full.shape[1]} → {list(data.feature_names)}")
print(f"  Target    : precio de vivienda (en $100k)")
print(f"  Rango Y   : ${y_full.min():.2f}k — ${y_full.max():.2f}k")
print(f"  Media Y   : ${y_full.mean():.2f}k")

# ─────────────────────────────────────────────
# 2. SPLIT TRAIN / VALIDATION / TEST
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  2. SPLIT TRAIN / VALIDATION / TEST (60 / 20 / 20)")
print("─" * 62)

# Primero separamos test (20%), luego del resto sacamos validation (25% = 20% del total)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42)

print(f"  Train      : {len(X_train):,} ejemplos")
print(f"  Validation : {len(X_val):,} ejemplos")
print(f"  Test       : {len(X_test):,} ejemplos  ← solo se toca al final")

# Escalado (necesario para regresión polinómica y regularización)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit solo en train
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. REGRESIÓN LINEAL BASE
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  3. REGRESIÓN LINEAL BASE")
print("─" * 62)

lr = LinearRegression()
lr.fit(X_train, y_train)

mse_train = mean_squared_error(y_train, lr.predict(X_train))
mse_val   = mean_squared_error(y_val,   lr.predict(X_val))
r2_val    = r2_score(y_val,   lr.predict(X_val))

print(f"  MSE Train      : {mse_train:.4f}")
print(f"  MSE Validation : {mse_val:.4f}")
print(f"  R² Validation  : {r2_val:.4f}  (1.0 = perfecto)")
print()
print("  Coeficientes aprendidos:")
for nombre, coef in zip(data.feature_names, lr.coef_):
    barra = "+" * int(abs(coef) * 3) if coef > 0 else "-" * int(abs(coef) * 3)
    print(f"    {nombre:<15} {coef:+.4f}  {barra}")

# ─────────────────────────────────────────────
# 4. DIAGNÓSTICO: UNDERFITTING / OVERFITTING
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  4. DIAGNÓSTICO: UNDERFITTING vs OVERFITTING")
print("─" * 62)

resultados = []

for grado in [1, 2, 3]:
    pipe = Pipeline([
        ("poly",  PolynomialFeatures(degree=grado, include_bias=False)),
        ("model", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    mse_tr = mean_squared_error(y_train, pipe.predict(X_train))
    mse_va = mean_squared_error(y_val,   pipe.predict(X_val))
    resultados.append((grado, mse_tr, mse_va))

    if grado == 1:
        diagnostico = "posible underfitting (modelo simple)"
    elif grado == 2:
        diagnostico = "mejor ajuste"
    else:
        diagnostico = "posible overfitting (demasiado complejo)"

    print(f"  Grado {grado}: MSE train={mse_tr:.4f}  MSE val={mse_va:.4f}  → {diagnostico}")

# ─────────────────────────────────────────────
# 5. CURVAS DE APRENDIZAJE
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  5. CURVAS DE APRENDIZAJE  →  curvas_aprendizaje.png")
print("─" * 62)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (grado, titulo) in zip(axes, [(1, "Lineal (grado 1)"), (2, "Polinómico (grado 2)")]):
    pipe = Pipeline([
        ("poly",  PolynomialFeatures(degree=grado, include_bias=False)),
        ("model", LinearRegression())
    ])
    sizes, scores_train, scores_val = learning_curve(
        pipe, X_train, y_train,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    mse_tr_lc = -scores_train.mean(axis=1)
    mse_va_lc = -scores_val.mean(axis=1)

    ax.plot(sizes, mse_tr_lc, label="Train",      color="steelblue")
    ax.plot(sizes, mse_va_lc, label="Validation", color="tomato")
    ax.set_title(f"Curva de aprendizaje — {titulo}")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("curvas_aprendizaje.png", dpi=100)
plt.close()
print("  Guardado: curvas_aprendizaje.png")
print("  Si train y val convergen alto  → underfitting")
print("  Si train bajo y val alto       → overfitting")

# ─────────────────────────────────────────────
# 6. K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  6. K-FOLD CROSS VALIDATION (k=5)")
print("─" * 62)

kf      = KFold(n_splits=5, shuffle=True, random_state=42)
mses_cv = []

for fold, (idx_tr, idx_va) in enumerate(kf.split(X_train), 1):
    X_tr_k, X_va_k = X_train[idx_tr], X_train[idx_va]
    y_tr_k, y_va_k = y_train[idx_tr], y_train[idx_va]

    m = LinearRegression()
    m.fit(X_tr_k, y_tr_k)
    mse_k = mean_squared_error(y_va_k, m.predict(X_va_k))
    mses_cv.append(mse_k)
    print(f"  Fold {fold}: MSE = {mse_k:.4f}")

print(f"\n  MSE promedio : {np.mean(mses_cv):.4f}")
print(f"  Desv. estándar: {np.std(mses_cv):.4f}  (estabilidad del modelo)")

# ─────────────────────────────────────────────
# 7. REGULARIZACIÓN: RIDGE Y LASSO
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  7. REGULARIZACIÓN: Ridge (L2) vs Lasso (L1)")
print("─" * 62)

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

print(f"\n  {'Alpha':<10} {'Ridge MSE val':<20} {'Lasso MSE val':<20}")
print("  " + "-" * 48)

mejores = {"ridge": (None, float("inf")), "lasso": (None, float("inf"))}

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_train, y_train)

    mse_r = mean_squared_error(y_val, ridge.predict(X_val))
    mse_l = mean_squared_error(y_val, lasso.predict(X_val))

    print(f"  {alpha:<10} {mse_r:<20.4f} {mse_l:<20.4f}")

    if mse_r < mejores["ridge"][1]:
        mejores["ridge"] = (alpha, mse_r)
    if mse_l < mejores["lasso"][1]:
        mejores["lasso"] = (alpha, mse_l)

print(f"\n  Mejor Ridge: alpha={mejores['ridge'][0]}  MSE={mejores['ridge'][1]:.4f}")
print(f"  Mejor Lasso: alpha={mejores['lasso'][0]}  MSE={mejores['lasso'][1]:.4f}")

# Comparar coeficientes: sin reg vs Ridge vs Lasso
best_ridge = Ridge(alpha=mejores["ridge"][0]).fit(X_train, y_train)
best_lasso = Lasso(alpha=mejores["lasso"][0], max_iter=5000).fit(X_train, y_train)

print("\n  Comparación de coeficientes (qué aprendió cada modelo):")
print(f"  {'Feature':<15} {'Sin reg':>10} {'Ridge':>10} {'Lasso':>10}")
print("  " + "-" * 48)
for nombre, c_lr, c_r, c_l in zip(
        data.feature_names, lr.coef_, best_ridge.coef_, best_lasso.coef_):
    print(f"  {nombre:<15} {c_lr:>10.4f} {c_r:>10.4f} {c_l:>10.4f}")

print("\n  Lasso pone coeficientes en exactamente 0 (selección de features)")
lasso_cero = [n for n, c in zip(data.feature_names, best_lasso.coef_) if c == 0]
if lasso_cero:
    print(f"  Features eliminadas por Lasso: {lasso_cero}")

# ─────────────────────────────────────────────
# 8. EVALUACIÓN FINAL SOBRE TEST
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  8. EVALUACIÓN FINAL SOBRE TEST  ← solo se toca aquí")
print("─" * 62)

modelos_finales = {
    "Lineal sin reg": lr,
    "Ridge (mejor)":  best_ridge,
    "Lasso (mejor)":  best_lasso,
}

print(f"\n  {'Modelo':<20} {'MSE Test':>12} {'R² Test':>10}")
print("  " + "-" * 44)

for nombre, modelo in modelos_finales.items():
    mse_t = mean_squared_error(y_test, modelo.predict(X_test))
    r2_t  = r2_score(y_test, modelo.predict(X_test))
    print(f"  {nombre:<20} {mse_t:>12.4f} {r2_t:>10.4f}")

# ─────────────────────────────────────────────
# 9. GRÁFICO FINAL: predicción vs real
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  9. GRÁFICO PREDICCIÓN vs REAL  →  prediccion_vs_real.png")
print("─" * 62)

y_pred_test = best_ridge.predict(X_test)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_test, alpha=0.2, s=10, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=1.5, label="Predicción perfecta")
plt.xlabel("Precio real ($100k)")
plt.ylabel("Precio predicho ($100k)")
plt.title("Ridge — Predicción vs Real (Test)")
plt.legend()
plt.tight_layout()
plt.savefig("prediccion_vs_real.png", dpi=100)
plt.close()
print("  Guardado: prediccion_vs_real.png")

print("\n" + "=" * 62)
print("  RESUMEN FINAL")
print("=" * 62)
print("  Train/Val/Test : split 60/20/20")
print("  Lineal base    : R² ≈ {:.3f} en validation".format(r2_val))
print("  Mejor modelo   : Ridge con alpha={}".format(mejores["ridge"][0]))
print("  Archivos       : curvas_aprendizaje.png, prediccion_vs_real.png")
print("=" * 62)
