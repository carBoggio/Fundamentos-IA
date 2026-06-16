"""
TP Bloque 10: Aprendizaje Supervisado — Clasificación
Dataset: Titanic (sobrevivió o no)

Cubre:
  1. Carga y preprocesamiento del dataset
  2. Split Train / Validation / Test
  3. Árbol de Decisión con visualización de reglas
  4. Random Forest
  5. XGBoost
  6. Evaluación: matriz de confusión, accuracy, precision, recall, F1, AUC
  7. Comparación final de los tres modelos
  8. Importancia de features
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. CARGA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────

print("=" * 62)
print("  BLOQUE 10 — Clasificación: Titanic")
print("=" * 62)

df = sns.load_dataset("titanic")

print(f"\n  Ejemplos totales : {len(df)}")
print(f"  Sobrevivieron    : {df['survived'].sum()} ({df['survived'].mean():.1%})")
print(f"  No sobrevivieron : {(df['survived'] == 0).sum()} ({1 - df['survived'].mean():.1%})")
print(f"\n  Valores faltantes por columna:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# Preprocesamiento
df["age"]      = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["sex"]      = (df["sex"] == "male").astype(int)          # male=1, female=0
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[features].values
y = df["survived"].values

print(f"\n  Features usadas : {features}")
print(f"  Encoding        : sex (male=1, female=0), embarked (S=0, C=1, Q=2)")

# ─────────────────────────────────────────────
# 2. SPLIT TRAIN / VALIDATION / TEST
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  2. SPLIT TRAIN / VALIDATION / TEST (60 / 20 / 20)")
print("─" * 62)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"  Train      : {len(X_train)} ejemplos")
print(f"  Validation : {len(X_val)} ejemplos")
print(f"  Test       : {len(X_test)} ejemplos  ← solo se toca al final")

# ─────────────────────────────────────────────
# 3. ÁRBOL DE DECISIÓN
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  3. ÁRBOL DE DECISIÓN")
print("─" * 62)

# Comparar distintas profundidades para mostrar overfitting
print("\n  Efecto de la profundidad máxima:")
print(f"  {'Profundidad':<15} {'Acc Train':>12} {'Acc Val':>12}")
print("  " + "-" * 40)

mejor_arbol = None
mejor_acc_val = 0

for depth in [2, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, dt.predict(X_train))
    acc_va = accuracy_score(y_val,   dt.predict(X_val))
    label  = str(depth) if depth else "sin límite"
    ovfit  = " ← overfitting" if acc_tr - acc_va > 0.08 else ""
    print(f"  {label:<15} {acc_tr:>12.3f} {acc_va:>12.3f}{ovfit}")
    if acc_va > mejor_acc_val:
        mejor_acc_val = acc_va
        mejor_arbol = dt

print(f"\n  Mejor profundidad: {mejor_arbol.max_depth}")

# Mostrar reglas del árbol (profundidad 3 para que sea legible)
dt_legible = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_legible.fit(X_train, y_train)
print("\n  Reglas aprendidas (profundidad 3):")
print(export_text(dt_legible, feature_names=features))

# ─────────────────────────────────────────────
# 4. RANDOM FOREST
# ─────────────────────────────────────────────

print("─" * 62)
print("  4. RANDOM FOREST")
print("─" * 62)

print("\n  Efecto del número de árboles:")
print(f"  {'N árboles':<15} {'Acc Train':>12} {'Acc Val':>12}")
print("  " + "-" * 40)

mejor_rf = None
mejor_acc_rf = 0

for n in [10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, rf.predict(X_train))
    acc_va = accuracy_score(y_val,   rf.predict(X_val))
    print(f"  {n:<15} {acc_tr:>12.3f} {acc_va:>12.3f}")
    if acc_va > mejor_acc_rf:
        mejor_acc_rf = acc_va
        mejor_rf = rf

print(f"\n  Mejor n_estimators: {mejor_rf.n_estimators}")

# ─────────────────────────────────────────────
# 5. XGBOOST
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  5. XGBOOST")
print("─" * 62)

print("\n  Efecto del número de rondas de boosting:")
print(f"  {'N rondas':<15} {'Acc Train':>12} {'Acc Val':>12}")
print("  " + "-" * 40)

mejor_xgb = None
mejor_acc_xgb = 0

for n in [50, 100, 200, 300]:
    xgb = XGBClassifier(n_estimators=n, learning_rate=0.1,
                        max_depth=4, random_state=42,
                        eval_metric="logloss", verbosity=0)
    xgb.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, xgb.predict(X_train))
    acc_va = accuracy_score(y_val,   xgb.predict(X_val))
    print(f"  {n:<15} {acc_tr:>12.3f} {acc_va:>12.3f}")
    if acc_va > mejor_acc_xgb:
        mejor_acc_xgb = acc_va
        mejor_xgb = xgb

print(f"\n  Mejor n_estimators: {mejor_xgb.n_estimators}")

# ─────────────────────────────────────────────
# 6. EVALUACIÓN FINAL SOBRE TEST
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  6. EVALUACIÓN FINAL SOBRE TEST  ← solo se toca aquí")
print("─" * 62)

modelos = {
    "Árbol":         mejor_arbol,
    "Random Forest": mejor_rf,
    "XGBoost":       mejor_xgb,
}

metricas = {}
for nombre, modelo in modelos.items():
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    metricas[nombre] = {
        "acc":  accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred),
        "rec":  recall_score(y_test, y_pred),
        "f1":   f1_score(y_test, y_pred),
        "auc":  roc_auc_score(y_test, y_proba),
        "cm":   confusion_matrix(y_test, y_pred),
        "proba": y_proba,
        "pred":  y_pred,
    }

print(f"\n  {'Modelo':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("  " + "-" * 62)
for nombre, m in metricas.items():
    print(f"  {nombre:<16} {m['acc']:>9.3f} {m['prec']:>10.3f} {m['rec']:>8.3f} {m['f1']:>8.3f} {m['auc']:>8.3f}")

# Matrices de confusión
print("\n  Matrices de confusión (Test):")
for nombre, m in metricas.items():
    cm = m["cm"]
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {nombre}:")
    print(f"    Verdaderos Negativos (no sobrevivió, bien) : {tn}")
    print(f"    Falsos Positivos (predijo sobrevivió, mal) : {fp}")
    print(f"    Falsos Negativos (predijo no, era sí)      : {fn}")
    print(f"    Verdaderos Positivos (sobrevivió, bien)    : {tp}")

# ─────────────────────────────────────────────
# 7. GRÁFICOS
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  7. GRÁFICOS")
print("─" * 62)

colores = {"Árbol": "steelblue", "Random Forest": "seagreen", "XGBoost": "tomato"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Matrices de confusión
for ax, (nombre, m) in zip(axes, metricas.items()):
    sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    ax.set_title(f"{nombre}\nAcc={m['acc']:.3f}  F1={m['f1']:.3f}")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.suptitle("Matrices de Confusión — Test", y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Guardado: confusion_matrices.png")

# Curvas ROC
plt.figure(figsize=(6, 5))
for nombre, m in metricas.items():
    fpr, tpr, _ = roc_curve(y_test, m["proba"])
    plt.plot(fpr, tpr, label=f"{nombre} (AUC={m['auc']:.3f})", color=colores[nombre])
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio (AUC=0.5)")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curvas ROC — Test")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("curvas_roc.png", dpi=100)
plt.close()
print("  Guardado: curvas_roc.png")

# Importancia de features (Random Forest y XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (nombre, modelo) in zip(axes, [("Random Forest", mejor_rf), ("XGBoost", mejor_xgb)]):
    importancias = modelo.feature_importances_
    orden = np.argsort(importancias)[::-1]
    ax.bar(range(len(features)), importancias[orden],
           color="steelblue" if nombre == "Random Forest" else "tomato")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in orden], rotation=30, ha="right")
    ax.set_title(f"Importancia de Features — {nombre}")
    ax.set_ylabel("Importancia")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()
print("  Guardado: feature_importance.png")

# ─────────────────────────────────────────────
# 8. RESUMEN FINAL
# ─────────────────────────────────────────────

ganador = max(metricas, key=lambda n: metricas[n]["f1"])

print("\n" + "=" * 62)
print("  RESUMEN FINAL")
print("=" * 62)
print(f"  Dataset      : Titanic ({len(X)} pasajeros)")
print(f"  Split        : 60/20/20 estratificado")
print(f"  Mejor modelo : {ganador} (F1={metricas[ganador]['f1']:.3f})")
print()
print("  Interpretación de métricas en este contexto:")
print("  Precision: de los que predije que sobrevivieron, cuántos realmente lo hicieron")
print("  Recall   : de los que sobrevivieron, cuántos detecté correctamente")
print("  F1       : balance entre precision y recall")
print("  AUC      : qué tan bien separa el modelo las dos clases en general")
print()
print("  Archivos generados:")
print("    confusion_matrices.png")
print("    curvas_roc.png")
print("    feature_importance.png")
print("=" * 62)
