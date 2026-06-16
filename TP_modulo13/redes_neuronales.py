"""
TP Bloque 13: Fundamentos de Redes Neuronales
Datasets: XOR, Moons, Circles

Cubre:
  1. Problema XOR: perceptrón lineal vs MLP
  2. MLP en Moons y Circles: fronteras de decisión
  3. Efecto de capas ocultas y neuronas
  4. Curvas de pérdida: train vs validation
  5. Overfitting: red grande sin regularización
  6. Regularización: L2, Dropout (early stopping)
  7. Efecto del learning rate
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ─────────────────────────────────────────────
# UTILIDAD: frontera de decisión
# ─────────────────────────────────────────────

def graficar_frontera(ax, modelo, X, y, titulo=""):
    h = 0.02
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_bg = ListedColormap(['#AACFEE', '#FFAAAA'])
    cmap_pt = ListedColormap(['#1f77b4', '#d62728'])
    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap_pt, s=20, edgecolors='k', lw=0.3)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

def graficar_perdida(ax, modelo, titulo="Curva de pérdida"):
    ax.plot(modelo.loss_curve_, color='steelblue', lw=1.5, label='Train loss')
    if hasattr(modelo, 'validation_fraction'):
        pass
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Loss")
    ax.set_title(titulo, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# ─────────────────────────────────────────────
# 1. PROBLEMA XOR
# ─────────────────────────────────────────────

print("=" * 62)
print("  BLOQUE 13 — Redes Neuronales")
print("=" * 62)
print("\n" + "─" * 62)
print("  1. PROBLEMA XOR")
print("─" * 62)

# XOR: no linealmente separable
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

# Perceptrón (sin capa oculta)
p_lineal = MLPClassifier(hidden_layer_sizes=(), max_iter=1000,
                         activation='logistic', random_state=42)
p_lineal.fit(X_xor, y_xor)

# MLP con capa oculta
p_mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000,
                      activation='tanh', random_state=42)
p_mlp.fit(X_xor, y_xor)

print(f"\n  Perceptrón lineal (sin capa oculta):")
print(f"    Predicciones : {p_lineal.predict(X_xor)}")
print(f"    Esperado     : {y_xor}")
print(f"    Accuracy     : {accuracy_score(y_xor, p_lineal.predict(X_xor)):.0%}")

print(f"\n  MLP con capa oculta (4 neuronas, tanh):")
print(f"    Predicciones : {p_mlp.predict(X_xor)}")
print(f"    Esperado     : {y_xor}")
print(f"    Accuracy     : {accuracy_score(y_xor, p_mlp.predict(X_xor)):.0%}")

print(f"""
  Conclusión XOR:
    El perceptrón lineal no puede resolver XOR porque no existe
    ninguna línea recta que separe las clases.
    El MLP con una capa oculta aprende una representación intermedia
    que transforma el espacio hasta hacer el problema separable.
""")

# ─────────────────────────────────────────────
# 2. DATASETS: MOONS Y CIRCLES
# ─────────────────────────────────────────────

print("─" * 62)
print("  2. MOONS Y CIRCLES — Fronteras de decisión")
print("─" * 62)

X_moons, y_moons   = make_moons(n_samples=300, noise=0.2, random_state=42)
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

scaler = StandardScaler()
X_moons_s   = scaler.fit_transform(X_moons)
X_circles_s = scaler.fit_transform(X_circles)

def entrenar_y_reportar(X, y, capas, nombre, **kwargs):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    m = MLPClassifier(hidden_layer_sizes=capas, max_iter=2000,
                      random_state=42, **kwargs)
    m.fit(Xtr, ytr)
    acc_tr = accuracy_score(ytr, m.predict(Xtr))
    acc_va = accuracy_score(yva, m.predict(Xva))
    print(f"  {nombre:<40} acc_train={acc_tr:.3f}  acc_val={acc_va:.3f}  iters={m.n_iter_}")
    return m

print("\n  Moons:")
m_moons_sin  = entrenar_y_reportar(X_moons_s, y_moons, (),       "Sin capa oculta (lineal)")
m_moons_ch   = entrenar_y_reportar(X_moons_s, y_moons, (4,),     "1 capa, 4 neuronas")
m_moons_med  = entrenar_y_reportar(X_moons_s, y_moons, (16,),    "1 capa, 16 neuronas")
m_moons_deep = entrenar_y_reportar(X_moons_s, y_moons, (16,8),   "2 capas (16,8)")

print("\n  Circles:")
m_circ_sin   = entrenar_y_reportar(X_circles_s, y_circles, (),      "Sin capa oculta (lineal)")
m_circ_ch    = entrenar_y_reportar(X_circles_s, y_circles, (4,),    "1 capa, 4 neuronas")
m_circ_med   = entrenar_y_reportar(X_circles_s, y_circles, (16,),   "1 capa, 16 neuronas")
m_circ_deep  = entrenar_y_reportar(X_circles_s, y_circles, (16,8),  "2 capas (16,8)")

# ─────────────────────────────────────────────
# 3. OVERFITTING Y REGULARIZACIÓN
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  3. OVERFITTING Y REGULARIZACIÓN")
print("─" * 62)

Xtr, Xva, ytr, yva = train_test_split(X_moons_s, y_moons, test_size=0.2, random_state=42)

print("\n  Comparación en Moons:")
print(f"  {'Modelo':<35} {'Train':>8} {'Val':>8}")
print("  " + "-" * 54)

configs_reg = [
    ("Red grande sin reg",    dict(hidden_layer_sizes=(128,64,32), alpha=0.0)),
    ("L2 suave (α=0.01)",     dict(hidden_layer_sizes=(128,64,32), alpha=0.01)),
    ("L2 fuerte (α=0.1)",     dict(hidden_layer_sizes=(128,64,32), alpha=0.1)),
    ("Red chica (sin overfit)",dict(hidden_layer_sizes=(8,),        alpha=0.0)),
]

modelos_reg = {}
for nombre, kw in configs_reg:
    m = MLPClassifier(max_iter=2000, random_state=42, **kw)
    m.fit(Xtr, ytr)
    acc_tr = accuracy_score(ytr, m.predict(Xtr))
    acc_va = accuracy_score(yva, m.predict(Xva))
    gap = acc_tr - acc_va
    flag = " ← OVERFITTING" if gap > 0.05 else ""
    print(f"  {nombre:<35} {acc_tr:>8.3f} {acc_va:>8.3f}{flag}")
    modelos_reg[nombre] = m

# ─────────────────────────────────────────────
# 4. EFECTO DEL LEARNING RATE
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  4. EFECTO DEL LEARNING RATE")
print("─" * 62)

print(f"\n  {'Learning Rate':<20} {'Iters':>8} {'Val Acc':>10} {'Convergió':>12}")
print("  " + "-" * 54)

lrs = [0.001, 0.01, 0.1, 0.5]
modelos_lr = {}
for lr in lrs:
    m = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=2000,
                      learning_rate_init=lr, random_state=42,
                      solver='sgd', learning_rate='constant')
    m.fit(Xtr, ytr)
    acc_va  = accuracy_score(yva, m.predict(Xva))
    convergio = "Sí" if m.n_iter_ < 2000 else "No (max_iter)"
    print(f"  {lr:<20} {m.n_iter_:>8} {acc_va:>10.3f} {convergio:>12}")
    modelos_lr[str(lr)] = m

# ─────────────────────────────────────────────
# 5. FUNCIONES DE ACTIVACIÓN
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  5. FUNCIONES DE ACTIVACIÓN")
print("─" * 62)

print(f"\n  {'Activación':<15} {'Train':>8} {'Val':>8} {'Iters':>8}")
print("  " + "-" * 42)

for act in ['logistic', 'tanh', 'relu']:
    m = MLPClassifier(hidden_layer_sizes=(16,8), activation=act,
                      max_iter=2000, random_state=42)
    m.fit(Xtr, ytr)
    acc_tr = accuracy_score(ytr, m.predict(Xtr))
    acc_va = accuracy_score(yva, m.predict(Xva))
    print(f"  {act:<15} {acc_tr:>8.3f} {acc_va:>8.3f} {m.n_iter_:>8}")

# ─────────────────────────────────────────────
# 6. GRÁFICOS
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  6. GRÁFICOS")
print("─" * 62)

# --- Figura 1: XOR ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Grilla densa para XOR
X_xor_g = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
# Expandir para visualizar mejor
X_xor_vis = np.array([[-0.2,-0.2],[-0.2,1.2],[1.2,-0.2],[1.2,1.2]])

for ax, modelo, titulo in zip(axes,
    [p_lineal, p_mlp],
    ["Perceptrón lineal (sin capa oculta)\nNo puede resolver XOR",
     "MLP con capa oculta (4 neuronas, tanh)\nResuelve XOR correctamente"]):

    h = 0.02
    xx, yy = np.meshgrid(np.arange(-0.5, 1.6, h), np.arange(-0.5, 1.6, h))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap_bg = ListedColormap(['#AACFEE','#FFAAAA'])
    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
    ax.scatter(X_xor[:,0], X_xor[:,1], c=y_xor,
               cmap=ListedColormap(['#1f77b4','#d62728']),
               s=200, edgecolors='k', lw=1.5, zorder=5)
    for xi, yi, li in zip(X_xor[:,0], X_xor[:,1], y_xor):
        ax.text(xi+0.05, yi+0.05, f"({int(xi)},{int(yi)})={li}", fontsize=8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(titulo, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Problema XOR: Perceptrón vs MLP", fontsize=12)
plt.tight_layout()
plt.savefig("xor.png", dpi=100)
plt.close()
print("  Guardado: xor.png")

# --- Figura 2: fronteras de decisión Moons y Circles ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (modelo, titulo) in enumerate([
    (m_moons_sin,  "Moons\nSin capa oculta"),
    (m_moons_ch,   "Moons\n1 capa, 4 neur."),
    (m_moons_med,  "Moons\n1 capa, 16 neur."),
    (m_moons_deep, "Moons\n2 capas (16,8)"),
]):
    graficar_frontera(axes[0,i], modelo, X_moons_s, y_moons, titulo)

for i, (modelo, titulo) in enumerate([
    (m_circ_sin,   "Circles\nSin capa oculta"),
    (m_circ_ch,    "Circles\n1 capa, 4 neur."),
    (m_circ_med,   "Circles\n1 capa, 16 neur."),
    (m_circ_deep,  "Circles\n2 capas (16,8)"),
]):
    graficar_frontera(axes[1,i], modelo, X_circles_s, y_circles, titulo)

plt.suptitle("Fronteras de Decisión — Moons y Circles", fontsize=12)
plt.tight_layout()
plt.savefig("fronteras_decision.png", dpi=100)
plt.close()
print("  Guardado: fronteras_decision.png")

# --- Figura 3: overfitting y regularización ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

colores_map = {
    "Red grande sin reg":     '#e74c3c',
    "L2 suave (α=0.01)":      '#f39c12',
    "L2 fuerte (α=0.1)":      '#2ecc71',
    "Red chica (sin overfit)": '#3498db',
}
for ax, (nombre, modelo) in zip(axes, modelos_reg.items()):
    graficar_frontera(ax, modelo, X_moons_s, y_moons,
                      titulo=nombre.replace(" (", "\n("))

plt.suptitle("Overfitting y Regularización L2 — Moons", fontsize=12)
plt.tight_layout()
plt.savefig("overfitting_regularizacion.png", dpi=100)
plt.close()
print("  Guardado: overfitting_regularizacion.png")

# --- Figura 4: curvas de pérdida por learning rate ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
colores_lr = ['#3498db','#2ecc71','#e74c3c','#9b59b6']

for ax, (lr, color) in zip(axes, zip(lrs, colores_lr)):
    m = modelos_lr[str(lr)]
    ax.plot(m.loss_curve_, color=color, lw=1.5)
    ax.set_title(f"LR = {lr}\niters={m.n_iter_}  val={accuracy_score(yva, m.predict(Xva)):.3f}",
                 fontsize=9)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

plt.suptitle("Efecto del Learning Rate — Curvas de Pérdida", fontsize=12)
plt.tight_layout()
plt.savefig("learning_rate.png", dpi=100)
plt.close()
print("  Guardado: learning_rate.png")

# --- Figura 5: funciones de activación (visualización matemática) ---
fig, ax = plt.subplots(figsize=(8, 4))
x_act = np.linspace(-4, 4, 200)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x):    return np.maximum(0, x)

ax.plot(x_act, sigmoid(x_act), label='Sigmoid', color='#3498db', lw=2)
ax.plot(x_act, np.tanh(x_act), label='Tanh',    color='#2ecc71', lw=2)
ax.plot(x_act, relu(x_act),    label='ReLU',    color='#e74c3c', lw=2)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel("Entrada (z = Wx + b)")
ax.set_ylabel("Activación f(z)")
ax.set_title("Funciones de Activación")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig("activaciones.png", dpi=100)
plt.close()
print("  Guardado: activaciones.png")

print("\n" + "=" * 62)
print("  RESUMEN FINAL")
print("=" * 62)
print("  XOR         : perceptrón falla (lineal), MLP lo resuelve")
print("  Moons/Circles: sin capa oculta → frontera recta (insuficiente)")
print("               más neuronas/capas → fronteras curvas correctas")
print("  Overfitting  : red grande memoriza train, falla en val")
print("  L2 reg       : reduce overfitting penalizando pesos grandes")
print("  Learning rate: muy alto diverge, muy bajo converge lento")
print("  Activaciones : ReLU converge más rápido en la mayoría de casos")
print()
print("  Archivos generados:")
print("    xor.png")
print("    fronteras_decision.png")
print("    overfitting_regularizacion.png")
print("    learning_rate.png")
print("    activaciones.png")
print("=" * 62)
