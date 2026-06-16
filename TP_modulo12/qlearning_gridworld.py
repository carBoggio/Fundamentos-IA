"""
TP Bloque 12: Aprendizaje por Refuerzo
Q-Learning en GridWorld

Cubre:
  1. Entorno GridWorld configurable
  2. Agente Q-Learning con ε-greedy
  3. Entrenamiento y curva de retorno
  4. Visualización de la tabla Q y política aprendida
  5. Comparación exploración vs explotación
  6. Experimentos: efecto de α, γ, ε y cambios en el entorno
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. ENTORNO GRIDWORLD
# ─────────────────────────────────────────────

class GridWorld:
    """
    Grilla configurable con:
      - START : posición inicial del agente
      - META  : recompensa positiva, termina el episodio
      - TRAMPAS: recompensa negativa, termina el episodio
      - MUROS : celdas bloqueadas
      - PASO  : pequeña penalización por cada movimiento
    """

    ACCIONES      = [(-1,0), (1,0), (0,-1), (0,1)]   # arriba, abajo, izq, der
    NOMBRES_ACC   = ["↑", "↓", "←", "→"]

    def __init__(self, config):
        self.filas     = config["filas"]
        self.cols      = config["cols"]
        self.inicio    = config["inicio"]
        self.meta      = config["meta"]
        self.muros     = set(map(tuple, config.get("muros", [])))
        self.trampas   = {tuple(k): v for k, v in config.get("trampas", {}).items()}
        self.r_meta    = config.get("r_meta",   10.0)
        self.r_trampa  = config.get("r_trampa", -5.0)
        self.r_paso    = config.get("r_paso",   -0.1)
        self.reset()

    def reset(self):
        self.pos = self.inicio
        return self.pos

    def step(self, accion):
        dr, dc     = self.ACCIONES[accion]
        nueva_pos  = (self.pos[0] + dr, self.pos[1] + dc)

        # Si choca con muro o sale del borde, se queda
        if (0 <= nueva_pos[0] < self.filas and
            0 <= nueva_pos[1] < self.cols and
            nueva_pos not in self.muros):
            self.pos = nueva_pos

        if self.pos == self.meta:
            return self.pos, self.r_meta, True
        if self.pos in self.trampas:
            return self.pos, self.trampas[self.pos], True
        return self.pos, self.r_paso, False

    def estados_validos(self):
        return [(r, c) for r in range(self.filas)
                       for c in range(self.cols)
                       if (r, c) not in self.muros]

# ─────────────────────────────────────────────
# 2. AGENTE Q-LEARNING
# ─────────────────────────────────────────────

class AgenteQLearning:
    """
    Q-Learning tabular con ε-greedy.

    Actualización:
      Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

    α (alpha) : tasa de aprendizaje — cuánto peso le damos a lo nuevo
    γ (gamma) : factor de descuento — cuánto importa el futuro
    ε (epsilon): probabilidad de explorar en vez de explotar
    """

    def __init__(self, n_acciones, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.n_acciones = n_acciones
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.Q          = defaultdict(lambda: np.zeros(n_acciones))

    def elegir_accion(self, estado, forzar_explotacion=False):
        if not forzar_explotacion and np.random.random() < self.epsilon:
            return np.random.randint(self.n_acciones)   # exploración
        return int(np.argmax(self.Q[estado]))            # explotación

    def actualizar(self, s, a, r, s_siguiente, terminado):
        q_actual   = self.Q[s][a]
        q_futuro   = 0 if terminado else np.max(self.Q[s_siguiente])
        td_error   = r + self.gamma * q_futuro - q_actual
        self.Q[s][a] += self.alpha * td_error

    def politica(self, estado):
        return int(np.argmax(self.Q[estado]))

# ─────────────────────────────────────────────
# 3. ENTRENAMIENTO
# ─────────────────────────────────────────────

def entrenar(env_config, alpha=0.1, gamma=0.95, epsilon=0.1,
             n_episodios=500, max_pasos=200):
    env    = GridWorld(env_config)
    agente = AgenteQLearning(4, alpha, gamma, epsilon)
    retornos  = []
    pasos_ep  = []

    for ep in range(n_episodios):
        s         = env.reset()
        retorno   = 0
        pasos     = 0

        for _ in range(max_pasos):
            a              = agente.elegir_accion(s)
            s_sig, r, done = env.step(a)
            agente.actualizar(s, a, r, s_sig, done)
            s       = s_sig
            retorno += r
            pasos   += 1
            if done:
                break

        retornos.append(retorno)
        pasos_ep.append(pasos)

    return agente, retornos, pasos_ep

# ─────────────────────────────────────────────
# 4. VISUALIZACIÓN
# ─────────────────────────────────────────────

def suavizar(valores, ventana=20):
    return np.convolve(valores, np.ones(ventana)/ventana, mode='valid')

def graficar_grilla(ax, env_config, agente=None, titulo="GridWorld"):
    filas  = env_config["filas"]
    cols   = env_config["cols"]
    inicio = env_config["inicio"]
    meta   = env_config["meta"]
    muros  = set(map(tuple, env_config.get("muros", [])))
    trampas = {tuple(k): v for k, v in env_config.get("trampas", {}).items()}

    ax.set_xlim(0, cols)
    ax.set_ylim(0, filas)
    ax.set_aspect('equal')
    ax.set_title(titulo, fontsize=10)
    ax.axis('off')

    for r in range(filas):
        for c in range(cols):
            pos = (r, c)
            # color de celda
            if pos in muros:
                color = '#333333'
            elif pos == meta:
                color = '#2ecc71'
            elif pos in trampas:
                color = '#e74c3c'
            elif pos == inicio:
                color = '#3498db'
            else:
                color = '#f5f5f5'

            rect = plt.Rectangle([c, filas-1-r], 1, 1,
                                  facecolor=color, edgecolor='gray', lw=0.8)
            ax.add_patch(rect)

            # etiqueta
            if pos == meta:
                ax.text(c+0.5, filas-r-0.5, "META", ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')
            elif pos in trampas:
                ax.text(c+0.5, filas-r-0.5, f"T\n{trampas[pos]:.0f}",
                        ha='center', va='center', fontsize=6, color='white')
            elif pos == inicio:
                ax.text(c+0.5, filas-r-0.5, "S", ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
            elif pos not in muros and agente:
                # flecha de la política
                acc = agente.politica(pos)
                drs = [-0.3, 0.3, 0, 0]
                dcs = [0, 0, -0.3, 0.3]
                ax.annotate("", xy=(c+0.5+dcs[acc], filas-r-0.5+drs[acc]),
                            xytext=(c+0.5, filas-r-0.5),
                            arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2))

    # grid
    for r in range(filas+1):
        ax.axhline(r, color='gray', lw=0.5)
    for c in range(cols+1):
        ax.axvline(c, color='gray', lw=0.5)

def graficar_q_table(ax, env_config, agente, titulo="Tabla Q (max por estado)"):
    filas  = env_config["filas"]
    cols   = env_config["cols"]
    muros  = set(map(tuple, env_config.get("muros", [])))

    q_max = np.full((filas, cols), np.nan)
    for r in range(filas):
        for c in range(cols):
            if (r, c) not in muros:
                q_max[r, c] = np.max(agente.Q[(r, c)])

    im = ax.imshow(q_max, cmap='RdYlGn', aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(filas))

    for r in range(filas):
        for c in range(cols):
            if not np.isnan(q_max[r, c]):
                ax.text(c, r, f"{q_max[r,c]:.1f}", ha='center', va='center',
                        fontsize=7, color='black')

# ─────────────────────────────────────────────
# 5. CONFIG BASE
# ─────────────────────────────────────────────

CONFIG_BASE = {
    "filas": 6, "cols": 6,
    "inicio": (5, 0),
    "meta":   (0, 5),
    "muros":  [(1,1),(1,2),(2,2),(3,2),(3,3),(4,3)],
    "trampas": {(2,4): -5, (4,1): -5},
    "r_meta":  10.0,
    "r_paso":  -0.1,
}

# ─────────────────────────────────────────────
# 6. EXPERIMENTO PRINCIPAL
# ─────────────────────────────────────────────

print("=" * 62)
print("  BLOQUE 12 — Q-Learning en GridWorld")
print("=" * 62)

print("\n  Entrenando agente base (α=0.1, γ=0.95, ε=0.1, 500 episodios)...")
agente_base, retornos_base, pasos_base = entrenar(CONFIG_BASE, n_episodios=500)

print(f"  Retorno medio (últimos 50 ep): {np.mean(retornos_base[-50:]):.2f}")
print(f"  Pasos medios  (últimos 50 ep): {np.mean(pasos_base[-50:]):.1f}")

# ─────────────────────────────────────────────
# 7. COMPARACIÓN EXPLORACIÓN vs EXPLOTACIÓN
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  COMPARACIÓN: Exploración vs Explotación")
print("─" * 62)

configs_eps = [(0.01, "ε=0.01 (casi pura explotación)"),
               (0.10, "ε=0.10 (balance)"),
               (0.50, "ε=0.50 (mucha exploración)"),
               (1.00, "ε=1.00 (pura exploración aleatoria)")]

resultados_eps = {}
for eps, label in configs_eps:
    _, rets, _ = entrenar(CONFIG_BASE, epsilon=eps, n_episodios=500)
    resultados_eps[label] = rets
    print(f"  {label:<38} → retorno final: {np.mean(rets[-50:]):.2f}")

# ─────────────────────────────────────────────
# 8. EXPERIMENTOS: α, γ y CAMBIOS EN ENTORNO
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  EXPERIMENTOS: efecto de α, γ y cambios en el entorno")
print("─" * 62)

# Efecto de alpha
print("\n  Efecto de α (tasa de aprendizaje):")
resultados_alpha = {}
for alpha in [0.01, 0.1, 0.5, 0.9]:
    _, rets, _ = entrenar(CONFIG_BASE, alpha=alpha, n_episodios=500)
    resultados_alpha[f"α={alpha}"] = rets
    print(f"    α={alpha} → retorno final: {np.mean(rets[-50:]):.2f}")

# Efecto de gamma
print("\n  Efecto de γ (factor de descuento):")
resultados_gamma = {}
for gamma in [0.5, 0.8, 0.95, 0.99]:
    _, rets, _ = entrenar(CONFIG_BASE, gamma=gamma, n_episodios=500)
    resultados_gamma[f"γ={gamma}"] = rets
    print(f"    γ={gamma} → retorno final: {np.mean(rets[-50:]):.2f}")

# Cambio de entorno: sin trampas
print("\n  Cambio de entorno:")
config_sin_trampas = {**CONFIG_BASE, "trampas": {}}
_, rets_st, _ = entrenar(config_sin_trampas, n_episodios=500)
print(f"    Sin trampas              → retorno final: {np.mean(rets_st[-50:]):.2f}")

config_mas_muros = {**CONFIG_BASE, "muros": CONFIG_BASE["muros"] + [(0,3),(1,3),(2,3)]}
_, rets_mm, _ = entrenar(config_mas_muros, n_episodios=500)
print(f"    Con más muros (laberinto)→ retorno final: {np.mean(rets_mm[-50:]):.2f}")

config_r_neg = {**CONFIG_BASE, "r_paso": -1.0}
agente_rneg, rets_rn, _ = entrenar(config_r_neg, n_episodios=500)
print(f"    Penalización paso=-1.0   → retorno final: {np.mean(rets_rn[-50:]):.2f}")

# ─────────────────────────────────────────────
# 9. GRÁFICOS
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  GRÁFICOS")
print("─" * 62)

# --- Figura 1: grilla, política, Q-table y curva de entrenamiento ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

graficar_grilla(axes[0,0], CONFIG_BASE, titulo="GridWorld (sin política)")
graficar_grilla(axes[0,1], CONFIG_BASE, agente_base, titulo="Política aprendida")
graficar_q_table(axes[1,0], CONFIG_BASE, agente_base, titulo="Tabla Q — max(Q) por estado")

ax = axes[1,1]
ax.plot(retornos_base, alpha=0.2, color='steelblue')
ax.plot(range(len(suavizar(retornos_base))),
        suavizar(retornos_base), color='steelblue', lw=2, label='Suavizado')
ax.axhline(np.mean(retornos_base[-50:]), color='tomato', linestyle='--',
           label=f"Media final: {np.mean(retornos_base[-50:]):.2f}")
ax.set_xlabel("Episodio")
ax.set_ylabel("Retorno acumulado")
ax.set_title("Curva de aprendizaje")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Q-Learning en GridWorld — Agente Base", fontsize=13)
plt.tight_layout()
plt.savefig("gridworld_base.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Guardado: gridworld_base.png")

# --- Figura 2: exploración vs explotación ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colores = ['#e74c3c','#2ecc71','#3498db','#9b59b6']
ax = axes[0]
for (label, rets), color in zip(resultados_eps.items(), colores):
    s = suavizar(rets)
    ax.plot(s, label=label, color=color, lw=1.8)
ax.set_xlabel("Episodio")
ax.set_ylabel("Retorno (suavizado)")
ax.set_title("Exploración vs Explotación (ε)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
medias = [np.mean(rets[-50:]) for _, rets in resultados_eps.items()]
etiquetas = [f"ε={e:.2f}" for e, _ in configs_eps]
bars = ax.bar(etiquetas, medias, color=colores)
ax.set_xlabel("Epsilon")
ax.set_ylabel("Retorno medio (últimos 50 ep)")
ax.set_title("Retorno final por estrategia")
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("exploracion_vs_explotacion.png", dpi=100)
plt.close()
print("  Guardado: exploracion_vs_explotacion.png")

# --- Figura 3: efecto de parámetros ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colores_a = ['#e74c3c','#2ecc71','#3498db','#9b59b6']
ax = axes[0]
for (label, rets), color in zip(resultados_alpha.items(), colores_a):
    s = suavizar(rets)
    ax.plot(s, label=label, color=color, lw=1.8)
ax.set_xlabel("Episodio")
ax.set_ylabel("Retorno (suavizado)")
ax.set_title("Efecto de α (tasa de aprendizaje)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for (label, rets), color in zip(resultados_gamma.items(), colores_a):
    s = suavizar(rets)
    ax.plot(s, label=label, color=color, lw=1.8)
ax.set_xlabel("Episodio")
ax.set_ylabel("Retorno (suavizado)")
ax.set_title("Efecto de γ (factor de descuento)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("efecto_parametros.png", dpi=100)
plt.close()
print("  Guardado: efecto_parametros.png")

# --- Figura 4: comparación de entornos ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
configs_env = [
    ("Base",           retornos_base, '#3498db'),
    ("Sin trampas",    rets_st,       '#2ecc71'),
    ("Más muros",      rets_mm,       '#e74c3c'),
    ("Penalización -1",rets_rn,       '#9b59b6'),
]
for label, rets, color in configs_env:
    s = suavizar(rets)
    ax.plot(s, label=label, color=color, lw=1.8)
ax.set_xlabel("Episodio")
ax.set_ylabel("Retorno (suavizado)")
ax.set_title("Efecto de cambios en el entorno")
ax.legend()
ax.grid(True, alpha=0.3)

# política con penalización alta vs base
ax = axes[1]
graficar_grilla(ax, config_r_neg, agente_rneg,
                titulo="Política con penalización paso=-1.0\n(el agente prioriza llegar rápido)")

plt.tight_layout()
plt.savefig("comparacion_entornos.png", dpi=100)
plt.close()
print("  Guardado: comparacion_entornos.png")

# ─────────────────────────────────────────────
# 10. ANÁLISIS DE LA TABLA Q
# ─────────────────────────────────────────────

print("\n" + "─" * 62)
print("  ANÁLISIS DE LA TABLA Q APRENDIDA")
print("─" * 62)

print(f"\n  Estados visitados: {len(agente_base.Q)}")
print(f"\n  {'Estado':<12} {'↑':>8} {'↓':>8} {'←':>8} {'→':>8}  Mejor acción")
print("  " + "-" * 60)

estados_interes = [(5,0),(4,0),(3,0),(3,1),(2,1),(1,0),(0,0),(0,5)]
for s in estados_interes:
    q = agente_base.Q[s]
    mejor = GridWorld.NOMBRES_ACC[int(np.argmax(q))]
    print(f"  {str(s):<12} {q[0]:>8.2f} {q[1]:>8.2f} {q[2]:>8.2f} {q[3]:>8.2f}  {mejor}")

print(f"""
  Lectura de la tabla Q:
    Cada fila es un estado (posición en la grilla)
    Cada columna es el valor estimado de tomar esa acción desde ahí
    El agente elige siempre la acción con mayor Q (argmax)
    Valores altos cerca de la meta, negativos cerca de trampas
""")

print("=" * 62)
print("  RESUMEN")
print("=" * 62)
print(f"  Grilla         : {CONFIG_BASE['filas']}x{CONFIG_BASE['cols']}")
print(f"  Muros          : {len(CONFIG_BASE['muros'])}")
print(f"  Trampas        : {len(CONFIG_BASE['trampas'])}")
print(f"  Episodios      : 500")
print(f"  Parámetros base: α=0.1  γ=0.95  ε=0.1")
print(f"  Retorno final  : {np.mean(retornos_base[-50:]):.2f}")
print(f"  Archivos       : gridworld_base.png")
print(f"                   exploracion_vs_explotacion.png")
print(f"                   efecto_parametros.png")
print(f"                   comparacion_entornos.png")
print("=" * 62)
