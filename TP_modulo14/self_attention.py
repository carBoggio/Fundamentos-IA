"""

Fórmula:
  Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Implementación ────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """
    X   : (seq_len, d_model) — secuencia de embeddings
    W_Q : (d_model, d_k)
    W_K : (d_model, d_k)
    W_V : (d_model, d_v)

    Retorna: salida (seq_len, d_v) y pesos de atención (seq_len, seq_len)
    """
    Q      = X @ W_Q                          # proyección a espacio query
    K      = X @ W_K                          # proyección a espacio key
    V      = X @ W_V                          # proyección a espacio value

    d_k    = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # similitud escalada
    pesos  = softmax(scores)                  # distribución sobre tokens
    salida = pesos @ V                        # combinación ponderada de values

    return Q, K, V, scores, pesos, salida

# ── Ejemplo con una oración ───────────────────────────────────────────────────

tokens  = ["El", "gato", "que", "viste", "ayer", "duerme"]
seq_len = len(tokens)
d_model = 8
d_k     = 4
d_v     = 4

# Embeddings aleatorios (en la práctica son aprendidos)
X   = np.random.randn(seq_len, d_model)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_v) * 0.5

Q, K, V, scores, pesos, salida = self_attention(X, W_Q, W_K, W_V)

# ── Imprimir flujo paso a paso ────────────────────────────────────────────────

print("=" * 62)
print("  EXPERIMENTO 3 — Flujo Q/K/V en Self-Attention")
print("=" * 62)

print(f"\n  Tokens: {tokens}")
print(f"  d_model={d_model}  d_k={d_k}  d_v={d_v}")

print(f"\n  PASO 1 — Proyecciones Q, K, V")
print(f"  Cada token genera tres vectores:")
print(f"    Q (query) : qué busca este token")
print(f"    K (key)   : qué ofrece este token")
print(f"    V (value) : qué información aporta si es seleccionado")
print(f"\n  Q shape: {Q.shape}  K shape: {K.shape}  V shape: {V.shape}")

print(f"\n  PASO 2 — Scores de atención (Q·Kᵀ / √d_k)")
print(f"  Cuánto le 'presta atención' cada token a cada otro:")
print(f"\n  {'':>8}", end="")
for t in tokens:
    print(f"  {t:>8}", end="")
print()
for i, t in enumerate(tokens):
    print(f"  {t:>8}", end="")
    for j in range(seq_len):
        print(f"  {scores[i,j]:>8.2f}", end="")
    print()

print(f"\n  PASO 3 — Pesos de atención (softmax)")
print(f"  Los scores se normalizan → distribución de probabilidad:")
print(f"\n  {'':>8}", end="")
for t in tokens:
    print(f"  {t:>8}", end="")
print()
for i, t in enumerate(tokens):
    print(f"  {t:>8}", end="")
    for j in range(seq_len):
        print(f"  {pesos[i,j]:>8.3f}", end="")
    print()

print(f"\n  PASO 4 — Salida (pesos · V)")
print(f"  Cada token produce un nuevo embedding que es una")
print(f"  combinación ponderada de todos los values.")
print(f"  Salida shape: {salida.shape}")

# Token que más atiende a otros (mayor entropía de atención)
entropias = -np.sum(pesos * np.log(pesos + 1e-9), axis=1)
token_disperso = tokens[entropias.argmax()]
token_enfocado = tokens[entropias.argmin()]
print(f"\n  Token más disperso (atiende a todos por igual): '{token_disperso}'")
print(f"  Token más enfocado (atiende principalmente a uno): '{token_enfocado}'")

print(f"\n  Propiedad clave del self-attention:")
print(f"  Cada token puede atender a CUALQUIER otro en la secuencia")
print(f"  en un solo paso — sin importar la distancia entre ellos.")
print(f"  Esto resuelve el problema de dependencias largas de las RNN.")

# ── Gráficos ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Heatmap de scores
im0 = axes[0].imshow(scores, cmap='RdYlGn', aspect='auto')
axes[0].set_xticks(range(seq_len)); axes[0].set_xticklabels(tokens, rotation=30)
axes[0].set_yticks(range(seq_len)); axes[0].set_yticklabels(tokens)
axes[0].set_title("Scores (Q·Kᵀ / √d_k)\nantes de softmax")
plt.colorbar(im0, ax=axes[0])

# Heatmap de pesos (después de softmax)
im1 = axes[1].imshow(pesos, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[1].set_xticks(range(seq_len)); axes[1].set_xticklabels(tokens, rotation=30)
axes[1].set_yticks(range(seq_len)); axes[1].set_yticklabels(tokens)
axes[1].set_title("Pesos de atención\n(después de softmax)")
plt.colorbar(im1, ax=axes[1])
for i in range(seq_len):
    for j in range(seq_len):
        axes[1].text(j, i, f"{pesos[i,j]:.2f}", ha='center', va='center',
                     fontsize=7, color='black' if pesos[i,j] < 0.6 else 'white')

# Diagrama del flujo Q/K/V
ax = axes[2]
ax.axis('off')
ax.set_title("Flujo Q / K / V", fontsize=11)
pasos = [
    ("Input X", "Embeddings de entrada\n(seq_len × d_model)", 0.85, '#3498db'),
    ("Q  K  V", "3 proyecciones lineales\n(aprend. durante training)", 0.65, '#9b59b6'),
    ("Q·Kᵀ/√d_k", "Scores de similitud\n(cada token vs todos)", 0.45, '#e67e22'),
    ("softmax(·)", "Normalización → pesos\n(distribución de atención)", 0.25, '#2ecc71'),
    ("pesos · V", "Salida: nuevo embedding\n(mezcla contextualizada)", 0.05, '#e74c3c'),
]
for label, desc, y, color in pasos:
    ax.add_patch(plt.Rectangle((0.1, y-0.07), 0.8, 0.13,
                                facecolor=color, alpha=0.2, edgecolor=color, lw=1.5))
    ax.text(0.5, y, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=color)
    ax.text(0.5, y-0.04, desc, ha='center', va='center', fontsize=7, color='gray')
    if y > 0.1:
        ax.annotate("", xy=(0.5, y-0.08), xytext=(0.5, y-0.13),
                    arrowprops=dict(arrowstyle="->", color='gray'))

plt.suptitle("Self-Attention — Flujo Q/K/V paso a paso", fontsize=12)
plt.tight_layout()
plt.savefig("self_attention.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n  Guardado: self_attention.png")
