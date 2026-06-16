"""
Experimento 2: RNN vs LSTM en series temporales

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ── Dataset: dependencia larga ────────────────────────────────────────────────

def generar_datos(n=2000, largo=50):
    """
    Secuencia de largo L.
    La señal en posición 0 determina la etiqueta al final:
      posicion[0] > 0 → etiqueta 1
      posicion[0] < 0 → etiqueta 0
    El resto son valores aleatorios con ruido fuerte para dificultar
    que la RNN propague el gradiente hasta el paso 0.
    """
    X = np.random.randn(n, largo, 1).astype(np.float32)
    # Ruido fuerte en el resto de la secuencia → el gradiente debe
    # viajar 50 pasos hacia atrás, lo que colapsa en RNN vanilla
    X[:, 1:, :] = np.random.randn(n, largo - 1, 1)
    y = (X[:, 0, 0] > 0).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

X, y = generar_datos()
split = 1600
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

# ── Modelos ───────────────────────────────────────────────────────────────────

class RNNClasificador(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0)).squeeze(1)

class LSTMClasificador(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0)).squeeze(1)

# ── Entrenamiento ─────────────────────────────────────────────────────────────

def entrenar(modelo, epochs=30, batch=64):
    opt      = optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.BCEWithLogitsLoss()
    hist_loss, hist_acc = [], []

    for ep in range(epochs):
        modelo.train()
        idx     = torch.randperm(len(X_train))
        ep_loss = 0
        for i in range(0, len(X_train), batch):
            b     = idx[i:i+batch]
            out   = modelo(X_train[b])
            loss  = criterio(out, y_train[b])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        modelo.eval()
        with torch.no_grad():
            preds = (modelo(X_test) > 0).float()
            acc   = (preds == y_test).float().mean().item()
        hist_loss.append(ep_loss / (len(X_train) // batch))
        hist_acc.append(acc)

    return hist_loss, hist_acc

print("=" * 60)
print("  EXPERIMENTO 2 — RNN vs LSTM en series temporales")
print("=" * 60)
print(f"\n  Secuencias de largo 30. La señal en t=0 determina la etiqueta.")
print(f"  El modelo debe recordarla durante 29 pasos más.")
print(f"  Entrenando RNN y LSTM (30 épocas)...\n")

rnn  = RNNClasificador()
lstm = LSTMClasificador()

loss_rnn,  acc_rnn  = entrenar(rnn)
loss_lstm, acc_lstm = entrenar(lstm)

print(f"  RNN  — acc final: {acc_rnn[-1]:.3f}  (baseline aleatorio: 0.500)")
print(f"  LSTM — acc final: {acc_lstm[-1]:.3f}")
print(f"\n  La LSTM retiene la señal inicial gracias a la celda de memoria.")
print(f"  La RNN la olvida por el vanishing gradient en secuencias largas.")

# ── Gráficos ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(acc_rnn,  label='RNN',  color='steelblue', lw=2)
axes[0].plot(acc_lstm, label='LSTM', color='tomato',    lw=2)
axes[0].axhline(0.5, color='gray', linestyle='--', label='Baseline aleatorio')
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy (test)")
axes[0].set_title("RNN vs LSTM — Dependencia larga (t=0)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss_rnn,  label='RNN',  color='steelblue', lw=2)
axes[1].plot(loss_lstm, label='LSTM', color='tomato',    lw=2)
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss (train)")
axes[1].set_title("RNN vs LSTM — Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rnn_vs_lstm.png", dpi=100)
plt.close()
print("\n  Guardado: rnn_vs_lstm.png")
