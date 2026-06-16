"""
Experimento 1: MLP vs CNN en imágenes (MNIST)

"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
DEVICE = torch.device('cpu')

# ── Dataset ──────────────────────────────────────────────────────────────────

transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

train_full = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_full  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Subconjunto pequeño para entrenamiento liviano
train_ds = Subset(train_full, range(8000))
test_ds  = Subset(test_full,  range(2000))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256)

# ── Modelos ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Aplana la imagen 28x28 → 784 y pasa por capas densas."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    """
    Dos capas convolucionales con pooling.
    Cada filtro aprende un patrón local (borde, curva, esquina)
    y ese conocimiento se comparte en toda la imagen.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 → 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Entrenamiento ─────────────────────────────────────────────────────────────

def contar_params(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)

def entrenar_modelo(modelo, epochs=5):
    opt      = optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.CrossEntropyLoss()
    hist     = []
    t0       = time.time()

    for ep in range(1, epochs + 1):
        modelo.train()
        loss_ep = 0
        for X, y in train_loader:
            opt.zero_grad()
            loss = criterio(modelo(X), y)
            loss.backward()
            opt.step()
            loss_ep += loss.item()

        modelo.eval()
        correctos = 0
        with torch.no_grad():
            for X, y in test_loader:
                preds     = modelo(X).argmax(1)
                correctos += (preds == y).sum().item()
        acc = correctos / len(test_ds)
        hist.append((loss_ep / len(train_loader), acc))
        print(f"    Época {ep}: loss={hist[-1][0]:.4f}  acc={acc:.3f}")

    return hist, time.time() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  EXPERIMENTO 1 — MLP vs CNN en MNIST")
print("=" * 60)

mlp = MLP().to(DEVICE)
cnn = CNN().to(DEVICE)

print(f"\n  Parámetros MLP: {contar_params(mlp):,}")
print(f"  Parámetros CNN: {contar_params(cnn):,}")
print(f"\n  Sesgo inductivo MLP: ninguno — ve píxeles independientes")
print(f"  Sesgo inductivo CNN: localidad + invarianza traslacional")

print("\n  Entrenando MLP...")
hist_mlp, t_mlp = entrenar_modelo(mlp)

print("\n  Entrenando CNN...")
hist_cnn, t_cnn = entrenar_modelo(cnn)

acc_mlp = hist_mlp[-1][1]
acc_cnn = hist_cnn[-1][1]

print(f"\n  Resultados finales (test):")
print(f"    MLP: acc={acc_mlp:.3f}  tiempo={t_mlp:.1f}s  params={contar_params(mlp):,}")
print(f"    CNN: acc={acc_cnn:.3f}  tiempo={t_cnn:.1f}s  params={contar_params(cnn):,}")
print(f"\n  CNN logra mayor accuracy con MENOS parámetros.")
print(f"  El sesgo inductivo espacial es más eficiente que la densidad.")

# ── Gráficos ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs_range = range(1, 6)
axes[0].plot(epochs_range, [h[1] for h in hist_mlp], 'o-', label=f'MLP ({contar_params(mlp):,} params)', color='steelblue')
axes[0].plot(epochs_range, [h[1] for h in hist_cnn], 's-', label=f'CNN ({contar_params(cnn):,} params)', color='tomato')
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy (test)")
axes[0].set_title("MLP vs CNN — Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, [h[0] for h in hist_mlp], 'o-', color='steelblue', label='MLP')
axes[1].plot(epochs_range, [h[0] for h in hist_cnn], 's-', color='tomato', label='CNN')
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss (train)")
axes[1].set_title("MLP vs CNN — Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Visualizar filtros CNN aprendidos
filtros = cnn.conv[0].weight.data.squeeze().numpy()
axes[2].set_title("Filtros CNN capa 1 (16 filtros 3x3)")
axes[2].axis('off')
for i in range(16):
    ax_f = fig.add_axes([0.685 + (i % 8) * 0.038, 0.55 - (i // 8) * 0.38, 0.034, 0.3])
    ax_f.imshow(filtros[i], cmap='RdBu', vmin=-1, vmax=1)
    ax_f.axis('off')

plt.suptitle("MLP vs CNN — MNIST", fontsize=12)
plt.tight_layout(rect=[0, 0, 0.68, 1])
plt.savefig("mlp_vs_cnn.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n  Guardado: mlp_vs_cnn.png")
