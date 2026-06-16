# TP Bloque 13 — Fundamentos de Redes Neuronales

MLP sobre XOR, Moons y Circles con visualización de fronteras de decisión, curvas de pérdida, overfitting y regularización.

## Cómo correrlo

```bash
cd TP_modulo13
venv/bin/python redes_neuronales.py
```

## Archivos

| Archivo | Descripción |
|---|---|
| `redes_neuronales.py` | Script principal |
| `xor.png` | Perceptrón lineal vs MLP en el problema XOR |
| `fronteras_decision.png` | Fronteras de decisión en Moons y Circles con distintas arquitecturas |
| `overfitting_regularizacion.png` | Efecto de L2 sobre una red grande |
| `learning_rate.png` | Curvas de pérdida para distintos learning rates |
| `activaciones.png` | Sigmoid, Tanh y ReLU graficadas |
| `requirements.txt` | Dependencias |
| `venv/` | Entorno virtual |

## Recrear el entorno

```bash
cd TP_modulo13
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```
