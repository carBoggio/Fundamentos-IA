# TP Bloque 12 — Aprendizaje por Refuerzo: Q-Learning en GridWorld

Implementación de un agente Q-Learning en un entorno GridWorld 6x6 con muros, trampas y meta. Incluye análisis de la tabla Q, comparación exploración vs explotación, y experimentos con distintos parámetros y entornos.

## Cómo correrlo

```bash
cd TP_modulo12
venv/bin/python qlearning_gridworld.py
```

## Archivos

| Archivo | Descripción |
|---|---|
| `qlearning_gridworld.py` | Script principal |
| `gridworld_base.png` | Grilla, política aprendida, tabla Q y curva de aprendizaje |
| `exploracion_vs_explotacion.png` | Comparación de distintos valores de ε |
| `efecto_parametros.png` | Efecto de α y γ sobre el retorno |
| `comparacion_entornos.png` | Efecto de cambiar muros, trampas y penalizaciones |
| `requirements.txt` | Dependencias del entorno |
| `venv/` | Entorno virtual |

## Recrear el entorno

```bash
cd TP_modulo12
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```
