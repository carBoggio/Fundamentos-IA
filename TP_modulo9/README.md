# TP Bloque 9 — Aprendizaje Supervisado: Regresión

Regresión sobre el dataset California Housing con diagnóstico de overfitting, evaluación train/validation/test y regularización.

## Requisitos

Las dependencias están instaladas en un entorno virtual local. No se necesita instalar nada globalmente.

## Cómo correrlo

```bash
cd TP_modulo9
venv/bin/python regresion.py
```

## Archivos

| Archivo | Descripción |
|---|---|
| `regresion.py` | Script principal |
| `curvas_aprendizaje.png` | Generado al correr — curvas de train vs validation |
| `prediccion_vs_real.png` | Generado al correr — predicción vs precio real en test |
| `venv/` | Entorno virtual con las dependencias |

## Dependencias

Instaladas dentro de `venv/`:

- `scikit-learn`
- `numpy`
- `matplotlib`
- `pandas`

Si por algún motivo el entorno no existe, recrearlo con:

```bash
cd TP_modulo9
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```
