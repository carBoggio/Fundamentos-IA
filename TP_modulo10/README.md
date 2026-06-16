# TP Bloque 10 — Aprendizaje Supervisado: Clasificación

Comparación de Árbol de Decisión, Random Forest y XGBoost sobre el dataset Titanic.

## Cómo correrlo

```bash
cd TP_modulo10
venv/bin/python clasificacion.py
```

## Archivos

| Archivo | Descripción |
|---|---|
| `clasificacion.py` | Script principal |
| `confusion_matrices.png` | Generado al correr — matrices de confusión de los 3 modelos |
| `curvas_roc.png` | Generado al correr — curvas ROC comparativas |
| `feature_importance.png` | Generado al correr — importancia de features en RF y XGBoost |
| `requirements.txt` | Dependencias del entorno |
| `venv/` | Entorno virtual con las dependencias |

## Recrear el entorno

```bash
cd TP_modulo10
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```
