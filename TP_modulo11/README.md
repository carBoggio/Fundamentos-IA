# TP Bloque 11 — Aprendizaje No Supervisado: Datos Estelares GAIA

Análisis no supervisado de datos estelares reales del archivo GAIA mediante clustering, PCA y detección de anomalías.

## Cómo correrlo

```bash
cd TP_modulo11
venv/bin/python analisis_estelar.py
```

El script intenta descargar datos reales del archivo GAIA automáticamente. Si no hay conexión, genera un dataset sintético equivalente.

## Archivos

| Archivo | Descripción |
|---|---|
| `analisis_estelar.py` | Script principal |
| `analisis_estelar.png` | Generado al correr — 6 gráficos: HR diagram, PCA, codo, silueta, DBSCAN, anomalías |
| `pca_varianza.png` | Generado al correr — varianza explicada por componente |
| `requirements.txt` | Dependencias del entorno |
| `venv/` | Entorno virtual con las dependencias |

## Recrear el entorno

```bash
cd TP_modulo11
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```
