# TP Bloque 14 — Arquitecturas Modernas e IA Híbrida

Cuatro experimentos sobre arquitecturas de redes neuronales modernas.

## Experimentos

| Script | Qué hace |
|---|---|
| `mlp_vs_cnn.py` | Compara MLP y CNN en MNIST — muestra el efecto del sesgo inductivo espacial |
| `rnn_vs_lstm.py` | Compara RNN y LSTM en series con dependencia larga — muestra el vanishing gradient |
| `self_attention.py` | Implementa self-attention desde cero en numpy — muestra el flujo Q/K/V paso a paso |
| `rag_analisis.py` | Implementa un sistema RAG con TF-IDF — compara conocimiento paramétrico vs recuperado |

## Cómo correrlo

Cada experimento es independiente:

```bash
cd TP_modulo14

venv/bin/python self_attention.py       # no necesita GPU, solo numpy
venv/bin/python rag_analisis.py         # no necesita GPU
venv/bin/python rnn_vs_lstm.py          # PyTorch CPU, ~30 segundos
venv/bin/python mlp_vs_cnn.py           # PyTorch CPU + descarga MNIST, ~2 minutos
```

## Archivos generados

| Archivo | Descripción |
|---|---|
| `mlp_vs_cnn.png` | Accuracy, loss y filtros aprendidos por la CNN |
| `rnn_vs_lstm.png` | Accuracy y loss comparativo RNN vs LSTM |
| `self_attention.png` | Heatmaps de scores y pesos de atención + diagrama de flujo |
| `rag_analisis.png` | Heatmap de retrieval scores + diagrama del sistema RAG |

## Recrear el entorno

```bash
cd TP_modulo14
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

> Los datos de MNIST se descargan automáticamente en `./data/` la primera vez que se corre `mlp_vs_cnn.py`.
