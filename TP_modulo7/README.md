# TP Bloque 8 — Razonamiento Bajo Incertidumbre

Clasificador de spam usando Naive Bayes implementado desde cero en Python.

## Requisitos

Python 3.x, sin dependencias externas.

## Archivos

| Archivo | Descripción |
|---|---|
| `script.py` | Clasificador principal — leer y ejecutar este |
| `naive_bayes_spam.py` | Implementación base con traza detallada del algoritmo |
| `mail1.txt` | Email de ejemplo (spam) |
| `mail2.txt` | Email de ejemplo (ham) |
| `mail3.txt` | Email de ejemplo (spam) |
| `mail4.txt` | Email de ejemplo (ham) |

## Cómo correrlo

Clasificar todos los mails del directorio:

```bash
python3 script.py
```

Clasificar archivos específicos:

```bash
python3 script.py mail1.txt mail2.txt
```

Clasificar un mail propio:

```bash
python3 script.py mi_mail.txt
```

Ver la traza interna del algoritmo (cómo razona palabra por palabra):

```bash
python3 naive_bayes_spam.py
```

## Agregar mails propios

Creá un archivo `.txt` con este formato y pasalo como argumento:

```
Asunto: El texto del asunto va acá

Cuerpo del mail acá.
```

## Modificar el modelo

El conjunto de entrenamiento está al inicio de `script.py` en la variable `EMAILS_ENTRENAMIENTO`. Cada entrada es una tupla `("texto del mail", True/False)` donde `True` = spam y `False` = ham. Agregar más ejemplos mejora la precisión.
