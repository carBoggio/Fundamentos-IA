"""
Clasificador de Spam con Naive Bayes
TP Bloque 8: Razonamiento Bajo Incertidumbre

Uso:
  python3 script.py mail1.txt mail2.txt mail3.txt mail4.txt

Si no se pasan argumentos, clasifica todos los .txt del directorio.
"""

import math
import sys
import os
from collections import defaultdict

# ─────────────────────────────────────────────
# DATASET DE ENTRENAMIENTO
# ─────────────────────────────────────────────

EMAILS_ENTRENAMIENTO = [
    ("oferta ganaste dinero gratis premio",                          True),
    ("click aqui oferta limitada dinero facil",                     True),
    ("ganaste un viaje gratis click aqui",                          True),
    ("dinero rapido oferta exclusiva ganaste",                      True),
    ("gratis gratis oferta click premio dinero",                    True),
    ("felicitaciones ganaste premio mayor click",                   True),
    ("oferta increible dinero gratis ahora click",                  True),
    ("premio exclusivo click para reclamar gratis",                 True),
    ("inscripcion gratis ganar dinero sistema facil",               True),
    ("miles personas ganaron dinero click oferta",                  True),

    ("reunion equipo manana a las 10",                              False),
    ("adjunto el informe de la semana",                             False),
    ("podemos hablar del proyecto hoy",                             False),
    ("te mando el documento que pediste",                           False),
    ("confirmo la reunion del jueves",                              False),
    ("revisa el informe adjunto por favor",                         False),
    ("el equipo termino el proyecto a tiempo",                      False),
    ("manana hay reunion de seguimiento del equipo",                False),
    ("revision de tareas para la proxima semana",                   False),
    ("presentacion con el cliente el viernes revisa documento",     False),
]

# ─────────────────────────────────────────────
# MODELO NAIVE BAYES
# ─────────────────────────────────────────────

def tokenizar(texto):
    tokens = texto.lower().split()
    # Eliminar puntuacion basica
    limpios = []
    for t in tokens:
        t = t.strip(".,!?:;()[]\"'")
        if t:
            limpios.append(t)
    return limpios

def entrenar(emails):
    conteo_clase   = defaultdict(int)
    conteo_palabra = defaultdict(lambda: defaultdict(int))
    vocabulario    = set()

    for texto, es_spam in emails:
        clase = "spam" if es_spam else "ham"
        conteo_clase[clase] += 1
        for palabra in tokenizar(texto):
            conteo_palabra[clase][palabra] += 1
            vocabulario.add(palabra)

    total = len(emails)
    V     = len(vocabulario)

    prior = {
        "spam": conteo_clase["spam"] / total,
        "ham":  conteo_clase["ham"]  / total,
    }
    total_palabras = {
        "spam": sum(conteo_palabra["spam"].values()),
        "ham":  sum(conteo_palabra["ham"].values()),
    }

    return prior, conteo_palabra, total_palabras, V

def log_p(palabra, clase, conteo_palabra, total_palabras, V):
    conteo = conteo_palabra[clase].get(palabra, 0)
    return math.log((conteo + 1) / (total_palabras[clase] + V))

def clasificar(texto, prior, conteo_palabra, total_palabras, V):
    palabras = tokenizar(texto)
    scores   = {}

    for clase in ["spam", "ham"]:
        score = math.log(prior[clase])
        for p in palabras:
            score += log_p(p, clase, conteo_palabra, total_palabras, V)
        scores[clase] = score

    # Convertir a probabilidades
    max_s    = max(scores.values())
    exp_vals = {c: math.exp(scores[c] - max_s) for c in scores}
    total_e  = sum(exp_vals.values())
    probs    = {c: exp_vals[c] / total_e for c in exp_vals}

    prediccion = "SPAM" if scores["spam"] > scores["ham"] else "HAM"
    return prediccion, probs["spam"], probs["ham"]

# ─────────────────────────────────────────────
# LEER Y CLASIFICAR ARCHIVOS
# ─────────────────────────────────────────────

def leer_asunto(texto):
    for linea in texto.splitlines():
        if linea.lower().startswith("asunto:"):
            return linea[7:].strip()
    return "(sin asunto)"

def clasificar_archivo(ruta, prior, conteo_palabra, total_palabras, V):
    with open(ruta, encoding="utf-8") as f:
        contenido = f.read()

    asunto     = leer_asunto(contenido)
    pred, ps, ph = clasificar(contenido, prior, conteo_palabra, total_palabras, V)

    palabras   = tokenizar(contenido)
    conocidas  = [p for p in palabras if
                  conteo_palabra["spam"].get(p, 0) + conteo_palabra["ham"].get(p, 0) > 0]
    nuevas     = [p for p in palabras if p not in conocidas]

    # Palabras que mas empujaron hacia spam
    pesos = []
    for p in set(palabras):
        diff = log_p(p, "spam", conteo_palabra, total_palabras, V) \
             - log_p(p, "ham",  conteo_palabra, total_palabras, V)
        pesos.append((p, diff))
    pesos.sort(key=lambda x: x[1], reverse=True)
    top_spam = [f"'{w}'" for w, _ in pesos[:3]]
    top_ham  = [f"'{w}'" for w, _ in pesos[-3:]]

    return {
        "archivo":  os.path.basename(ruta),
        "asunto":   asunto,
        "pred":     pred,
        "p_spam":   ps,
        "p_ham":    ph,
        "palabras": len(palabras),
        "nuevas":   len(set(nuevas)),
        "top_spam": top_spam,
        "top_ham":  top_ham,
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Determinar archivos a clasificar
    if len(sys.argv) > 1:
        archivos = sys.argv[1:]
    else:
        directorio = os.path.dirname(os.path.abspath(__file__))
        archivos = sorted([
            os.path.join(directorio, f)
            for f in os.listdir(directorio)
            if f.startswith("mail") and f.endswith(".txt")
        ])

    if not archivos:
        print("No se encontraron archivos de mail.")
        sys.exit(1)

    # Entrenar modelo
    prior, conteo_palabra, total_palabras, V = entrenar(EMAILS_ENTRENAMIENTO)

    n_spam = sum(1 for _, s in EMAILS_ENTRENAMIENTO if s)
    n_ham  = sum(1 for _, s in EMAILS_ENTRENAMIENTO if not s)

    print("=" * 62)
    print("  CLASIFICADOR DE SPAM — Naive Bayes")
    print("  Bloque 8: Razonamiento Bajo Incertidumbre")
    print("=" * 62)
    print(f"  Modelo entrenado con {len(EMAILS_ENTRENAMIENTO)} emails "
          f"({n_spam} spam / {n_ham} ham)")
    print(f"  Vocabulario: {V} palabras")
    print(f"  Archivos a clasificar: {len(archivos)}")
    print("=" * 62)

    resultados = []
    for ruta in archivos:
        if not os.path.exists(ruta):
            print(f"  [ERROR] No existe: {ruta}")
            continue
        r = clasificar_archivo(ruta, prior, conteo_palabra, total_palabras, V)
        resultados.append(r)

    # Mostrar resultados detallados
    for r in resultados:
        icono = "SPAM" if r["pred"] == "SPAM" else "HAM "
        barra_spam = "#" * int(r["p_spam"] * 20)
        barra_ham  = "#" * int(r["p_ham"]  * 20)

        print(f"\n  Archivo : {r['archivo']}")
        print(f"  Asunto  : {r['asunto']}")
        print(f"  Palabras: {r['palabras']} totales, {r['nuevas']} no vistas en entrenamiento")
        print(f"  Spam    : [{barra_spam:<20}] {r['p_spam']:.1%}")
        print(f"  Ham     : [{barra_ham:<20}] {r['p_ham']:.1%}")
        print(f"  Palabras que mas empujan a SPAM : {', '.join(r['top_spam'])}")
        print(f"  Palabras que mas empujan a HAM  : {', '.join(r['top_ham'])}")
        print(f"  Resultado >>> {icono} <<<")
        print("  " + "-" * 58)

    # Resumen final
    print(f"\n{'=' * 62}")
    print("  RESUMEN")
    print(f"{'=' * 62}")
    for r in resultados:
        marca = "[SPAM]" if r["pred"] == "SPAM" else "[ HAM]"
        print(f"  {marca}  {r['archivo']:<12}  P(spam)={r['p_spam']:.1%}  — {r['asunto'][:38]}")
    print(f"{'=' * 62}")
