"""
Naive Bayes para clasificación de Spam
TP Bloque 8: Razonamiento Bajo Incertidumbre

Idea central:
  P(Spam | palabras) ∝ P(Spam) × ∏ P(palabra_i | Spam)

Dado un email nuevo, calculamos la probabilidad de que sea spam
y de que sea legítimo (ham), y elegimos la mayor.
"""

import math
from collections import defaultdict

# ─────────────────────────────────────────────
# PASO 1: Dataset de entrenamiento
# ─────────────────────────────────────────────
# Emails etiquetados manualmente: (texto, es_spam)

EMAILS = [
    # SPAM
    ("oferta ganaste dinero gratis premio",          True),
    ("click aqui oferta limitada dinero facil",      True),
    ("ganaste un viaje gratis click aqui",           True),
    ("dinero rapido oferta exclusiva ganaste",       True),
    ("gratis gratis oferta click premio dinero",     True),
    ("felicitaciones ganaste premio mayor click",    True),
    ("oferta increible dinero gratis ahora click",   True),
    ("premio exclusivo click para reclamar gratis",  True),

    # HAM (legítimo)
    ("reunion equipo manana a las 10",               False),
    ("adjunto el informe de la semana",              False),
    ("podemos hablar del proyecto hoy",              False),
    ("te mando el documento que pediste",            False),
    ("confirmo la reunion del jueves",               False),
    ("revisa el informe adjunto por favor",          False),
    ("el equipo termino el proyecto a tiempo",       False),
    ("manana hay reunion de seguimiento del equipo", False),
]

# ─────────────────────────────────────────────
# PASO 2: Entrenamiento
# ─────────────────────────────────────────────

def tokenizar(texto):
    return texto.lower().split()

def entrenar(emails):
    """
    Calcula:
      - P(Spam) y P(Ham): probabilidades prior
      - P(palabra | Spam) y P(palabra | Ham): verosimilitudes

    Usa suavizado de Laplace (+1 a cada conteo) para evitar
    probabilidad cero en palabras no vistas durante entrenamiento.
    """
    conteo_clase  = defaultdict(int)    # cuántos emails por clase
    conteo_palabra = defaultdict(lambda: defaultdict(int))  # palabra → clase → conteo
    vocabulario   = set()

    for texto, es_spam in emails:
        clase = "spam" if es_spam else "ham"
        conteo_clase[clase] += 1
        for palabra in tokenizar(texto):
            conteo_palabra[clase][palabra] += 1
            vocabulario.add(palabra)

    total = len(emails)
    V     = len(vocabulario)   # tamaño del vocabulario (para suavizado Laplace)

    # Probabilidades prior: P(Spam) y P(Ham)
    prior = {
        "spam": conteo_clase["spam"] / total,
        "ham":  conteo_clase["ham"]  / total,
    }

    # Total de palabras por clase (para normalizar verosimilitudes)
    total_palabras = {
        "spam": sum(conteo_palabra["spam"].values()),
        "ham":  sum(conteo_palabra["ham"].values()),
    }

    return prior, conteo_palabra, total_palabras, V, vocabulario

# ─────────────────────────────────────────────
# PASO 3: Clasificación
# ─────────────────────────────────────────────

def log_verosimilitud(palabra, clase, conteo_palabra, total_palabras, V):
    """
    P(palabra | clase) con suavizado de Laplace:
      (conteo + 1) / (total_palabras_clase + V)

    Usamos logaritmo para evitar underflow numérico
    (multiplicar muchos números pequeños lleva a cero en punto flotante).
    """
    conteo = conteo_palabra[clase].get(palabra, 0)
    return math.log((conteo + 1) / (total_palabras[clase] + V))

def clasificar(texto, prior, conteo_palabra, total_palabras, V, verbose=True):
    """
    Aplica Naive Bayes:
      log P(clase | texto) ∝ log P(clase) + Σ log P(palabra | clase)

    Retorna la clase con mayor probabilidad posterior.
    """
    palabras = tokenizar(texto)

    if verbose:
        print(f"\nEmail: '{texto}'")
        print(f"Palabras: {palabras}")
        print(f"\n{'Palabra':<15} {'log P(·|SPAM)':<20} {'log P(·|HAM)'}")
        print("-" * 55)

    scores = {}
    for clase in ["spam", "ham"]:
        # Empezamos con el log del prior
        score = math.log(prior[clase])
        for palabra in palabras:
            lv = log_verosimilitud(palabra, clase, conteo_palabra, total_palabras, V)
            score += lv

            if verbose and clase == "spam":
                lv_ham = log_verosimilitud(palabra, "ham", conteo_palabra, total_palabras, V)
                print(f"  {palabra:<13} {lv:<20.4f} {lv_ham:.4f}")

        scores[clase] = score

    if verbose:
        print(f"\n  log P(spam | email) = {scores['spam']:.4f}")
        print(f"  log P(ham  | email) = {scores['ham']:.4f}")

    # Convertir log-scores a probabilidades reales (softmax sobre 2 clases)
    max_score = max(scores.values())
    exp_spam  = math.exp(scores["spam"] - max_score)
    exp_ham   = math.exp(scores["ham"]  - max_score)
    total_exp = exp_spam + exp_ham

    prob_spam = exp_spam / total_exp
    prob_ham  = exp_ham  / total_exp

    prediccion = "SPAM" if scores["spam"] > scores["ham"] else "HAM"

    if verbose:
        print(f"\n  P(spam | email) ≈ {prob_spam:.1%}")
        print(f"  P(ham  | email) ≈ {prob_ham:.1%}")
        print(f"\n  Clasificacion: {prediccion}")

    return prediccion, prob_spam

# ─────────────────────────────────────────────
# PASO 4: Evaluación
# ─────────────────────────────────────────────

def evaluar(emails_test, prior, conteo_palabra, total_palabras, V):
    """
    Corre el clasificador sobre un conjunto de test y muestra
    aciertos, fallos y exactitud total.
    """
    print("\n" + "=" * 60)
    print("EVALUACION SOBRE EMAILS DE TEST")
    print("=" * 60)

    correctos = 0
    for texto, es_spam in emails_test:
        etiqueta_real = "SPAM" if es_spam else "HAM"
        prediccion, prob = clasificar(texto, prior, conteo_palabra, total_palabras, V, verbose=False)
        acierto = "OK" if prediccion == etiqueta_real else "FALLO"
        correctos += (prediccion == etiqueta_real)
        print(f"  [{acierto}] Real={etiqueta_real:<4}  Pred={prediccion:<4}  P(spam)={prob:.1%}  '{texto[:45]}'")

    exactitud = correctos / len(emails_test)
    print(f"\n  Exactitud: {correctos}/{len(emails_test)} = {exactitud:.0%}")

# ─────────────────────────────────────────────
# PASO 5: Mostrar lo aprendido
# ─────────────────────────────────────────────

def mostrar_modelo(prior, conteo_palabra, total_palabras, V, vocabulario):
    print("\n" + "=" * 60)
    print("LO QUE APRENDIO EL MODELO")
    print("=" * 60)

    print(f"\n  Prior P(spam) = {prior['spam']:.2f}")
    print(f"  Prior P(ham)  = {prior['ham']:.2f}")

    print("\n  Palabras mas indicativas de SPAM (mayor P(palabra|spam)):")
    palabras_spam = sorted(
        vocabulario,
        key=lambda w: conteo_palabra["spam"].get(w, 0) / total_palabras["spam"],
        reverse=True
    )
    for w in palabras_spam[:6]:
        p = (conteo_palabra["spam"].get(w, 0) + 1) / (total_palabras["spam"] + V)
        print(f"    '{w}' → P(·|spam) = {p:.3f}")

    print("\n  Palabras mas indicativas de HAM (mayor P(palabra|ham)):")
    palabras_ham = sorted(
        vocabulario,
        key=lambda w: conteo_palabra["ham"].get(w, 0) / total_palabras["ham"],
        reverse=True
    )
    for w in palabras_ham[:6]:
        p = (conteo_palabra["ham"].get(w, 0) + 1) / (total_palabras["ham"] + V)
        print(f"    '{w}' → P(·|ham)  = {p:.3f}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NAIVE BAYES PARA DETECCION DE SPAM")
    print("Bloque 8: Razonamiento Bajo Incertidumbre")
    print("=" * 60)

    # Entrenamiento
    prior, conteo_palabra, total_palabras, V, vocabulario = entrenar(EMAILS)
    print(f"\nModelo entrenado con {len(EMAILS)} emails ({sum(1 for _,s in EMAILS if s)} spam, {sum(1 for _,s in EMAILS if not s)} ham)")
    print(f"Vocabulario: {V} palabras unicas")

    # Mostrar lo aprendido
    mostrar_modelo(prior, conteo_palabra, total_palabras, V, vocabulario)

    # Clasificar emails nuevos con traza completa
    print("\n" + "=" * 60)
    print("CLASIFICACION DE EMAILS NUEVOS (con traza)")
    print("=" * 60)

    emails_nuevos = [
        "oferta gratis click aqui ahora",
        "reunion de equipo el lunes",
        "ganaste dinero llama ahora",
        "adjunto el informe del proyecto",
        "gratis dinero oferta proyecto equipo",   # palabras de ambas clases
    ]

    for email in emails_nuevos:
        clasificar(email, prior, conteo_palabra, total_palabras, V, verbose=True)
        print()

    # Evaluación sobre los mismos emails de entrenamiento (sanity check)
    evaluar(EMAILS, prior, conteo_palabra, total_palabras, V)
