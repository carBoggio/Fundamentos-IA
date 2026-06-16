"""
Experimento 4: Análisis de RAG (Retrieval-Augmented Generation)


"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Base de conocimiento ──────────────────────────────────────────────────────

BASE_CONOCIMIENTO = [
    {"id": 1, "titulo": "Redes neuronales convolucionales",
     "texto": "Las CNN aplican filtros locales que se comparten en toda la imagen. "
              "Esto les da invarianza a la traslación y reduce el número de parámetros "
              "comparado con capas densas. Son el estándar para visión por computadora."},

    {"id": 2, "titulo": "Transformers y self-attention",
     "texto": "Los Transformers usan self-attention para que cada token atienda a todos "
              "los otros tokens de la secuencia en paralelo. Esto elimina la recurrencia "
              "y permite capturar dependencias largas sin vanishing gradient."},

    {"id": 3, "titulo": "LSTM y compuertas",
     "texto": "Las LSTM resuelven el vanishing gradient de las RNN mediante compuertas: "
              "forget gate (qué olvidar), input gate (qué escribir) y output gate (qué leer). "
              "La celda de memoria puede retener información durante muchos pasos."},

    {"id": 4, "titulo": "Overfitting y regularización",
     "texto": "El overfitting ocurre cuando el modelo memoriza el entrenamiento y no generaliza. "
              "Se controla con dropout, weight decay (L2), early stopping y data augmentation. "
              "La regularización penaliza parámetros grandes para simplificar el modelo."},

    {"id": 5, "titulo": "Aprendizaje por transferencia",
     "texto": "El fine-tuning toma un modelo preentrenado en datos masivos y lo adapta "
              "a una tarea específica con pocos datos. El modelo ya aprendió representaciones "
              "generales que se transfieren a la tarea objetivo."},

    {"id": 6, "titulo": "Modelos generativos y discriminativos",
     "texto": "Los modelos discriminativos aprenden P(y|x) directamente. "
              "Los generativos aprenden P(x,y) o P(x) y pueden generar muestras nuevas. "
              "GPT es generativo, BERT es discriminativo en su pretraining."},

    {"id": 7, "titulo": "Límites del deep learning",
     "texto": "El deep learning tiene limitaciones en razonamiento lógico, causalidad "
              "y consistencia. Los modelos pueden aluciinar datos no visto en entrenamiento. "
              "No tienen conocimiento editable ni trazabilidad de fuentes."},

    {"id": 8, "titulo": "RAG: Retrieval-Augmented Generation",
     "texto": "RAG combina un recuperador (retriever) y un generador. El retriever busca "
              "documentos relevantes en una base de conocimiento externa. El generador "
              "produce la respuesta condicionada al contexto recuperado. Reduce alucinaciones "
              "y permite actualizar el conocimiento sin reentrenar el modelo."},
]

# ── Recuperador TF-IDF ────────────────────────────────────────────────────────

class RecuperadorTFIDF:
    """
    Recuperador basado en similitud TF-IDF.
    En sistemas RAG reales se usan embeddings densos (dense retrieval).
    TF-IDF es más simple pero ilustra el mismo principio.
    """
    def __init__(self, documentos):
        self.docs = documentos
        textos    = [d["texto"] + " " + d["titulo"] for d in documentos]
        self.vec  = TfidfVectorizer(min_df=1)
        self.mat  = self.vec.fit_transform(textos)

    def recuperar(self, pregunta, top_k=2):
        q_vec  = self.vec.transform([pregunta])
        scores = cosine_similarity(q_vec, self.mat)[0]
        top    = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], scores[i]) for i in top]

# ── Generador basado en templates ─────────────────────────────────────────────

def generar_con_contexto(pregunta, contextos):
    """Simula la generación condicionada al contexto recuperado."""
    if not contextos or contextos[0][1] < 0.05:
        return (f"[SIN CONTEXTO] No encuentro información relevante sobre '{pregunta}'. "
                f"Esta sería una alucinación potencial del modelo.")

    resp = f"Basándome en la documentación recuperada:\n\n"
    for doc, score in contextos:
        resp += f"  [{doc['titulo']} — relevancia: {score:.3f}]\n"
        resp += f"  {doc['texto']}\n\n"
    return resp.strip()

def generar_sin_contexto(pregunta):
    """Simula generación puramente paramétrica (sin recuperación)."""
    return (f"[CONOCIMIENTO PARAMÉTRICO] El modelo respondería desde lo que aprendió "
            f"durante el entrenamiento sobre '{pregunta}', sin poder citar fuentes "
            f"específicas ni garantizar actualidad de la información.")

# ── Análisis de RAG ───────────────────────────────────────────────────────────

print("=" * 62)
print("  EXPERIMENTO 4 — Análisis de RAG")
print("=" * 62)

recuperador = RecuperadorTFIDF(BASE_CONOCIMIENTO)

preguntas = [
    "¿Cómo funcionan las redes convolucionales?",
    "¿Por qué los transformers son mejores que las RNN?",
    "¿Cómo evitar el overfitting en redes neuronales?",
    "¿Cuál es la diferencia entre RAG y fine-tuning?",
    "¿Cuántos planetas tiene el sistema solar?",   # fuera del dominio
]

print(f"\n  Base de conocimiento: {len(BASE_CONOCIMIENTO)} documentos")
print(f"  Recuperador: TF-IDF + similitud coseno")

scores_matrix = np.zeros((len(preguntas), len(BASE_CONOCIMIENTO)))

for i, pregunta in enumerate(preguntas):
    print(f"\n{'─'*62}")
    print(f"  PREGUNTA: {pregunta}")
    print(f"{'─'*62}")

    contextos = recuperador.recuperar(pregunta, top_k=2)
    for doc, score in contextos:
        scores_matrix[i, doc['id']-1] = score

    print(f"\n  Documentos recuperados:")
    for doc, score in contextos:
        print(f"    [{score:.3f}] {doc['titulo']}")

    print(f"\n  Respuesta SIN recuperación (paramétrica):")
    print(f"  {generar_sin_contexto(pregunta)}")

    print(f"\n  Respuesta CON recuperación (RAG):")
    respuesta_rag = generar_con_contexto(pregunta, contextos)
    for linea in respuesta_rag.split('\n'):
        print(f"  {linea}")

# ── Comparación paramétrico vs RAG ────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"  COMPARACIÓN: Conocimiento paramétrico vs RAG")
print(f"{'='*62}")

tabla = [
    ("Fuente de conocimiento", "Pesos del modelo (fijo)", "Base de datos externa (actualizable)"),
    ("Actualización",          "Requiere reentrenar",      "Editar o agregar documentos"),
    ("Trazabilidad",           "No cita fuentes",          "Cita documento recuperado"),
    ("Alucinaciones",          "Mayor riesgo",             "Reducidas por el contexto"),
    ("Conocimiento reciente",  "Solo hasta el cutoff",     "Acceso a info actualizada"),
    ("Costo en inferencia",    "Solo forward pass",        "Retrieval + forward pass"),
]

print(f"\n  {'Dimensión':<25} {'Paramétrico':^22} {'RAG':^22}")
print(f"  {'─'*70}")
for dim, param, rag in tabla:
    print(f"  {dim:<25} {param:^22} {rag:^22}")

# ── Gráficos ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap de retrieval scores
titulos_cortos = [d['titulo'][:20] + '...' if len(d['titulo']) > 20
                  else d['titulo'] for d in BASE_CONOCIMIENTO]
preguntas_cortas = [p[:35] + '...' if len(p) > 35 else p for p in preguntas]

im = axes[0].imshow(scores_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
axes[0].set_xticks(range(len(BASE_CONOCIMIENTO)))
axes[0].set_xticklabels(titulos_cortos, rotation=45, ha='right', fontsize=7)
axes[0].set_yticks(range(len(preguntas)))
axes[0].set_yticklabels(preguntas_cortas, fontsize=8)
axes[0].set_title("Scores de retrieval (TF-IDF)\nPregunta vs Documento")
plt.colorbar(im, ax=axes[0])

# Diagrama del flujo RAG
ax = axes[1]
ax.axis('off')
ax.set_title("Flujo de un sistema RAG", fontsize=11)

componentes = [
    (0.5, 0.88, "PREGUNTA DEL USUARIO", '#3498db', 0.7),
    (0.5, 0.68, "RECUPERADOR (Retriever)", '#9b59b6', 0.7),
    (0.15, 0.48, "BASE DE\nCONOCIMIENTO", '#2ecc71', 0.25),
    (0.5, 0.48, "DOCUMENTOS\nRELEVANTES", '#e67e22', 0.25),
    (0.5, 0.28, "GENERADOR (LLM)", '#e74c3c', 0.7),
    (0.5, 0.08, "RESPUESTA FUNDAMENTADA", '#1abc9c', 0.7),
]
for x, y, label, color, w in componentes:
    ax.add_patch(plt.Rectangle((x-w/2, y-0.07), w, 0.12,
                                facecolor=color, alpha=0.2,
                                edgecolor=color, lw=1.5, transform=ax.transAxes))
    ax.text(x, y, label, ha='center', va='center', fontsize=8,
            fontweight='bold', color=color, transform=ax.transAxes)

flechas = [(0.5,0.81,0.5,0.76), (0.5,0.61,0.5,0.56),
           (0.15,0.55,0.35,0.56), (0.5,0.41,0.5,0.36), (0.5,0.21,0.5,0.16)]
for x1,y1,x2,y2 in flechas:
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

ax.text(0.15, 0.63, "consulta", ha='center', fontsize=7,
        color='gray', transform=ax.transAxes)
ax.text(0.32, 0.52, "top-k docs", ha='center', fontsize=7,
        color='gray', transform=ax.transAxes)

plt.suptitle("RAG — Retrieval-Augmented Generation", fontsize=12)
plt.tight_layout()
plt.savefig("rag_analisis.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\n  Guardado: rag_analisis.png")
