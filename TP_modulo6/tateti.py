"""
Ta-Te-Ti con Minimax y Poda Alfa-Beta
TP Bloque 6: Búsqueda Adversaria

Representación del tablero:
  0 | 1 | 2
  ---------
  3 | 4 | 5
  ---------
  6 | 7 | 8

Valores: ' ' = vacío, 'X' = MAX, 'O' = MIN
"""

TRAZAR = True          # muestra el árbol de decisiones por default
PROF_TRAZA = 2         # hasta qué profundidad se imprime (evita flood)

# ─────────────────────────────────────────────
# PASO 1: Representación del estado del juego
# ─────────────────────────────────────────────

def tablero_inicial():
    return [' '] * 9

def imprimir_tablero(tablero):
    def celda(i):
        return tablero[i] if tablero[i] != ' ' else str(i)
    print()
    print(f" {celda(0)} | {celda(1)} | {celda(2)} ")
    print("---+---+---")
    print(f" {celda(3)} | {celda(4)} | {celda(5)} ")
    print("---+---+---")
    print(f" {celda(6)} | {celda(7)} | {celda(8)} ")
    print()

# ─────────────────────────────────────────────
# PASO 2: Funciones de reglas del juego
# ─────────────────────────────────────────────

LINEAS_GANADORAS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # filas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columnas
    (0, 4, 8), (2, 4, 6)              # diagonales
]

def ganador(tablero):
    """Retorna 'X', 'O', o None."""
    for a, b, c in LINEAS_GANADORAS:
        if tablero[a] == tablero[b] == tablero[c] != ' ':
            return tablero[a]
    return None

def es_terminal(tablero):
    """True si el juego terminó (hay ganador o empate)."""
    return ganador(tablero) is not None or ' ' not in tablero

def utilidad(tablero):
    """
    Valor del estado terminal:
      +1 si gana X (MAX)
      -1 si gana O (MIN)
       0 empate
    """
    g = ganador(tablero)
    if g == 'X':
        return 1
    if g == 'O':
        return -1
    return 0

def acciones_disponibles(tablero):
    """Retorna lista de índices vacíos."""
    return [i for i, c in enumerate(tablero) if c == ' ']

def aplicar_accion(tablero, accion, jugador):
    """Retorna nuevo tablero con la jugada aplicada (sin mutar el original)."""
    nuevo = tablero[:]
    nuevo[accion] = jugador
    return nuevo

# ─────────────────────────────────────────────
# PASO 3: Minimax puro
# ─────────────────────────────────────────────
#
# Recorre el árbol completo en profundidad.
# MAX (X) quiere maximizar, MIN (O) quiere minimizar.
# contador_minimax cuenta nodos explorados para comparar con alfa-beta.

contador_minimax = 0

def minimax(tablero, es_max, prof=0):
    global contador_minimax
    contador_minimax += 1

    if es_terminal(tablero):
        u = utilidad(tablero)
        if TRAZAR and prof <= PROF_TRAZA:
            etiqueta = "GANA X" if u == 1 else ("GANA O" if u == -1 else "EMPATE")
            print(f"{'  ' * prof}terminal → {etiqueta} ({u:+d})")
        return u

    jugador = 'X' if es_max else 'O'
    rol     = "MAX" if es_max else "MIN"
    acciones = acciones_disponibles(tablero)

    if TRAZAR and prof <= PROF_TRAZA:
        print(f"{'  ' * prof}[{rol}] posiciones disponibles: {acciones}")

    if es_max:
        mejor = -float('inf')
        for a in acciones:
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  MAX prueba posición {a}:")
            valor = minimax(aplicar_accion(tablero, a, jugador), False, prof + 1)
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  posición {a} → valor {valor:+d}")
            mejor = max(mejor, valor)
        return mejor
    else:
        mejor = float('inf')
        for a in acciones:
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  MIN prueba posición {a}:")
            valor = minimax(aplicar_accion(tablero, a, jugador), True, prof + 1)
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  posición {a} → valor {valor:+d}")
            mejor = min(mejor, valor)
        return mejor

def elegir_accion_minimax(tablero):
    global contador_minimax
    contador_minimax = 0

    if TRAZAR:
        print("\n--- TRAZA MINIMAX (primeros 2 niveles) ---")

    mejor_valor = -float('inf')
    mejor_accion = None
    for a in acciones_disponibles(tablero):
        if TRAZAR:
            print(f"\n[raíz] MAX evalúa posición {a}:")
        valor = minimax(aplicar_accion(tablero, a, 'X'), False, 1)
        if TRAZAR:
            print(f"[raíz] posición {a} → valor final: {valor:+d}")
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_accion = a

    if TRAZAR:
        print(f"\n  Mejor acción: posición {mejor_accion} (valor {mejor_valor:+d})")
        print("--- FIN TRAZA ---\n")

    print(f"  [Minimax] Nodos explorados: {contador_minimax} | Valor: {mejor_valor}")
    return mejor_accion

# ─────────────────────────────────────────────
# PASO 4: Alfa-Beta
# ─────────────────────────────────────────────
#
# Igual que minimax pero poda ramas que no cambian el resultado.
# α = mejor garantía para MAX hasta ahora
# β = mejor garantía para MIN hasta ahora
# Si α ≥ β → podamos (el oponente nunca elegiría este camino)

contador_alfabeta = 0

def alfabeta(tablero, es_max, alfa, beta, prof=0):
    global contador_alfabeta
    contador_alfabeta += 1

    if es_terminal(tablero):
        u = utilidad(tablero)
        if TRAZAR and prof <= PROF_TRAZA:
            etiqueta = "GANA X" if u == 1 else ("GANA O" if u == -1 else "EMPATE")
            print(f"{'  ' * prof}terminal → {etiqueta} ({u:+d})")
        return u

    jugador  = 'X' if es_max else 'O'
    rol      = "MAX" if es_max else "MIN"
    acciones = acciones_disponibles(tablero)

    if TRAZAR and prof <= PROF_TRAZA:
        a_fmt = f"{alfa:+.0f}" if alfa != -float('inf') else "-inf"
        b_fmt = f"{beta:+.0f}" if beta != float('inf') else "+inf"
        print(f"{'  ' * prof}[{rol}] posiciones: {acciones}  (α={a_fmt}, β={b_fmt})")

    if es_max:
        valor = -float('inf')
        for a in acciones:
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  MAX prueba posición {a}:")
            valor = max(valor, alfabeta(aplicar_accion(tablero, a, jugador), False, alfa, beta, prof + 1))
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  posición {a} → valor {valor:+d}  (α={alfa:+.0f})")
            alfa = max(alfa, valor)
            if alfa >= beta:
                if TRAZAR and prof <= PROF_TRAZA:
                    print(f"{'  ' * prof}  *** PODA: α={alfa:+.0f} >= β={beta:+.0f} → descartamos el resto ***")
                break
        return valor
    else:
        valor = float('inf')
        for a in acciones:
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  MIN prueba posición {a}:")
            valor = min(valor, alfabeta(aplicar_accion(tablero, a, jugador), True, alfa, beta, prof + 1))
            if TRAZAR and prof <= PROF_TRAZA:
                print(f"{'  ' * prof}  posición {a} → valor {valor:+d}  (β={beta:+.0f})")
            beta = min(beta, valor)
            if alfa >= beta:
                if TRAZAR and prof <= PROF_TRAZA:
                    print(f"{'  ' * prof}  *** PODA: α={alfa:+.0f} >= β={beta:+.0f} → descartamos el resto ***")
                break
        return valor

def elegir_accion_alfabeta(tablero):
    global contador_alfabeta
    contador_alfabeta = 0

    if TRAZAR:
        print("\n--- TRAZA ALFA-BETA (primeros 2 niveles) ---")

    mejor_valor = -float('inf')
    mejor_accion = None
    alfa = -float('inf')
    beta = float('inf')

    for a in acciones_disponibles(tablero):
        if TRAZAR:
            print(f"\n[raíz] MAX evalúa posición {a}:")
        valor = alfabeta(aplicar_accion(tablero, a, 'X'), False, alfa, beta, 1)
        if TRAZAR:
            print(f"[raíz] posición {a} → valor final: {valor:+d}")
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_accion = a
        alfa = max(alfa, mejor_valor)

    if TRAZAR:
        print(f"\n  Mejor acción: posición {mejor_accion} (valor {mejor_valor:+d})")
        print("--- FIN TRAZA ---\n")

    print(f"  [Alfa-Beta] Nodos explorados: {contador_alfabeta} | Valor: {mejor_valor}")
    return mejor_accion

# ─────────────────────────────────────────────
# PASO 5: Comparación de rendimiento
# ─────────────────────────────────────────────

def comparar_rendimiento():
    """Compara minimax vs alfa-beta en el tablero inicial (peor caso)."""
    tablero = tablero_inicial()
    print("=" * 50)
    print("COMPARACIÓN DE RENDIMIENTO (tablero vacío)")
    print("=" * 50)

    global contador_minimax, contador_alfabeta
    contador_minimax = 0
    elegir_accion_minimax(tablero)
    m = contador_minimax

    contador_alfabeta = 0
    elegir_accion_alfabeta(tablero)
    ab = contador_alfabeta

    reduccion = (1 - ab / m) * 100
    print(f"\n  Minimax: {m} nodos")
    print(f"  Alfa-Beta: {ab} nodos")
    print(f"  Reducción: {reduccion:.1f}%")
    print()

# ─────────────────────────────────────────────
# PASO 6: Juego interactivo humano vs IA
# ─────────────────────────────────────────────

def jugar(modo='alfabeta'):
    """
    Partida interactiva: humano (O) vs IA (X).
    modo: 'minimax' o 'alfabeta'
    """
    elegir_fn = elegir_accion_minimax if modo == 'minimax' else elegir_accion_alfabeta
    tablero = tablero_inicial()

    print(f"\n{'='*50}")
    print(f"TA-TE-TI — IA usa: {modo.upper()}")
    print(f"Tú eres O, la IA es X")
    print(f"{'='*50}")

    # Humano elige si quiere ir primero
    turno = input("¿Querés ir primero? (s/n): ").strip().lower()
    turno_humano = turno == 's'

    while not es_terminal(tablero):
        imprimir_tablero(tablero)

        if turno_humano:
            opciones = acciones_disponibles(tablero)
            print(f"Tu turno (O). Posiciones disponibles: {opciones}")
            while True:
                try:
                    mov = int(input("Ingresá posición: "))
                    if mov in opciones:
                        break
                    print("Posición inválida, intentá de nuevo.")
                except ValueError:
                    print("Ingresá un número.")
            tablero = aplicar_accion(tablero, mov, 'O')
        else:
            print("Turno de la IA (X)...")
            mov = elegir_fn(tablero)
            tablero = aplicar_accion(tablero, mov, 'X')
            print(f"  IA jugó en posición {mov}")

        turno_humano = not turno_humano

    imprimir_tablero(tablero)
    g = ganador(tablero)
    if g == 'X':
        print("¡Ganó la IA (X)!")
    elif g == 'O':
        print("¡Ganaste (O)!")
    else:
        print("¡Empate!")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\nBLOQUE 6 — Búsqueda Adversaria: Ta-Te-Ti")
    print("==========================================")
    print("1. Comparar rendimiento Minimax vs Alfa-Beta")
    print("2. Jugar contra Minimax")
    print("3. Jugar contra Alfa-Beta")

    opcion = input("\nElegí opción (1/2/3): ").strip()

    if opcion == '1':
        comparar_rendimiento()
    elif opcion == '2':
        jugar(modo='minimax')
    elif opcion == '3':
        jugar(modo='alfabeta')
    else:
        print("Opción inválida.")
