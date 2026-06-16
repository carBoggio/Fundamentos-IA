# Conclusión — Bloque 6: Búsqueda Adversaria

La implementación de Minimax y Poda Alfa-Beta sobre Ta-Te-Ti permite verificar empíricamente las propiedades teóricas de ambos algoritmos. Minimax garantiza optimalidad bajo el supuesto de juego perfecto del oponente, recorriendo el árbol completo con complejidad O(b^d): en el tablero vacío esto implica explorar 549.945 nodos para un espacio de estados de apenas 9! = 362.880 partidas posibles. La Poda Alfa-Beta reduce ese recorrido a 18.296 nodos mediante el mantenimiento de las cotas α y β, logrando una reducción del 96.7% sin alterar la decisión óptima, lo que verifica la complejidad teórica O(b^(d/2)) en el caso promedio.

La traza del algoritmo evidencia el mecanismo de poda: cuando α ≥ β, el nodo actual no puede influir en la decisión del ancestro correspondiente, independientemente de los valores de sus descendientes. Esto equivale a demostrar que el oponente racional nunca elegiría ese camino dado lo que ya se conoce del árbol.

El límite de escala queda implícito en el ejercicio: Ta-Te-Ti con b≈9 y d≈9 es tratable en milisegundos, pero un juego como ajedrez con b≈35 y d≈100 hace que incluso Alfa-Beta sea insuficiente sin estrategias adicionales como profundidad limitada, funciones heurísticas o MCTS, lo que motiva la evolución hacia enfoques híbridos como AlphaGo.
