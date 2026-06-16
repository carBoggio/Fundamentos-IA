# Conclusión — TP Bloque 13: Fundamentos de Redes Neuronales

## El problema XOR como punto de partida

El experimento con XOR es el más ilustrativo del bloque porque demuestra empíricamente el límite fundamental del perceptrón. Un modelo lineal no puede separar los puntos de XOR porque no existe ninguna recta que lo haga: el 25% de accuracy que obtuvo es peor que tirar una moneda, lo que confirma que no encontró ningún patrón útil. El MLP con una sola capa oculta de 4 neuronas y activación tanh lo resuelve al 100%. La diferencia no es la cantidad de parámetros sino la no-linealidad: la capa oculta aprende una representación intermedia del espacio que transforma el problema hasta hacerlo linealmente separable en la capa de salida.

## Capacidad del modelo y fronteras de decisión

Los experimentos sobre Moons y Circles mostraron cómo la capacidad del modelo determina qué tipo de frontera puede aprender. En Circles, un modelo sin capa oculta alcanzó 43% de accuracy en validación, peor que el azar, porque ninguna línea recta puede separar dos círculos concéntricos. Con 4 neuronas ya llegó al 100%, porque la red aprendió una frontera circular. En Moons el patrón es similar: la frontera recta captura algo pero no la forma curva real de los datos, y agregar capas mejora progresivamente la precisión. Esto ilustra el concepto central de capacidad: un modelo demasiado simple no puede representar la función objetivo, independientemente de cuántos datos tenga.

## Learning rate y convergencia

El efecto del learning rate fue uno de los resultados más claros. Con LR=0.1 el modelo convergió en 92 iteraciones y alcanzó 100% de accuracy en validación. Con LR=0.001 necesitó 481 iteraciones y solo llegó al 86.7%, no porque el modelo sea incapaz sino porque en el tiempo disponible no alcanzó a converger. Con LR=0.5 convergió en 32 iteraciones pero quedó en 90%, lo que sugiere que dio pasos tan grandes que se saltó el mínimo. El learning rate no solo afecta la velocidad de entrenamiento sino la calidad de la solución final.

## Funciones de activación

ReLU superó a sigmoid y tanh en el mismo entorno y con la misma arquitectura: 98.3% vs 87-88% en validación. La razón es que sigmoid y tanh sufren del problema del gradiente que se desvanece: en sus regiones de saturación el gradiente es casi cero y los pesos de las capas iniciales apenas se actualizan. ReLU no satura en los valores positivos, lo que mantiene gradientes más grandes y permite que el entrenamiento avance de forma más efectiva.

## Overfitting y regularización

El experimento de regularización mostró algo contraintuitivo: en Moons, incluso la red grande sin regularización no sobreajustó visiblemente porque el dataset tiene suficiente estructura y el problema no es muy complejo. En datasets más ruidosos o con menos datos el efecto de L2 sería más pronunciado. Lo que sí quedó claro es el principio: una red más grande tiene más capacidad de memorizar patrones espurios del entrenamiento, y L2 penaliza pesos grandes forzando al modelo a encontrar soluciones más simples que generalizan mejor.

## Síntesis

Los experimentos confirman que la potencia de las redes neuronales proviene de tres elementos combinados: la composición de capas que permite aprender representaciones jerárquicas, la no-linealidad que hace posible separar clases no linealmente separables, y el entrenamiento por gradiente descendente que ajusta automáticamente todos los parámetros a partir del error. Ninguno de los tres funciona sin los otros dos. El perceptrón tiene gradiente y tiene aprendizaje, pero le falta no-linealidad. Una red profunda sin activaciones no lineales es equivalente a una regresión lineal. Y una arquitectura correcta sin optimización no converge a nada útil.
