# Conclusión — TP Bloque 12: Aprendizaje por Refuerzo

La implementación de Q-Learning en GridWorld permitió verificar empíricamente los conceptos teóricos del bloque sobre un entorno controlado y observable.

## Lo que aprendió el agente

El agente partió sin ningún conocimiento del entorno y construyó su política únicamente a partir de la interacción. Después de 500 episodios logró encontrar un camino consistente hacia la meta en aproximadamente 10 pasos, esquivando muros y trampas sin que nadie le indicara dónde estaban. Esto ilustra la idea central del aprendizaje por refuerzo: el conocimiento emerge de la experiencia acumulada, no de supervisión externa.

## Exploración vs explotación

El experimento con distintos valores de ε mostró la tensión del dilema con claridad. Un agente con ε=1.0 que solo explora obtiene retorno promedio de -5.55 porque nunca aprovecha lo aprendido. Un agente con ε=0.01 que casi no explora obtiene 9.09 porque ejecuta consistentemente la mejor política conocida. El valor intermedio ε=0.1 es el que mejor generaliza durante el entrenamiento porque mantiene la capacidad de corregir errores mientras explota lo ya aprendido.

## Efecto de los parámetros

- **α (tasa de aprendizaje)**: valores muy bajos como 0.01 aprenden demasiado lento y no convergen bien en 500 episodios. Valores altos convergen más rápido pero pueden ser inestables si el entorno tiene ruido.
- **γ (factor de descuento)**: controla el horizonte temporal del agente. Un γ bajo hace que el agente sea miope y solo optimice la recompensa inmediata. En GridWorld no cambia mucho el resultado final porque el entorno es pequeño, pero en entornos con caminos largos un γ bajo llevaría a políticas subóptimas.

## Cambios en el entorno

El experimento más revelador fue aumentar la penalización por paso de -0.1 a -1.0. El retorno final cae a -0.28 porque el agente paga un costo alto por cada movimiento, pero la política resultante es cualitativamente distinta: el agente aprende a ser más directo y evitar rodeos. Esto muestra que la función de recompensa no solo determina cuánto aprende el agente sino qué tipo de comportamiento emerge.

## La tabla Q como representación del conocimiento

La tabla Q aprendida es una codificación explícita de todo lo que el agente sabe sobre el entorno. Cada entrada Q(s,a) representa el retorno esperado de ejecutar la acción a desde el estado s y luego seguir la política óptima. Lo interesante es que esta estructura es completamente interpretable: se puede leer directamente qué valores asignó el agente a cada celda y verificar si tiene sentido geométrico con la disposición de la grilla.

## Límites del enfoque tabular

El enfoque funciona bien en una grilla 6x6 con 30 estados visitados. Pero si la grilla fuera de 100x100, o si el estado incluyera variables continuas como velocidad o ángulo, la tabla crecería hasta ser imposible de mantener en memoria. Ese es el límite que motiva Deep Q-Networks, donde la tabla Q se reemplaza por una red neuronal que generaliza entre estados similares sin necesitar verlos todos explícitamente.
