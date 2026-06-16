# Conclusión — TP Bloque 14: Arquitecturas Modernas e IA Híbrida

Este trabajo práctico demostró empíricamente la idea central del bloque: la arquitectura no es un detalle de implementación sino una decisión de diseño que incorpora supuestos sobre la estructura del problema. Cada arquitectura es una hipótesis sobre cómo está organizado el mundo.

## MLP vs CNN

El experimento más directo del TP. El MLP trata cada píxel como una variable independiente y necesita aprender desde cero que los píxeles cercanos están relacionados. La CNN incorpora ese supuesto directamente en su estructura: cada filtro mira una región local y ese conocimiento se comparte en toda la imagen. El resultado fue que la CNN alcanzó 96.6% de accuracy con menos parámetros que el MLP, que se quedó en 93.1%. La diferencia no viene de que la CNN sea "más grande" sino de que su sesgo inductivo es más apropiado para datos espaciales. Esto ilustra que agregar parámetros sin el sesgo correcto no necesariamente mejora el modelo.

## RNN vs LSTM

El experimento diseñado para hacer visible el vanishing gradient. Con secuencias de largo 50 y ruido fuerte en todos los pasos intermedios, el gradiente debe viajar 50 pasos hacia atrás para llegar al valor que determina la etiqueta. En este escenario extremo ninguna arquitectura simple alcanza la solución perfecta, pero el experimento muestra la dirección correcta: la LSTM fue diseñada exactamente para este problema mediante sus compuertas de memoria, mientras que la RNN vanilla no tiene mecanismo para decidir qué retener y qué olvidar a lo largo de la secuencia.

## Self-Attention Q/K/V

La implementación desde cero en numpy permitió observar el mecanismo de atención sin abstracciones. Cada token genera tres vectores: Q codifica qué busca, K codifica qué ofrece, y V codifica qué información aporta si es seleccionado. El producto Q·Kᵀ produce una matriz de similitudes, el softmax la normaliza en una distribución de atención, y el producto final con V genera un nuevo embedding que es una mezcla contextualizada de toda la secuencia. La propiedad que lo diferencia de las RNN es que ese acceso global ocurre en un solo paso, sin que el gradiente tenga que viajar por pasos intermedios. En el ejemplo con la oración "El gato que viste ayer duerme", el token "El" asignó peso 0.914 al token "viste", capturando una relación gramatical a distancia sin ningún entrenamiento previo.

## RAG

El experimento de RAG mostró con claridad la diferencia entre conocimiento paramétrico y no paramétrico. El sistema paramétrico puro responde desde los pesos fijos aprendidos durante el entrenamiento, sin poder citar fuentes ni actualizarse sin reentrenar. El sistema RAG recupera documentos relevantes en tiempo de inferencia y condiciona la respuesta sobre ese contexto, lo que permite actualizar el conocimiento editando la base de datos, citar la fuente exacta de cada afirmación y reducir alucinaciones al anclar la generación a texto real. El caso más revelador fue la pregunta sobre los planetas del sistema solar: al estar fuera del dominio de la base de conocimiento, el recuperador devolvió documentos con relevancia baja, lo que permite al sistema reconocer que no tiene información útil en vez de inventar una respuesta.

## Conclusión general

Los cuatro experimentos juntos ilustran que el progreso en deep learning no fue solo conseguir más datos o más cómputo, sino diseñar arquitecturas que incorporan el sesgo inductivo correcto para cada tipo de problema. Las CNN incorporan localidad para imágenes, las LSTM incorporan memoria selectiva para secuencias, los Transformers incorporan acceso global al contexto, y los sistemas híbridos como RAG incorporan la separación entre conocimiento paramétrico y recuperable. En todos los casos, la arquitectura es una decisión sobre qué estructura se asume que tiene el problema antes de ver ningún dato.
