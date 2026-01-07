# Inferring species membership using DNA sequences with back-propagation neural networks
Abordan problema de alcanzar clasificación taxonómica a partir del barcode, alternativa al método por distancia (blast, árboles filogenéticos).
	Es necesario definir un umbral de similitud para determinar pertenencia a un taxón (umbral universal puede no existir).
	Convertir las secuencias a distancias necesariamente descarta información
Modelos basados en aprendizaje Bayesiano y teoría de decisión, incluyen información de las secuencias pero en esencia siguen siendo basados en distancias (por la forma en la que usan la información de las secuencias?). Además estos métodos dependen de supuestos restrictivos (hipótesis evolutivas, postulados de genética de poblaciones, modelos evolutivos que pueden no siempre aplicar a datos reales).

Paper propone método basado en redes neuronales (enfasis en backpropagation)
Convierten bases a números 0.1, 0.2, 0.3, 0.4 (esto no asigna cardinalidad?)

Método difiere de enfoques basados en distancia dado que no requiere de un umbral definido a priori para distinguir entre grupos. Se utiliza toda la información contenida en la secuencia (en lugar de por el conjunto de diferencias entre secuencias).
Análisis de correlación de tasa de éxito en la identificación de especies diferencia el método basado en redes tiene mejor performance que los basados en distancias.
Método basado en redes no hace supuestos sobre los datos (ej: distancia corta indica cercanía entre organismos, lo cual puede no ser cierto en casos de lineage sorting incompleto). Métodos basados en distancia también son susceptibles a error cuando especies tienen pocas secuencias conocidas (o una sola, no permite evaluar tendencias dentro de la especie).

Enfoque basado en redes tiene la limitación de que las secuencias query siempre serán asignadas a un taxón conocido (solo funciona para detección de especies pre identificadas). Además, selección de parámetros para encontrar la mejor combinación puede ser complejo.

Longitud del fragmento marcador condiciona la cantidad de información, aumenta costo computacional. Es dificil definir si existe una longitud óptima universal, se recomienda trabajar con el fragmento más largo posible.

Uso de caracteres es un enfoque no lineal que puede permitir sotear alguna de las limitaciones de los métodos basados en distancias (retención de polimorfismos ancestrales, hibridación). Incrementar el número de referencias mejora la tasa de éxito en la identificación de especies. No es esperable que ningún método logre clasificar correctamente a todos los taxones. Estrategia mixta: marcadores usados para identificar grupos de taxones cercanamente emparentados, secuencias largas para diferenciar entre especies cercanas.