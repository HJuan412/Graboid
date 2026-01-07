# Statistical approaches for DNA barcoding
Similitud *misleading*
* Error humano en el etiquetado de la secuencia de referencia
* Especie verdadera no presente en la base de datos
* Lineage sorting: query puede estar más próxima a otra especie en la secuencia del marcador
* Mutaciones aleatorias pueden generar que miembros de otra especie sean más similares al query

Se plantea que barcoding requiere hacer supuestos de genética de poblaciones: especies presentan un grado de variación interna. Medidas de incertidumbre estadística en barcoding depende de supuestos fuertes sobre la genética de poblaciones de la especie objetivo.
Procedimientos de inferencia deberían ser robustos a violaciones en los supuestos.

Analizar el numero de variaciones entre una secuencia query y la secuencia más cercana de la base de datos no es óptimo. Es más adecuado analizar el grado de variación entre la query y un número de especies divergentes en la base de datos.