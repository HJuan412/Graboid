# Environmental DNA metabarcoding: Transforming how we  survey animal and plant communities

[[Introducción metabarcoding]]

eDNA es una herramienta que permite detectar multiples especies de forma rápida y eficiente a partir de muestras ambientales

Mencionan limitaciones de la identificación convencional de organismos (requieren expertos especializados, aun presentan errores, metodos invasivos pueden impactar en el ambiente, especies pequeñas o elusivas dificiles de muestrear)

Ventajas de eDNA (abarca mayor diversidad por muestra, mayor cantidad de fuentes de ADN, agua, aire, tierra, heces parásitos, permite detectar especies elusivas o raras de manera no disruptiva).
Desventajas: rango temporal y espacial de los organismos muestreados se ve distorsionado

eDNA puede ser usado para estimar abundancia relativa de especies (qPCR, inferido a partir de contenido inicial de ADN en la muestra). Pero es complicado. Correlación entre copias de adn y biomasa, no necesariamente entre copias y numero de  individuos. Factores ecológicos o ambientales pueden afectar la cantidad de eDNA presente en el entorno.

Bias de primers, errores durante la preparación de las librerías (sub muestreo puede excluir reads raros)

Uso de primers: COI es el marcador estándar en estudios animales, no obstante no todos los grupos taxonómicos contienen información suficiente para ser diferenciados a nivel de especie con este marcador. Otros marcadores deben ser empleados, es necesario conocer a priori la capacidad del marcador utilizado.

Etapas de procesamiento bioinformatico de reads
* Clustering de secuencias en MOTU, en base a un umbral de similitud
* Asignación taxonómica, comparación de entidades de consulta MOTU/secuencias contra una base de datos de referencia
	* [[Bazinet, Cummings 2012]] comparan algoritmos de secuenciación
	* Similitud por alineamiento
	* Modelos ocultos de markov
	* Composición de secuencia
	* ML (Wang et al 2007, Diaz et al 2009, eren et al 2015 )
	Utilidad de los programas está determinada por los marcadores y la resolución de las bases de referencia.
* Análisis de diversidad: calculo de índices de diversidad (alfa y beta) testeo de hipótesis.

Asignación taxonómica requiere bases de datos de referencia especializadas y curadas (actualmente, NCBI, BOLD son las más completas)