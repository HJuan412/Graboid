# Assessment of Common and Emerging  Bioinformatics Pipelines for Targeted  Metagenomics

*Targeted metagenomics* ~= metabarcoding, metagenómica basada en amplicones de secuencias marcador.

Aumento de disponibilidad de equipos de secuenciación posibilitan que estudios de este tipo sean más accesibles para laboratorios de menor escala.

Tasa de errores de secuenciación es un factor a tener en cuenta, dado el impacto que puede tener en el resultado de clasificación, y por lo tanto en la descripción final de la comunidad biológcia.

Múltiples herramientas de análisis bioinformático disponibles para procesar datos de secuenciación. Incluyen pasos de:
* Filtrado por calidad
* Detección de quimeras
* Clustering de reads en OTUs
* Asignación taxonómica
Pueden requerir habilidades bioinformáticas avanzadas y disponibilidad de recursos de computación.

Este paper evalúa múltiples métodos de análisis de metagenómica dirigida en su totalidad. Se dividen en 2 tipos:
1) Clustering primero: Los reads se agrupan en OTUs, luego las OTU se clasifican (permiten clasificar reads individualmente desconocidos al incorporarlos a un grupo). Mothur, QIIME, BMP.
2) Asignación primero: A cada read se le asigna una clasificación, luego se agregan en base a la anotación generada. (Detectan variedades en grupos taxonómicos). One Codex, Kraken, CLARK

El protocolo de evaluación utilizado emplea datos simulados para evaluar el impacto de las diferentes variables en el desempeño de los clasificadores. Validación cruzada con datos reales.

Parámetros:
* Selección de primers
* Cobertura de secuenciacón
* Tasa de error
Programas fueron evaluados con sus bases de datos recomendadas, pero también se utilizaron bases alternativas para evaluar el impacto en la clasificación.

Métricas:
* F score
* Riqueza
* Diversidad
* Indices de clustering
## Resultados
El desempeño general de clasificación no varía mucho (F cerca de la diagonal en todos los clasificadores, Fig3).
Performance de clasificadores más afectada por el rango taxonómico (clustering primero se comportan peor a nivel de género sin importar el marcador)
Errores de secuenciación afectan el recall (más reads sin clasificar, mothur y kraken se ven más afectado, qiime es el menos sensible).
Errores causan sobreestimación de riqueza (introducen más variantes, más OTUs con pocos representantes)
Metodos assignment-first tienen sobreestiman la riqueza en datasets con errores al aumentar volumen de reads (tiene sentido, mayor cantidad de variantes raras). Metodos clustering-first sobreestiman riqueza en datasets con mayor cantidad de reads.
Base de referencia: Diferentes herramientas recomiendan diferentes bases de datos. En este estudio, variar la base de datos afecta la estimación de riqueza más que la métrica F.
Datasets reales tienen una mayor cantidad de reads sin clasificar independientemente de la herramienta (ruido, microbioma rico que incluye taxones no presentes en la base de datos). Al agrupar las herramientas en función de la diversidad reportada.
Metodos clustering-first reportan menor riqueza al usar la base de datos por defecto. El resultado de clasificación está condicionado por la precisión de los datos de referencia.
## Discusión
Enfoque inicial se basa en paso previo de clusterización de reads en OTUs, la mayoría de los enfoques actuales se basan en este enfoque (más antiguo, más cantidad de publicaciones, ás accesible).
Adopción de nuevas estrategias requiere evaluación haciendo uso de equipos de computo significativos (no accesibles a todos los equipos).
Algoritmos basados en asignación antes de agrupación son menos demandantes en terminos de capacidad de computo. No fueron desarrollados con metagenómica dirigida en mente.
Herramientas assignment-first alcanzan resoluciones taxonómicas más detalladas, métodos clustering-first tienen evaluaciones de riqueza más robustas frente a variaciones en la cantidad de reads y cuando hay errores de secuenciación presentes.