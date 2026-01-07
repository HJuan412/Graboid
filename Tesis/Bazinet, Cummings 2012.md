# A comparative evaluation of sequence classification programs

Caracterización de comunidades biológicas depende de la precisión y sensibilidad de los clasificadores.

Estrategia básica de clasificación consiste en comparar contra secuencias anotadas en bases de datos de referencia. (Secuencias de consulta pueden no tener instancias similares en la base de referencia uh oh).

Este tipo de clasificación es aprendizaje *supervisado* (se utilizan instancias conocidas). Binning/clustering es aprendizaje *no supervisado*, util para caracterizar secuencias a niveles taxonómicos altos (ej filo), permite asignar clasificaciones potenciales a grupos altamente similares de secuencias.

Barcoding, uso de genes marcadores (alta variabilidad, presentes en múltiples taxones), gen 16S en bacterias, 18S y COI en animales, rbcL y matK en plantas (cloroplastos).

Tres enforques prinicpales de aprendizaje supervisado:
* Similitud de secuencia (homología o alineamiento, ej: blast)
* Composición de secuencias (modelos de Markov, conteo de kmeros)
* Métodos filogenéticos (aplican modelo evolutivo a las secuencias para ver donde encajan en el árbol filogenético)

La mayoría de los programas se basan en un método de clasificación único, pero algunos combinan métodos de dos tipos.

Clasificadores por similitud (CARMA, FACS, jMOTU, MARTA, MEGAN, MetaPhyler, MG-RAST, MTR, SOrt-ITEMS) se basan mayormente en blast. Alta precisión, pueden ser computacionalmente demandantes si las bases de referencia son grandes, y son suceptibles a taxones no representados.

Clasificadores por composición incluyen (Naive Bayes Classifier, PhiloPythia, PhymmBL, RAIphy, RDP, Scimm, SPHINX, TACOA). Tienden a emplear modelos de markoc, classificadores naive bayes, algoritmos k-means/k-nearest neighbour.

Programas basados en métodos filogenéticos (EPA, FastTreee, pplacer). Modelos evolutivos por maxima verosimilitud, métodos bayesianos o neighbour joining. Requieren una filogenia de referencia preexistente. Computacionalmente demandantes.

Métodos pueden ofrecer valores de confianza en la asignación, útiles para distinguir clasificaciones claras de ambiguas (potencialmente causadas por taxones no representados en la base de referencia)

Perfomrance de los programas se ve afectada por el dataset, sets que dan problemas tienden a ser igual de difíciles para todos los clasificadores basados en el mismo método.

Tradeoff entre sensibilidad y precisión, alta sensibilidad es indeseable si correlaciona con baja precisión.