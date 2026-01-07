# DNA barcode analysis: a comparison of phylogenetic and statistical classification methods
Problemas de clasificación de especies cercanas usando marcadores convencionales como COI.

Mutaciones pueden caracterizar especies incluso sin monofilia recíproca, cuando se consideran múltiples mutaciones que permiten, en conjunto, diferenciar a todos los miembros de un taxón del resto.

Mutación epsilon (fig 1d) representa un sitio de tipo 4 en el clasificador CL de graboid.

Métodos de barcoding se dividen en 4 categorías
1) Por similitud
2) Filogenéticos
3) KNN y enfoques estadísticos basados en algoritmos de clasificación sin modelos biológicos subyacentes
4) genealógicos

Ningún método es intrínsecamente superior

KNN basado en distancia k2p (kimura) con multiples valores de K y analogo del sistema de orbitales (mejor resultado siempre con K=1)

**Resultados**
1-NN y random forest se desempeñan mejor que arboles de decisión (métodos basados en algoritmos de clasificación), y de manera similar a los métodos filogenéticos (1-NN parece el más confiable).
Desempeño de 1-NN mejora con el tamaño muestral.
Multiples especies dificultan el proceso (más probabilidad de confusión, multiples candidatos), tasas de especiación elevadas facilitan discriminación.
En datasets reales, todos los métodos se comportan de forma adecuada, con 1-NN y NJ ofreciendo los mejores resultados.

**Discusión**
Ningún método sobresale de forma consistente sobre los otros, el mejor método depende de las características de los datos (número de taxonesy distancia entre los mismos). 1-NN parece ser el método más fiable (nunca se aparta mucho del mejor desempeño, cuando no lo presenta él mismo).
1-NN establece que el query pertenece al taxón de la secuencia más similar en la base de datos. Especies jóvenes pueden presentar mutaciones "específicas" pero no diagnósticas (<u>sitios tipo 2</u>)
Métodos como CART, RF (y CL) pueden ser engañados por mutaciones compartidas entre taxones, al enfocarse en el vecino más próximo, 1-NN tiene menor probabilidad de confundirse con organismos de otra especie con algunas mutaciones compartidas.
Métodos de clasificación supervisados dependen de qué tan completa sea la representación de la realidad en el conjunto de aprendizaje (se ven muy afectados por conjuntos de muestreo pequeños).
Sugieren estrategia *leave-one-out* para evaluar previamente el desempeño de los clasificadores con el dataset de entrenamiento.
Cantidad de información mejora el desempeño: tamaño muestral, cantidad de información al usar amplicones más largos, número de loci. Aumentar la cantidad de instancias de referencia mejora el desempeño de todos los métodos (lógico, aumenta la variabilidad registrada). Aumentar el número de marcadores puede complementar el resultado para taxones en los que el marcador convencional no tiene suficiente información.

**Conclusión**
Diferentes métodos generan el mejor resultado dependiendo de las características de los datos, selección de métodos debería ser en dos pasos: evaluación de métodos para la estructura de los datos (evaluar aptitud del set de referencia), selección de método para aplicar al conjunto de consulta.