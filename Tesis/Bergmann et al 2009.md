# Character based DNA barcoding: a superior tool for species classification
Criticas al barcoding usando el marcador universal COI:
* Desempeño no es igual de bueno para todos los taxones
* Método a partir de distancia genética no es uniforme, particularmente a nivel de especies, no existe un umbral de distancia único aplicable a todos los taxones.
Distancia intra/inter específica no es consistente y varía de acuerdo al organismo. Taxones con muestreo insuficiente y alta variabilidad pueden resultar difíciles de identificar por distancia genética (variaciones de la misma especie no representadas en la base de datos y genéticamente distintas a las que si lo están pueden ser erróneamente rechazadas)

**Barcoding basado en caracteres**
En lugar de distancias, se identifican caracteres (individuales o en vombinaciones) diagnósticos de un taxón. En el contexto de ADN, estos caracteres son SNP.
Barcoding basado en caracteres es más que un conjunto de SNPs, pueden ser combinaciones complejas.

Algoritmo CAOS (*characteristic attribute organisation system*): basado en el concepto de que miembros de un mismo grupo taxonómico comparten atributos que no estan presentes en toros grupos. Identifica atributos característicos (CA) para cada clado en cada nodo de un arbol guía.
CA son caracteres presentes en un único clado pero en ningún grupo alternativo que descienda del mismo nodo (como un sitio tipo 1 dentro de los grupos hermanos dentro de un mismo rango taxonómico).
CA se dividen en:
1) Simples puros (sPu): presentes en todos los miembros de un clado y nunca en otros clados (tipo 1)
2) Simples privados (sPr): presentes en algunos miembros de un clado y nunca en otros claods (tipo 2)
3) Compuestos puros (cPu): combinaciones de CA que se observan en todos los miembros de un clado y nunca en otros clados (tipo 3, resuelto)
4) Compuestos privados (cPr): combinaciones de CA que se observan en algunos miembros de un clado y nunca en miembros de otros clados.

Usan gen ND1 para complementar COI en pruebas del algoritmo CAOS con odonatos