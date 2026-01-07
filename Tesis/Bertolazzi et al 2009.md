# Learning to classify species with barcodes
Metodos de clasificación por barcoding se basan en distancias entre mOTUs o por caracteres diagnósticos.
Método propuesto se basa en caracteres y tiene dos etapas:
1) Selección de atributos
2) Identificación de formulas lógicas que separan las clases taxonómicas
Genera reglas de clasificación compactas que contienen información acerca de sitios informativos.

Método de clasificación es un separador en dos clases (*i*-ésimo individuo pertenece al taxón *t* o no), dependiendo de una función lógica.

**Feature selection**
Componente del análisis de datos en el que se extrae un subconjunto de atributos relevantes de un conjunto grande.

Individuos pertenecen a una clase A o B
Dado un atributo $f_j$, $P_A(j,k)$ y $P_B(j,k)$ son las proporciones de individuos para los que $j=k, k \in (A, C, G, T)$ 
Si $P_A(j,k)$ > $P_B(j,k)$, entonces $f_j=k$ es más probablemente una característica diagnóstico de A (y de B si ocurre lo contrario)
Sea $d_{ij}$ para un individuo i perteneciente a la clase A:
* 1 si $f_{ij}=k$ & $P_A(j,k) \ge \lambda P_B(j,k)$
* 0 si $f_{ij}=k$ & $\lambda P_A(j,k) \le P_B(j,k)$
* 1 si $f_{ij}\ne k$ & $\lambda P_A(j,k) \le P_B(j,k)$
* 0 si $f_{ij} \ne k$ & $P_A(j,k) \le \lambda P_B(j,k)$
Con $\lambda>1$ siendo un valor arbitrario que determina el grado de diferencia entre $P_A(j,k)$ y $P_B(j,k)$ para considerar un carácter como diagnóstico (ej, 1.2, 20%)
El vector $d_j$ colecta los valores para todas las instancias en el carácter j, un alto número de unos indica que j es un buen separador de las clases A y B.
Se selecciona un número $\beta$ de atributos que maximicen la capacidad de discriminar entre las dos clases.

Muy similar al método a base de entropía.

**Extracción de fórmulas lógicas**
Método *Lsquare*, define reglas de lógica proposicional para clasificar instancias en una de dos clases.
Las reglas son fórmulas lógicas en Forma Normal Disjuntiva (DNF), permite evaluar resultados de clasificación desde el punto de vista semántico. Reglas se determinan mediante formulación del *problema de minimo costo de satificibilidad* (MINSAT).
Reglas están conformadas por clausulas conjuntivas identificadas  desde las que abarcan la mayor cantidad del set de entrenamiento. (pocas clausulas con alta cobertura (tendencias observables en los datos) y multiples clausulas con alta cobertura (outliers)).
Para identificar un separador de A y B, se definen clausulas de forma iterativa, en cada etapa se identifica una clausula que se cumple para la mayor parte de A y rechaza todos los elementos de B.

Primero, variables seleccionadas son one-hot-encoded.
Segundo, se formula problema MINSAT:
	$p_j$ y $q_j$ variables binarias enlazadas con $v_j$ (esta última indica el valor del atributo v)
	$v_j$ se elige si $p_j \ne q_j$