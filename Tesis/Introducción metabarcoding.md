## Caracterización de comunidades biológicas
La identificación de los conjuntos de organismos presentes en comunidades biológicas es un paso fundamental en los estudios ecológicos y el monitoreo ambiental. Una descripción precisa de la comunidad brinda información acerca de la diversidad taxonómica y funcional en el ecosistema; permite identificar la presencia de especies relevantes para la salud humana o animal, la producción agrícola.

## Métodos de clasificación
### Clasificación por morfología
La modalidad clásica para la descripción de comunidades biológicas se basa en la asignación taxonómica de organismos individuales, generalmente a partir de la identificación de caracteres morfológicos distintivos. Este tipo de estudios implica dos grandes inversiones de tiempo y esfuerzo. La primera es la colecta de organismos, se debe emplear una estrategia de muestreo que permita obtener una muestra representativa de la diversidad en el ambiente. Asimismo se debe tener en cuenta el procesamiento, almacenamiento y transporte de las muestras. La presencia de organismos raros, elusivos o difíciles de manipular añade otro nivel de complejidad. Por otra parte, el muestreo puede involucrar procedimientos invasivos o el sacrificio de organismos, lo cual puede generar perjuicios no deseados en la comunidad en estudio. [[Ficetola, Taberlet 2023]], [[Thomsen, Willerslev 2014]]
La segunda inversión de tiempo consiste en la identificación en sí. Para algunos grupos de organismos como los nemátodos, las características morfológicas utilizadas como diagnóstico pueden resultar difíciles de identificar debido al tamaño del organismo, variaciones derivadas del ciclo de vida, o similitud con taxones cercanamente emparentados. La identificación morfológica puede requerir una alta cantidad de horas de trabajo de personal capacitado, incluso para un número reducido de muestras [[Taberlet 2012]].

El monitoreo de comunidades biológicas implica la repetición de las tareas de identificación taxonómica para múltiples ambientes, así como la repetición periódica de éstas. La escalabilidad del método empleado cobra relevancia.
### Métodos moleculares
El uso de eDNA ofrece un número de ventajas para el monitoreo regular y a gran escala de comunidades biológicas [[Deiner 2017]]

Herramienta útil para describir la composición de especies de una comunidad biológica. Principalmente enfocada en caracterizar taxones específicos, aunque [[Ficetola, Taberlet 2023]] plantean que puede ser utilizada para caracterizar comunidades enteras (problemas a resolver: uso de primers que puedan amplificar para todos los taxones de la comunidad, bases de datos de referencia incompletas)
### Marcadores moleculares
Barcoding, uso de genes marcadores (alta variabilidad, presentes en múltiples taxones), gen 16S en bacterias, 18S y COI en animales, rbcL y matK en plantas (cloroplastos) [[Bazinet, Cummings 2012]], [[Deiner 2017]].
Secuencia utilizada como marcador, depende del conjunto de primers empleados. La mayoría de los primers son específicos para grupos taxonómicos acotados [[Ficetola, Taberlet 2023]]. Los primers universales son difíciles de diseñar y pueden aportar baja resolución taxonómica.
La cantidad de información contenida en un marcador dado puede no ser la misma para distintos grupos taxonómicos [[Deiner 2017]].
## Clasificación de secuencias
Aprendizaje supervisado consiste en utilizar instancias de etiqueta/clase conocida para clasificar instancias desconocidas. Existen tres principales enfoques para la clasificación de secuencias mediante aprendizaje supervisado: clasificación por similitud de secuencias, por composición de secuencias y mediante modelos filogenéticos [[Bazinet, Cummings 2012]]. Los metodos de clasificación pueden ofrecer una métrica de confianza para las etiquetas asignadas que brinda contexto a los resultados.

### Bases de datos de referencia
### Importancia del dataset
El dataset de referencia empleado en la clasificación tiende a tener un impacto en la sensibilidad/precisión de los clasificadores [[Bazinet, Cummings 2012]].