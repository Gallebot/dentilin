# Dentilin

## Resumen

Dentilin es una aplicación experimental para localizar dientes permanentes, segmentar sus contornos y asignarles numeración FDI en radiografías panorámicas. El análisis produce una máscara, una caja delimitadora, una posición dental y una puntuación de confianza por pieza. La interfaz permite visualizar los resultados, seleccionar dientes y descargar una imagen anotada o tablas de predicciones.

El desarrollo se organizó en tres etapas: reconocimiento de la región dental, separación de dientes individuales y aprendizaje de las posiciones FDI. El detector final utiliza Mask R-CNN con ResNet-50 y una pirámide de características FPN. En la prueba interna de numeración, realizada sobre 32 radiografías con 882 dientes anotados, alcanzó un F1 de 0.9246 después del postprocesamiento.

## Objetivo del proyecto

La segmentación global identifica los píxeles que pertenecen a dientes. La segmentación de instancias distingue cada pieza y conserva su contorno individual. La numeración añade una identidad anatómica a cada instancia, de modo que el resultado describe tanto su ubicación como la posición dental predicha.

El modelo aprende las 32 posiciones de la dentición permanente: 11–18, 21–28, 31–38 y 41–48. Internamente utiliza 33 clases, correspondientes al fondo y a esas 32 posiciones. La numeración es una predicción del modelo y puede contener errores, especialmente en piezas vecinas o anatomías poco representadas en los datos.

## Datos utilizados

### Conjunto para la segmentación global

La etapa inicial reunió radiografías y anotaciones de varias colecciones. Se utilizaron máscaras de píxeles y polígonos; estos últimos se convirtieron en máscaras binarias para representar la región dental. Los siguientes conteos corresponden a los registros únicos y válidos conservados en la preparación local:

| Colección | Radiografías | Uso |
|---|---:|---|
| [Adult tooth segmentation dataset y sus colecciones complementarias](https://springernature.figshare.com/articles/dataset/Children_s_Dental_Panoramic_Radiographs_Dataset/21621705?file=38322366) | 2,197 | Desarrollo y prueba interna de segmentación global |
| [Children's dental caries segmentation dataset, incluido el material complementario](https://springernature.figshare.com/articles/dataset/Children_s_Dental_Panoramic_Radiographs_Dataset/21621705?file=38322366) | 193 | Desarrollo y prueba interna de segmentación global |
| [Teeth Segmentation on dental X-ray images, con polígonos Supervisely](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) | 595 | Desarrollo y prueba interna de segmentación global |
| [OPG Teeth Segmentation and Numbering](https://zenodo.org/records/10538750) | 248 | Desarrollo y prueba interna de segmentación global |
| [Dental Panoramic Radiography Dataset for Pixel level Semantic Segmentation of Teeth, identificado como Mendeley 2026](https://data.mendeley.com/datasets/jrz4nj82zv/1) | 329 | Prueba externa de segmentación global |
| **Total** | **3,562** | |

Antes de la limpieza se encontraron 4,062 pares de imagen y anotación. La revisión detectó 479 grupos de duplicados exactos, que reunían 972 archivos, y siete muestras inválidas. Tras la selección de una muestra válida por grupo quedaron 3,562 radiografías: 2,269 para entrenamiento, 480 para validación, 484 para prueba interna y 329 para prueba externa.

Las 329 radiografías externas se reservaron para evaluar la segmentación global. Estos conteos describen la línea base inicial; el entrenamiento de la numeración dental utilizó el conjunto específico detallado a continuación.

### Conjunto para dientes individuales y numeración FDI

La separación de instancias y la numeración se desarrollaron con [**OPG Teeth Segmentation and Numbering**](https://zenodo.org/records/10538750), que aporta polígonos por diente y etiquetas FDI en formato LabelMe.

La colección local contenía 250 radiografías. Se excluyeron dos con anotaciones vacías y tres con etiquetas FDI ausentes o inválidas, dejando 245 imágenes para la etapa de instancias. Para aprender la numeración se excluyeron además 12 radiografías con números FDI repetidos en sus anotaciones. El conjunto final de numeración quedó compuesto por 233 radiografías y 6,228 dientes.

| Etapa | Partición | Radiografías | Dientes anotados |
|---|---|---:|---:|
| Separación de instancias | Entrenamiento | 167 | 4,406 |
| Separación de instancias | Validación | 45 | 1,216 |
| Separación de instancias | Prueba interna | 33 | 909 |
| Numeración FDI | Entrenamiento | 158 | 4,177 |
| Numeración FDI | Validación | 43 | 1,169 |
| Numeración FDI | Prueba interna | 32 | 882 |

Las 32 posiciones FDI estuvieron representadas en entrenamiento, con entre 76 y 157 ejemplos por posición.

## Preparación y control de calidad

La preparación verificó la correspondencia entre imágenes y anotaciones, la lectura de las radiografías, las dimensiones de las máscaras y la validez de los polígonos. Los datos originales se conservaron y las exclusiones se registraron durante la preparación.

La detección de duplicados utilizó SHA-256 de los archivos de imagen. Las particiones se asignaron de forma determinista a partir de ese hash, con proporciones objetivo de 70 % para entrenamiento, 15 % para validación y 15 % para prueba interna. Los tamaños efectivos variaron por la distribución de los hashes y las exclusiones posteriores. Las copias exactas de una misma imagen recibieron la misma partición.

Para el entrenamiento del detector, las radiografías se convirtieron a escala de grises y se ajustaron a un lienzo de **1024 × 512 píxeles**, conservando la proporción mediante redimensionamiento y relleno. Las máscaras y las cajas se transformaron con la misma geometría. La intensidad se escaló al intervalo de 0 a 1 y la imagen se replicó en tres canales para la entrada de Mask R-CNN.

El aumento de datos incluyó variaciones de brillo y contraste entre 0.85 y 1.15, y reflexión horizontal con probabilidad de 0.5. En la numeración FDI, la reflexión también intercambió las etiquetas de los cuadrantes 1↔2 y 3↔4, manteniendo la correspondencia anatómica entre imagen y etiqueta. Estas transformaciones se aplicaron al entrenamiento.

## Proceso de entrenamiento

### Primera etapa: segmentación de la región dental

Se entrenó una U-Net para predecir una máscara binaria de la región dental. La función de pérdida combinó entropía cruzada binaria y pérdida Dice; la optimización utilizó AdamW. Esta etapa estableció una línea base para medir la cobertura de los dientes en conjuntos de distinta procedencia.

La ejecución completó 60 épocas con CUDA. La mejor época fue la 15, con Dice de validación de 0.9086. El modelo seleccionado obtuvo Dice de 0.9057 en las 484 radiografías de prueba interna y de 0.8803 en las 329 radiografías externas.

### Segunda etapa: separación de dientes individuales

Se adaptó Mask R-CNN ResNet-50 FPN v2, inicialmente con pesos preentrenados en COCO, a dos clases: fondo y diente. La red aprendió a proponer regiones y producir una caja y una máscara por instancia dental.

El entrenamiento se ejecutó durante 40 épocas con lotes de dos imágenes y optimización SGD. La configuración documentada utilizó tasa de aprendizaje inicial de 0.005, momentum de 0.9 y regularización de 0.0005. Para la transferencia a numeración se seleccionó el estado final de esta ejecución, cuya AP75 de validación era superior a la del modelo guardado originalmente por AP50.

En la prueba interna de instancias, ese modelo recuperó 905 de los 909 dientes anotados, con 11 falsos positivos y cuatro falsos negativos. El F1 de instancias fue de 0.9918 a IoU 0.5. Esta métrica evalúa la separación dental sin exigir una numeración correcta.

### Tercera etapa: aprendizaje de la numeración FDI

La transferencia del detector final siguió la secuencia **preentrenamiento COCO → instancias dentales → numeración FDI**. A partir del modelo de instancias se adaptaron las cabezas de salida a las 32 posiciones permanentes más el fondo.

Se transfirieron 426 tensores directamente y se expandieron cuatro tensores de regresión de cajas y máscaras. Los dos tensores de la salida de clasificación FDI se inicializaron de nuevo. Así se conservaron las representaciones aprendidas para localizar y segmentar dientes mientras se aprendía su identidad anatómica.

| Parámetro | Configuración documentada de numeración |
|---|---|
| Arquitectura | Mask R-CNN ResNet-50 FPN v2 |
| Entrada | 1024 × 512 píxeles |
| Clases | 32 posiciones FDI y fondo |
| Épocas ejecutadas | 30 |
| Tamaño de lote | 2 imágenes |
| Optimizador | SGD |
| Tasa de aprendizaje inicial | 0.0025 |
| Momentum | 0.9 |
| Regularización de pesos | 0.0005 |
| Ajuste de la tasa de aprendizaje | Reducción por un factor de 10 cada 10 épocas |
| Semilla configurada | 42 |
| Dispositivo registrado | CUDA |

El ajuste optimizó conjuntamente la clasificación FDI, la regresión de cajas, la segmentación de máscaras y las propuestas de regiones. Se utilizó precisión mixta durante la ejecución con CUDA. El modelo se seleccionó por el F1 numerado de validación: la mejor época fue la 26, con un valor de 0.9059 en la ejecución original.

## Calibración y postprocesamiento

La revisión de predicciones identificó duplicados de una misma posición FDI. Se compararon alternativas sobre validación y se adoptó una regla que conserva la predicción de mayor confianza para cada número FDI. El umbral operativo de confianza se mantuvo en **0.5**.

También se evaluó la supresión de cajas superpuestas entre clases. Esta opción quedó desactivada porque podía eliminar dientes vecinos cuyas cajas se solapan. La regla de FDI único elevó el F1 de validación de 0.9054 a 0.9284 en la comparación posterior al entrenamiento.

El postprocesamiento modifica la selección de predicciones y conserva los pesos del modelo entrenado.

## Resultados del detector final

La prueba interna de numeración utilizó **32 radiografías con 882 dientes de referencia**. Para contabilizar un acierto se exigió coincidencia de la clase FDI y una intersección sobre unión de máscaras —IoU— de al menos 0.5, con emparejamiento de una predicción por referencia.

| Métrica | Resultado |
|---|---:|
| Predicciones conservadas | 883 |
| Dientes correctamente segmentados y numerados | 816 |
| Precisión numerada | 0.9241 |
| Recall numerado | 0.9252 |
| F1 numerado | **0.9246** |
| Recall de localización, sin exigir coincidencia FDI | 0.9649 |
| Exactitud FDI entre dientes localizados | 0.9448 |
| IoU medio de las máscaras correctamente numeradas | 0.8607 |

La precisión indica la proporción de predicciones que cumplen el criterio de acierto; el recall indica la proporción de dientes de referencia recuperados correctamente. El F1 resume el equilibrio entre ambas métricas. La puntuación de confianza de una pieza es una salida del modelo y no representa por sí sola una probabilidad clínica calibrada de acierto.

Antes de aplicar la restricción de FDI único, el F1 de prueba interna era de 0.8966, con 965 predicciones. La regla redujo los duplicados y elevó el F1 a 0.9246, aunque también redujo los aciertos de 828 a 816. El conteo final próximo al de referencia debe interpretarse junto con los errores de localización y numeración.

## Análisis de errores y alcance

Las confusiones restantes se concentran en posiciones vecinas. Entre los recalls más bajos se registraron 0.8276 para FDI 14 y 24, y 0.8333 para FDI 44. Las radiografías con ausencias múltiples, prótesis, implantes, terceros molares o secuencias dentales alteradas fueron especialmente difíciles.

La numeración está limitada a las posiciones de la dentición permanente. La dentición temporal no forma parte de las clases del detector final. La prueba externa de 329 imágenes corresponde a la línea base de segmentación global; la evaluación disponible de numeración FDI es interna.

La separación documentada se realizó por hash de imagen y no acredita independencia por paciente o centro. La deduplicación detecta archivos idénticos y no garantiza la eliminación de todas las imágenes visualmente equivalentes. Estos límites y el tamaño del conjunto de numeración deben considerarse al interpretar las métricas.

Los resultados describen una validación técnica experimental. La validación externa del detector FDI y la revisión odontológica de sus errores siguen pendientes antes de establecer su desempeño clínico.

## Funcionamiento de la aplicación

Una radiografía cargada se adapta a la geometría del modelo y se procesa con el detector entrenado. Las cajas y las máscaras resultantes se transforman a las dimensiones de la imagen original. La interfaz muestra la numeración, el nombre anatómico y la confianza de cada pieza, permite resaltar dientes seleccionados y ofrece descargas de imagen, tabla CSV y resultados JSON.

Las radiografías cargadas se procesan en memoria. La ejecución utiliza una exportación de inferencia del modelo, con sus pesos y configuración, sin iniciar entrenamientos durante el análisis.

## Imágenes de ejemplo

El proyecto incluye cinco imágenes en la carpeta [img_ejemplo](img_ejemplo/) para probar la interfaz y el análisis dental. Con la aplicación en ejecución, una imagen de esta carpeta puede seleccionarse mediante **Cargar radiografía panorámica** y procesarse con **Analizar dientes**. El resultado permite explorar los contornos, las posiciones FDI, las puntuaciones de confianza y las opciones de descarga.

Estas imágenes sirven como ejemplos de uso. Las métricas de entrenamiento y evaluación corresponden a los conjuntos descritos anteriormente.

## Grafo del proyecto

El [grafo interactivo](graphify-out/graph.html), generado con Graphify, representa las relaciones entre los componentes de la aplicación, los conceptos de su documentación y las imágenes de ejemplo. Permite explorar cómo se conectan la interfaz, el preprocesamiento, la carga del modelo, la inferencia y la presentación de resultados.

| Elemento | Interpretación |
|---|---|
| Nodos | Componentes de código, funciones, conceptos documentados e imágenes |
| Conexiones | Relaciones como importaciones, llamadas, referencias y vínculos conceptuales |
| Colores | Comunidades de elementos relacionados, agrupadas según sus conexiones |
| Confianza de las conexiones | `EXTRACTED` indica evidencia explícita; `INFERRED`, una relación inferida; `AMBIGUOUS`, una relación incierta |

El flujo principal comienza con la carga de una radiografía, continúa con su preparación y la ejecución del modelo, y termina con la transformación de las predicciones a las coordenadas originales y su visualización. El grafo documenta esta organización del software; las métricas del modelo se explican en las secciones de entrenamiento y resultados.

## Disponibilidad del modelo

La distribución del modelo de detección se gestiona mediante las [Releases privadas de dentilin-models](https://github.com/Gallebot/dentilin-models/releases). La descarga requiere una sesión de GitHub con acceso autorizado al repositorio.

La instalación del modelo consta de estos pasos:

1. Descarga de `fdi.pt` desde las [Releases del modelo](https://github.com/Gallebot/dentilin-models/releases).
2. Colocación del archivo en la carpeta `models` de la aplicación.
3. Verificación del tamaño y del SHA-256 con el manifiesto incluido:

```bash
python tools/download_model.py --verify-only
```

El modelo ocupa aproximadamente 185 MB. El manifiesto correspondiente ya está incluido en el proyecto y no requiere una descarga adicional después de clonar.

Con el modelo disponible, la ejecución requiere las dependencias de Python o Docker, según el método elegido. El análisis utiliza los pesos ya entrenados y no requiere descargar los conjuntos de entrenamiento ni volver a entrenar.

## Ejecución con Docker

Requisitos: Docker con Compose y el modelo de detección disponible.

Inicio del servicio:

```bash
docker compose up --build -d
```

La interfaz queda disponible en <http://localhost:8501>.

Detención del servicio:

```bash
docker compose down
```

El modelo se monta en modo lectura y las imágenes cargadas se procesan en memoria.

## Ejecución local

Requisitos: Python 3.12 y un entorno virtual.

Instalación de dependencias e inicio de la aplicación:

```bash
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
streamlit run app.py
```

El contenedor utiliza CPU. En ejecución local, `DENTILIN_DEVICE` permite seleccionar `auto`, `cpu` o `cuda`, con `auto` como valor predeterminado. CUDA requiere una instalación compatible de PyTorch.
