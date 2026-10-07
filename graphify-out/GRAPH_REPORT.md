# Graph Report - dentilin  (2026-10-06)

## Corpus Check
- Corpus is ~36,793 words - fits in a single context window. You may not need a graph.

## Summary
- 129 nodes · 228 edges · 12 communities (9 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.94)
- Token cost: no disponible; el entorno no expone el consumo de los agentes locales. Los ceros del registro indican falta de medición, no ausencia de consumo.

## Community Hubs (Navigation)
- Interfaz e inferencia dental
- Datos y entrenamiento
- Arquitectura y dependencias
- Distribución del modelo
- Ejemplos y flujo documentado
- Numeración FDI y pruebas
- Ejecución con Docker
- Resultados y limitaciones
- Dependencias de la interfaz
- Dependencias de PyTorch

## God Nodes (most connected - your core abstractions)
1. `Dentilin README` - 29 edges
2. `run_fdi_inference()` - 13 edges
3. `render()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Escala de grises y lienzo 1024 × 512 con relleno y tres canales` --conceptually_related_to--> `preprocess_image()`  [INFERRED]
  README.md → runtime/inference.py
- `Inferencia en memoria y restauración de coordenadas originales` --conceptually_related_to--> `run_fdi_inference()`  [INFERRED]
  README.md → runtime/inference.py
- `Mask R-CNN ResNet-50 FPN v2` --conceptually_related_to--> `build_instance_model()`  [INFERRED]
  README.md → runtime/model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Etapas de desarrollo dental** — readme_entrenamiento_global, readme_entrenamiento_instancias, readme_entrenamiento_fdi [EXTRACTED 1.00]

## Communities (12 total, 1 thin omitted)

### Community 0 - "Interfaz e inferencia dental"
Cohesion: 0.12
Nodes (25): cache_resource, inference_mode, io, ndarray, pandas, draw_teeth_layer(), fdi_to_spanish_name(), highlight_teeth() (+17 more)

### Community 1 - "Datos y entrenamiento"
Cohesion: 0.17
Nodes (24): Dentilin README, Adult tooth segmentation y colecciones complementarias: 2,197 imágenes, Alcance experimental y validación externa FDI pendiente, Brillo, contraste y reflexión horizontal con intercambio de cuadrantes FDI, Children's dental caries segmentation: 193 imágenes, Preentrenamiento COCO, Dentilin: segmentación y numeración dental experimental, Ejecución local Python 3.12, Streamlit y selección de dispositivo (+16 more)

### Community 2 - "Arquitectura y dependencias"
Cohesion: 0.14
Nodes (17): csv, importlib.util, numpy, pathlib, PIL, build_instance_model(), checkpoint_model_args(), Any (+9 more)

### Community 3 - "Distribución del modelo"
Cohesion: 0.18
Nodes (11): argparse, hashlib, json, os, shutil, main(), Download or verify the exact model described by models/manifest.json., verify() (+3 more)

### Community 4 - "Ejemplos y flujo documentado"
Cohesion: 0.27
Nodes (11): Imagen de ejemplo 1, Imagen de ejemplo: ejemplo_(2).jpg, Radiografía dental panorámica, Imagen de ejemplo 3, Imagen de ejemplo: ejemplo_(4).jpg, Imagen de ejemplo 5, Grafo interactivo Graphify: componentes, conceptos, imágenes y conexiones, Cinco imágenes en img_ejemplo para probar interfaz y análisis (+3 more)

### Community 5 - "Numeración FDI y pruebas"
Cohesion: 0.32
Nodes (5): postprocess_fdi_prediction(), prepare_prediction(), Tensor, Elimina duplicados entre clases y limita cada número FDI a una aparición., RuntimeTests

### Community 6 - "Ejecución con Docker"
Cohesion: 0.40
Nodes (6): Configuración Docker Compose, Servicio download-model bajo perfil tools, Servicio teeth: CPU, puerto local 8501 y modelo de solo lectura, Consulta: disponibilidad del modelo en un clon Git, Modelo fdi.pt excluido de Git; descarga sin URL configurada, Ejecución Docker Compose en localhost:8501

### Community 7 - "Resultados y limitaciones"
Cohesion: 0.33
Nodes (6): Confusiones entre posiciones FDI vecinas, docs\RESULTADOS_FDI.md, Evaluación histórica interna 32 radiografías F1 numerado 0.9246, FDI único mayor confianza umbral 0.5 calibrado en validación, NMS descartado por eliminar dientes adyacentes, Validación externa y revisión odontológica pendientes

### Community 8 - "Dependencias de la interfaz"
Cohesion: 0.40
Nodes (4): numpy==1.26.4, pandas==2.3.3, Pillow==10.4.0, streamlit==1.28.2

## Knowledge Gaps
- **13 isolated node(s):** `Evaluación histórica interna 32 radiografías F1 numerado 0.9246`, `Confusiones entre posiciones FDI vecinas`, `Validación externa y revisión odontológica pendientes`, `torch==2.12.1`, `torchvision==0.27.1` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 44 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Dentilin README` connect `Datos y entrenamiento` to `Ejemplos y flujo documentado`, `Ejecución con Docker`?**
  _High betweenness centrality (0.289) - this node is a cross-community bridge._
- **Why does `run_fdi_inference()` connect `Interfaz e inferencia dental` to `Arquitectura y dependencias`, `Ejemplos y flujo documentado`, `Numeración FDI y pruebas`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Why does `postprocess_fdi_prediction()` connect `Numeración FDI y pruebas` to `Interfaz e inferencia dental`, `Datos y entrenamiento`, `Arquitectura y dependencias`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
## Integridad de la actualización

- Cinco imágenes de ejemplo incorporadas, con referencias desde la documentación.
- 129 nodos y 228 conexiones; todos los extremos de las conexiones existen.
- La extracción detallada conserva 233 relaciones. La vista de grafo agrupa cinco relaciones paralelas entre los mismos nodos; no hay referencias a nodos inexistentes ni duplicados exactos.
- El alcance corresponde únicamente al proyecto dentilin.
