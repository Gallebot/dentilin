# Resultados de la Fase 3: numeración FDI

## Decisión

La Fase 3 supera el criterio técnico para continuar. El modelo localiza, segmenta y
numera la mayoría de los dientes permanentes en la prueba interna. Se conserva
`runs/fdi_numbering_v1/best.pt` y se aplica la regla anatómica de una sola predicción
por número FDI en cada panorámica.

Esta evaluación es interna y no constituye validación clínica.

## Entrenamiento

- Entrenamiento: 158 radiografías y 4,177 dientes.
- Validación: 43 radiografías y 1,169 dientes.
- Épocas: 30.
- Mejor F1 numerado de validación durante la ejecución original: 0.9059.
- Transferencia: 426 tensores directos, 4 expandidos y 2 cabezas de clasificación
  inicializadas de nuevo.

## Resultado inicial

Sin restricción anatómica, Mask R-CNN podía conservar varias predicciones del mismo
diente bajo clases FDI diferentes:

| Métrica | Valor |
|---|---:|
| Referencias | 882 |
| Predicciones | 965 |
| Correctamente numeradas | 828 |
| Precisión numerada | 0.8580 |
| Recall numerado | 0.9388 |
| F1 numerado | 0.8966 |
| Recall de localización | 0.9898 |
| Exactitud FDI entre localizadas | 0.9164 |
| IoU medio correcto | 0.8604 |

## Calibración en validación

Se compararon únicamente sobre validación la salida original, NMS entre clases con
umbrales 0.3, 0.5 y 0.7, varios umbrales de confianza y la restricción de FDI único.
La mejor alternativa fue conservar la predicción de mayor confianza de cada número
FDI, sin NMS entre clases:

- F1 de validación original: 0.9054.
- F1 con FDI único: 0.9284.
- NMS 0.7 más FDI único: 0.9253.

El NMS se descartó porque podía eliminar dientes adyacentes cuyas cajas se superponen.
El umbral de confianza permanece en 0.5.

## Resultado final postprocesado

| Métrica | Valor |
|---|---:|
| Radiografías | 32 |
| Referencias | 882 |
| Predicciones | 883 |
| Correctamente numeradas | 816 |
| Precisión numerada | 0.9241 |
| Recall numerado | 0.9252 |
| F1 numerado | **0.9246** |
| Recall de localización | 0.9649 |
| Exactitud FDI entre localizadas | 0.9448 |
| IoU medio correcto | 0.8607 |

La regla reduce doce aciertos respecto de la salida sin filtrar, pero elimina muchos
más duplicados y falsos positivos; por eso mejora el equilibrio global y hace que el
conteo previsto, 883, sea prácticamente igual al real, 882.

## Errores restantes

Las confusiones se concentran principalmente entre posiciones vecinas. Los recalls
más bajos son:

- FDI 14 y 24: 0.8276;
- FDI 44: 0.8333;
- FDI 13: 0.8750;
- FDI 15 y 37: 0.8846.

Los casos visualmente más difíciles contienen ausencias múltiples, prótesis,
implantes, terceros molares o alteraciones de la secuencia dental. Antes de uso
clínico se requiere validación externa y revisión odontológica de estas confusiones.

## Artefactos finales

- `fdi_evaluation_internal_test_postprocessed.json`: métricas finales.
- `fdi_evaluation_internal_test_postprocessed_per_image.csv`: detalle por estudio.
- `fdi_evaluation_internal_test_postprocessed_per_fdi.csv`: métricas por posición.
- `fdi_evaluation_internal_test_postprocessed_confusion.csv`: confusiones FDI.
- `fdi_inspection_internal_test_postprocessed/`: veinte peores paneles finales.

Los scripts de entrenamiento, evaluación e inspección aplican ahora FDI único por
defecto. El NMS entre clases queda disponible como opción experimental, pero está
desactivado.
