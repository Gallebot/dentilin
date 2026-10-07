from __future__ import annotations
import hashlib
import json
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from .inference import (
    draw_teeth_layer, fdi_to_spanish_name, highlight_teeth,
    load_fdi_model, preprocess_image, run_fdi_inference,
)
from .settings import APP_ROOT, device, model_path


@st.cache_resource(show_spinner='Cargando el detector de dientes…')
def get_model(path: str, modified: int):
    selected_device = device()
    model, size = load_fdi_model(Path(path), selected_device)
    return model, size, selected_device


def prediction_document(original: Image.Image, teeth: list[dict]) -> dict:
    return {
        'format_version': 1,
        'kind': 'teeth',
        'image': {'width': original.width, 'height': original.height},
        'predictions': [
            {key: value for key, value in tooth.items() if key != 'mask'}
            for tooth in teeth
        ],
    }


def render() -> None:
    st.set_page_config(page_title='Dentilin · Detector de dientes', page_icon='🦷', layout='wide')
    st.title('Detector de dientes')
    st.caption('Detección, segmentación individual y numeración FDI de dientes permanentes.')
    checkpoint = model_path()
    with st.sidebar:
        threshold = st.slider('Confianza mínima de dientes', 0.20, 0.90, 0.50, 0.05,
                              help='0.50 es el umbral de la evaluación FDI final.')
        show_masks = st.checkbox('Mostrar dientes y numeración', value=True)
    with st.expander('Resultados y alcance', expanded=True):
        metrics_path = APP_ROOT / 'metrics' / 'fdi_internal_test.json'
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            columns = st.columns(4)
            for column, label, key in zip(columns, ('Precisión', 'Sensibilidad', 'F1', 'IoU máscara'),
                                          ('numbered_precision_at_iou50', 'numbered_recall_at_iou50', 'numbered_f1_at_iou50', 'mean_correct_mask_iou')):
                column.metric(label, f'{metrics[key]:.3f}')
            st.caption(f"Prueba interna: {metrics['images']} radiografías. Estas métricas no son la confianza de cada predicción.")
        st.write('Reconoce posiciones FDI 11–48. Puede confundir piezas vecinas, ausencias, prótesis, implantes y terceros molares atípicos.')
    if not checkpoint.is_file():
        st.error('Falta el modelo dental. Coloca fdi.pt en la carpeta models o utiliza tools/download_model.py con la URL del modelo.')
        st.info('Las instrucciones de instalación y descarga están en el README de esta aplicación.')
        return
    uploaded = st.file_uploader('Cargar radiografía panorámica', type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'])
    if uploaded is None:
        return
    if uploaded.size > 25 * 1024 * 1024:
        st.error('La imagen supera el límite de 25 MB.')
        return
    payload = uploaded.getvalue()
    key = (hashlib.sha256(payload).hexdigest(), threshold, str(checkpoint), checkpoint.stat().st_mtime_ns)
    if st.button('Analizar dientes', type='primary'):
        try:
            with Image.open(BytesIO(payload)) as source:
                source.verify()
            with Image.open(BytesIO(payload)) as source:
                source_image = source.copy()
            model, size, selected_device = get_model(str(checkpoint), checkpoint.stat().st_mtime_ns)
            tensor, original, geometry = preprocess_image(source_image, size)
            with st.spinner('Localizando y numerando dientes…'):
                teeth = run_fdi_inference(model, tensor, geometry, selected_device, threshold)
            st.session_state['teeth_result'] = (key, original, teeth, selected_device.type.upper())
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
            st.error(f'No se pudo analizar la radiografía: {error}')
            return
    saved = st.session_state.get('teeth_result')
    if saved is None or saved[0] != key:
        st.info('Pulsa «Analizar dientes» para procesar esta imagen con el umbral seleccionado.')
        return
    _, original, teeth, processing_device = saved
    st.metric('Dientes detectados', len(teeth))
    st.caption(f'{processing_device} · {original.width} × {original.height} px')
    image_container = st.container()
    table = pd.DataFrame({
        'Resaltar': [False] * len(teeth),
        'FDI': [int(row['fdi']) for row in teeth],
        'Pieza': [fdi_to_spanish_name(int(row['fdi'])) for row in teeth],
        'Confianza (%)': [float(row['score']) * 100 for row in teeth],
    })
    edited = st.data_editor(table, hide_index=True, use_container_width=True,
                            disabled=['FDI', 'Pieza', 'Confianza (%)'],
                            key=f'teeth_selection_{key[0]}_{threshold}',
                            column_config={'Resaltar': st.column_config.CheckboxColumn('Resaltar'),
                                           'Confianza (%)': st.column_config.NumberColumn('Confianza (%)', format='%.1f')})
    selected = set(edited.loc[edited['Resaltar'].astype(bool), 'FDI'].astype(int))
    result = draw_teeth_layer(original, teeth) if show_masks else original.copy()
    if show_masks:
        result = highlight_teeth(result, teeth, selected)
    with image_container:
        st.image(result, caption='Dientes y numeración FDI' if show_masks else 'Radiografía original', use_column_width=True)
    buffer = BytesIO()
    result.save(buffer, format='PNG')
    first, second, third = st.columns(3)
    stem = Path(uploaded.name).stem
    first.download_button('Descargar imagen', buffer.getvalue(), f'{stem}_dientes.png', 'image/png')
    second.download_button('Descargar tabla CSV', table.drop(columns='Resaltar').to_csv(index=False).encode('utf-8-sig'), f'{stem}_dientes.csv', 'text/csv')
    third.download_button('Descargar resultados JSON', json.dumps(prediction_document(original, teeth), ensure_ascii=False, indent=2), f'{stem}_dientes.json', 'application/json')
    st.caption('La imagen se procesa en memoria y no se guarda automáticamente. Los resultados son experimentales y requieren interpretación profesional.')
