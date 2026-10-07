FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    DENTILIN_DEVICE=cpu DENTILIN_MODEL_DIR=/app/models HOME=/home/app
WORKDIR /app
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements.txt
RUN python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install -r requirements.txt \
    && useradd --create-home --uid 10001 app
COPY --chown=app:app app.py app.py
COPY --chown=app:app runtime/ runtime/
COPY --chown=app:app metrics/ metrics/
COPY --chown=app:app tools/ tools/
COPY --chown=app:app models/manifest.json models/manifest.json
COPY --chown=app:app .streamlit/ .streamlit/

USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
