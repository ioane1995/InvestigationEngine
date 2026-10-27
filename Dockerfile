# Works on Hugging Face Spaces (sdk: docker, app_port 7860), Render, Koyeb, Fly.io, or locally.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed HF_HOME=/home/user/.cache/hf

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake the embedding model into the image so cold starts do not download it
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" && chown -R user:user /home/user/.cache || true

COPY --chown=user:user app ./app
COPY --chown=user:user eval ./eval
COPY --chown=user:user scripts ./scripts
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health')" || exit 1
CMD ["sh", "-c", "uvicorn app.api.asgi:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
