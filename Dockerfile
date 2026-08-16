FROM python:3.11-slim

# git is required by query_code_diffs (git diff in pipeline_repo)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# Bundle the target-system SQLite databases and pipeline_repo.
# For production, replace with a PersistentVolumeClaim mount.
COPY target-system/ ./target-system/

# Set the pipeline_repo as a git repo so query_code_diffs can run git diff.
# If the subdirectory already has its own .git this is a no-op.
RUN git -C /app/target-system/pipeline_repo init --quiet 2>/dev/null || true

ENV PYTHONPATH=/app/src
# tools.py resolves data paths from __file__, so this is only needed if you
# mount the DBs somewhere other than target-system/data/.
# ENV MLSYS_DATA_DIR=/data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app/src"]
