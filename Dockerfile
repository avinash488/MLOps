FROM python:3.11-slim

WORKDIR /app

COPY api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/models/ ./src/models/
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]