FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY fever_platform/ fever_platform/
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "fever_platform.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
