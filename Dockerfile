FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY btc_4h_knn_v12_lean.py .
COPY btc_4h_v1_improved_manual.py .
COPY btc_4h_v1_cloud_runner.py .

CMD ["python", "btc_4h_v1_cloud_runner.py"]
