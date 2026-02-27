FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# If using custom model, uncomment:
# COPY your_custom_model.pt .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]