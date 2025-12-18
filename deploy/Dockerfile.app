FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-vision.txt /app/
RUN pip install --no-cache-dir -r requirements-vision.txt

# Copy source
COPY . /app

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "app.py"]
