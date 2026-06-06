FROM python:3.11-slim
WORKDIR /app
COPY requirements-gateway.txt .
RUN pip install --no-cache-dir -r requirements-gateway.txt
COPY lbus/ lbus/
COPY lgateway/ lgateway/
COPY lproto/ lproto/
CMD ["uvicorn", "lgateway.main:app", "--host", "0.0.0.0", "--port", "8443"]
