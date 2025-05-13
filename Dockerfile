# imagem base Python
FROM python:3.11-slim

# setar workdir
WORKDIR /app

# copiar só requirements e instalar deps (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copiar todo o código
COPY . .

# expor porta e rodar uvicorn
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
