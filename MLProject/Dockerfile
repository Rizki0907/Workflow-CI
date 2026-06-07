FROM python:3.12.7-slim

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn dagshub

COPY . .

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "8080", "--no-conda"]