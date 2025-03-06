# 🚀 FastAPI + MLOps: API Deployment & LLM Integration

## 📌 Overview

This project demonstrates how to:
- Build a **FastAPI application** for serving LLM-based responses.
- Implement **GET, POST, PUT, DELETE endpoints**.
- Structure the project following best practices.
- Use **Docker** for deployment.
- Track model experiments with **MLflow & DVC**.
- Monitor data & model drift with **Evidently AI**.

## 📂 Project Structure
fastapi_project/ │── main.py # Entry point for FastAPI app │── requirements.txt # Dependencies (FastAPI, Uvicorn, OpenAI, etc.) │── Dockerfile # Docker container setup │ ├── app/ │ ├── routes/ # API routes │ │ ├── llm.py # LLM API endpoints │ │ ├── health.py # Health check endpoints │ │ ├── items.py # CRUD routes │ │ │ ├── models/ # Pydantic schemas │ │ ├── llm.py # Request models for LLM API │ │ ├── items.py # Request models for items │ │ │ ├── services/ # Business logic │ │ ├── openai_service.py # OpenAI API wrapper │ │ ├── database.py # Database interactions (if needed) │ └── tests/ # API tests ├── test_main.py # Test main API endpoints ├── test_llm.py # Test LLM API endpoints

shell
Copy
Edit

## 🛠 Installation & Setup

### 1️⃣ Install Dependencies
pip install -r requirements.txt

graphql
Copy
Edit

### 2️⃣ Run the FastAPI Server
uvicorn main:app --reload

bash
Copy
Edit
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc UI: http://127.0.0.1:8000/redoc

## 🚀 FastAPI Implementation

### `main.py`: Entry Point
from fastapi import FastAPI from app.routes import llm, health, items

app = FastAPI()

app.include_router(llm.router) app.include_router(health.router) app.include_router(items.router)

@app.get("/") def home(): return {"message": "Welcome to FastAPI!"}

shell
Copy
Edit

### `app/routes/llm.py`: LLM API Routes
from fastapi import APIRouter, HTTPException from app.models.llm import Query import openai

router = APIRouter() openai.api_key = "YOUR_API_KEY"

@router.get("/ask") def ask_get(query: str): response = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[{"role": "user", "content": query}] ) return {"response": response.choices[0].message.content}

@router.post("/ask") def ask_post(query: Query): response = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[{"role": "user", "content": query.query}] ) return {"response": response.choices[0].message["content"]}

shell
Copy
Edit

### `app/models/llm.py`: Pydantic Model
from pydantic import BaseModel, validator

class Query(BaseModel): query: str

python
Copy
Edit
@validator('query')
def validate_query(cls, v):
    if not v or len(v.strip()) == 0:
        raise ValueError("Query cannot be empty")
    return v
shell
Copy
Edit

### `app/routes/health.py`: Health Check
from fastapi import APIRouter

router = APIRouter()

@router.get("/healthcheck") def healthcheck(): return {"status": "API is running smoothly!"}

shell
Copy
Edit

## 🐳 Dockerizing FastAPI

### Dockerfile
FROM python:3.9 WORKDIR /app

COPY requirements.txt /app RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000 CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

shell
Copy
Edit

### Docker Commands
Build Docker Image
docker build -t fastapi-llm-app .

Run Docker Container
docker run -p 8000:8000 fastapi-llm-app

shell
Copy
Edit

## 📊 MLOps Components

### 🔹 MLflow: Experiment Tracking
import mlflow mlflow.set_experiment("MyExperiment")

with mlflow.start_run(): mlflow.log_param("learning_rate", 0.01) mlflow.log_metric("accuracy", 0.95) mlflow.log_artifact("model.pkl") mlflow.end_run()

shell
Copy
Edit

### 🔹 DVC: Dataset Versioning
dvc init dvc add data.csv git add data.csv.dvc .gitignore git commit -m "Versioning dataset with DVC"

shell
Copy
Edit

### 🔹 Evidently AI: Model Monitoring
import pandas as pd from evidently.test_suite import TestSuite from evidently.tests import DataDriftTestPreset

reference_data = pd.read_csv("training_data.csv") current_data = pd.read_csv("live_data.csv")

data_drift_suite = TestSuite(tests=[DataDriftTestPreset()]) data_drift_suite.run(reference_data=reference_data, current_data=current_data) data_drift_suite.save_html("data_drift_report.html")

markdown
Copy
Edit

## 🎯 Final Takeaways

✅ **FastAPI**: API for serving ML models and LLM queries.  
✅ **MLflow**: Logs model experiments and results.  
✅ **DVC**: Version controls datasets and models.  
✅ **Evidently AI**: Detects data drift in production.  
✅ **Docker**: Containerizes and deploys the FastAPI app.  

🚀 **Now you're ready to deploy your ML-powered APIs with FastAPI!**






