from fastapi import FastAPI
from app.routes import llm, health, items

app = FastAPI()

# Include all route modules
app.include_router(llm.router)
app.include_router(health.router)
app.include_router(items.router)

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}
