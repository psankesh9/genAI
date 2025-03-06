from fastapi import APIRouter, HTTPException
from app.models.llm import Query
import openai

router = APIRouter()

openai.api_key = "YOUR_API_KEY"

@router.get("/ask")
def ask_get(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    return {"response": response.choices[0].message.content}

@router.post("/ask")
def ask_post(query: Query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query.query}]
    )
    return {"response": response.choices[0].message["content"]}
