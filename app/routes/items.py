from fastapi import APIRouter
from app.models.items import Item

router = APIRouter()

fake_db = {}

@router.post("/items/")
def create_item(item: Item):
    fake_db[item.name] = item
    return {"message": "Item added", "item": item}

@router.get("/items/{item_name}")
def get_item(item_name: str):
    return fake_db.get(item_name, {"error": "Item not found"})
