from pydantic import BaseModel, validator

class Query(BaseModel):
    query: str

    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v
