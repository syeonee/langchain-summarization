from pydantic import BaseModel

class Summary(BaseModel):
    news: str
    language: str
    chrLimit: str