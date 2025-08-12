from pydantic import BaseModel

class UserKeywords(BaseModel):
    keywords: list[str]