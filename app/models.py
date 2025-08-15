from pydantic import BaseModel
from typing import List

class UserKeywords(BaseModel):
    keywords: list[str]

class UserKeywordsWithLocation(BaseModel):
    keywords: list[str]
    query: str

class RecommendationRequest(BaseModel):
    mood_keywords: List[str]
    place_category: str
    search_query: str