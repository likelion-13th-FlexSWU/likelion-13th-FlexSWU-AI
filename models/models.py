from pydantic import BaseModel
from typing import List, Optional

class UserKeywords(BaseModel):
    keywords: list[str]

class UserKeywordsWithLocation(BaseModel):
    keywords: list[str]
    query: str

class PreviousPlace(BaseModel):
    name: str
    address: Optional[str]

class RecommendationRequest(BaseModel):
    mood_keywords: Optional[List[str]] = None
    place_category: Optional[str] = None
    search_query: str
    previous_places: Optional[List[PreviousPlace]] = None

# 20개 카테고리 선택 횟수를 담을 모델
class UserBehaviorData(BaseModel):
    user_id: int
    survey_count: int = 0
    family_friendly: int = 0
    date_friendly: int = 0
    pet_friendly: int = 0
    solo_friendly: int = 0
    quiet: int = 0
    cozy: int = 0
    focus: int = 0
    noisy: int = 0
    lively: int = 0
    diverse_menu: int = 0
    book_friendly: int = 0
    plants: int = 0
    trendy: int = 0
    photo_friendly: int = 0
    good_view: int = 0
    spacious: int = 0
    aesthetic: int = 0
    long_stay: int = 0
    good_music: int = 0
    exotic: int = 0