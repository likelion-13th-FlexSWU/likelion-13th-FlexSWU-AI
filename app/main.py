# app/main.py

from fastapi import FastAPI, HTTPException, Query
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Any, List

# ai_service는 그대로 사용
from .ai_service import get_gpt_embedding, generate_place_description
from .map_service import geocode_address
from .models import UserKeywords

app = FastAPI()

# 카카오 지도 API 호출을 대신할 가짜 데이터
MOCK_PLACES = [
    {
        "place_name": "고즈넉한 카페",
        "category_name": "카페",
        "address_name": "서울특별시 종로구 어딘가"
    },
    {
        "place_name": "미술관 옆 갤러리",
        "category_name": "문화, 예술",
        "address_name": "서울특별시 중구 어딘가"
    },
    {
        "place_name": "숲속 한정식집",
        "category_name": "음식점 > 한정식",
        "address_name": "서울특별시 강남구 어딘가"
    }
]

# 카카오 지도 API 호출 함수 대신 이 함수 사용
async def mock_search_places_kakao(keywords: List[str]):
    return MOCK_PLACES

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    return cosine_similarity([vec1], [vec2])[0][0]

@app.post("/recommendations", response_model=Dict[str, Any])
async def get_recommendations(user_input: UserKeywords):
    try:
        keywords = user_input.keywords
        if not keywords:
            raise HTTPException(status_code=400, detail="키워드를 제공해주세요.")

        # 1. 사용자 키워드 임베딩
        user_keyword_embedding = await get_gpt_embedding(keywords)
        
        # 2. 가상의 장소 데이터 사용
        places = await mock_search_places_kakao(keywords)

        recommendations = []
        for place in places:
            # 3. GPT로 장소 설명 생성 및 임베딩
            description = await generate_place_description(place)
            description_embedding = await get_gpt_embedding(description)
            
            # 4. 유사도 계산
            similarity_score = calculate_cosine_similarity(user_keyword_embedding, description_embedding)
            
            recommendations.append({
                "place_name": place.get("place_name"),
                "category_name": place.get("category_name"),
                "address_name": place.get("address_name"),
                "description": description,
                "similarity_score": similarity_score
            })
            
        # 5. 유사도 점수 정렬 및 Top-5 반환
        sorted_recommendations = sorted(recommendations, key=lambda x: x["similarity_score"], reverse=True)
        return {"recommendations": sorted_recommendations[:5]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
    

@app.get("/geocode")
async def geocode(query: str = Query(..., min_length=1, description="예: '서울 중랑구'")):
    result = await geocode_address(query)
    if not result:
        raise HTTPException(status_code=404, detail="주소를 찾지 못했거나 API 키가 없습니다.")
    return result


@app.get("/")
def read_root():
    return {"message": "Hello FastAPI!"}
