# app/main.py

from fastapi import FastAPI, HTTPException, Query
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from typing import Dict, Any, List, Optional, Literal
import random

# ai_service는 그대로 사용
from .ai_service import get_gpt_embedding, generate_place_description
from .map_service import geocode_address, search_places_around, search_places_rect_sweep
from .models import UserKeywords
from .models import UserKeywordsWithLocation
from .models import RecommendationRequest


app = FastAPI()

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    return cosine_similarity([vec1], [vec2])[0][0]

@app.post("/recommendations", response_model=Dict[str, Any])
async def get_recommendations(user_input: RecommendationRequest):
    try:
        user_mood_keywords = user_input.mood_keywords
        place_category = user_input.place_category
        search_query = user_input.search_query
        
        if not user_mood_keywords or not place_category or not search_query:
            raise HTTPException(status_code=400, detail="모든 필드를 제공해주세요.")

        # 1. 주소 지오코딩
        geo_info = await geocode_address(search_query)
        if not geo_info:
            raise HTTPException(status_code=404, detail="지오코딩 결과가 없습니다. 주소를 확인하세요.")
        
        x, y = geo_info["x"], geo_info["y"]

        # 2. rect-sweep으로 장소 검색 (기존 로직 대체)
        places = await search_places_rect_sweep(
            center_x=x,
            center_y=y,
            keyword=place_category,
            category_code=None, # 이 예시에서는 keyword만 사용
            total_limit=200,    # 총 200개 검색
            span_m=20000,
            step_m=4000,
            concurrency=8,
            restrict_by_query_text=search_query,
        )

        # 30개 랜덤 추출 > 랜덤 제외하려면 이 부분 주석
        if len(places) > 30:
            places = random.sample(places, 30)
        
        if not places:
            raise HTTPException(status_code=404, detail="검색된 장소가 없습니다. 키워드를 변경하거나 범위를 넓혀보세요.")

        # 3. 사용자 무드 키워드 임베딩
        user_keyword_embedding = await get_gpt_embedding(user_mood_keywords)

        # 4. 각 장소에 대해 GPT 설명 생성 및 임베딩을 비동기적으로 처리
        async def process_place(place):
            description = await generate_place_description(place, user_mood_keywords)
            description_embedding = await get_gpt_embedding(description)
            return {
                "place_name": place.get("place_name"),
                "category_name": place.get("category_name"),
                "address_name": place.get("address_name"),
                "description": description,
                "embedding": description_embedding
            }
        
        tasks = [process_place(place) for place in places]
        processed_places = await asyncio.gather(*tasks)

        # 5. 유사도 계산 및 정렬
        recommendations = []
        for p in processed_places:
            similarity_score = calculate_cosine_similarity(user_keyword_embedding, p["embedding"])
            recommendations.append({
                "place_name": p["place_name"],
                "category_name": p["category_name"],
                "address_name": p["address_name"],
                "description": p["description"],
                "similarity_score": similarity_score
            })
            
        # 6. 유사도 점수 내림차순 정렬 및 상위 5개 반환
        sorted_recommendations = sorted(recommendations, key=lambda x: x["similarity_score"], reverse=True)
        return {"recommendations": sorted_recommendations[:5]}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")

def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = it.get("id") or f"{it.get('place_name')}|{it.get('road_address_name') or it.get('address_name')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

@app.get("/geocode")
async def geocode(
    query: str = Query(..., description="예: 서울 중랑구"),
    keywords: List[str] = Query(..., description="예: keywords=카페&keywords=빵집", min_items=1),
    radius: int = Query(1500, ge=10, le=20000, description="검색 반경(m)"),
    size: int = Query(15, ge=1, le=15, description="키워드별 최대 15"),
    page: int = Query(1, ge=1, description="페이지"),
    sort: Literal["accuracy", "distance"] = Query("accuracy", description="정렬")
):
    geo = await geocode_address(query)
    if not geo:
        raise HTTPException(status_code=404, detail="지오코딩 결과가 없습니다. 주소를 확인하세요.")
    x, y = geo["x"], geo["y"]

    tasks = [
        search_places_around(x, y, kw, radius=radius, size=size, page=page, sort=sort)
        for kw in keywords
    ]
    results_list = await asyncio.gather(*tasks)

    per_keyword = {kw: _dedup(res) for kw, res in zip(keywords, results_list)}
    merged = _dedup([it for res in results_list for it in res])

    return {
        "query": query,
        "center": {"x": x, "y": y, "address": geo["address"]},
        "keywords": keywords,
        "radius_m": radius,
        "sort": sort,
        "per_keyword": per_keyword,
        "all": merged,
    }


@app.get("/rect-sweep")
async def rect_sweep(
    query: str = Query(..., description="예: 서울 종로구"),
    keyword: Optional[str] = Query(None, description="예: 카페 (없으면 카테고리로만 탐색)"),
    category_code: Optional[str] = Query(None, description="예: 카페 CE7, 편의점 CS2"),
    total_limit: int = Query(200, ge=1, le=1000),
    span_m: int = Query(20000, ge=1000, le=40000, description="전체 박스 크기(미터)"),
    step_m: int = Query(4000, ge=500, le=10000, description="타일 간격(미터)"),
    concurrency: int = Query(8, ge=1, le=32),
):
    # 1) 중심 좌표 구하기
    geo = await geocode_address(query)
    if not geo:
        raise HTTPException(status_code=404, detail="지오코딩 결과 없음")
    x, y = geo["x"], geo["y"]

    # 2) rect 스윕 호출
    places = await search_places_rect_sweep(
        center_x=x,
        center_y=y,
        keyword=keyword,
        category_code=category_code,
        total_limit=total_limit,
        span_m=span_m,
        step_m=step_m,
        concurrency=concurrency,
        restrict_by_query_text=query,
    )

    # 30개 랜덤 추출 > 랜덤 제외하려면 이 부분 주석
    if len(places) > 30:
        places = random.sample(places, 30)

    return {
        "query": query,
        "center": {"x": x, "y": y, "address": geo["address"]},
        "keyword": keyword,
        "category_code": category_code,
        "span_m": span_m,
        "step_m": step_m,
        "total": len(places),
        "places": places,
    }

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI!"}