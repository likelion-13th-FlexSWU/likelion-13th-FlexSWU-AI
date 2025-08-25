# app/main.py

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from typing import Dict, Any, List, Optional, Literal
import random
import numpy as np
import os, re
import joblib

from .ai_service import get_gpt_embedding, generate_place_description
from .map_service import geocode_address, search_places_around, search_places_rect_sweep, extract_sgg_and_optional_dong, geocode_address_weather
# from .models import UserKeywords
# from .models import UserKeywordsWithLocation
from .db_service import get_all_user_behavior_data 
from models.model_service import get_user_cluster, train_and_save_model, CATEGORIES
from models.models import RecommendationRequest, UserBehaviorData


app = FastAPI()

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    return cosine_similarity([vec1], [vec2])[0][0]

# 중복 제거 함수
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def place_key(name: str, address: str) -> str:
    return f"{normalize(name)}|{normalize(address or '')}"

def safe_attr(obj: Any, key: str):
    if isinstance(obj, dict):         # dict이면 .get
        return obj.get(key)
    return getattr(obj, key, None)    # 객체면 .name, .address 접근

@app.post("/recommendations", response_model=Dict[str, Any])
async def get_recommendations(
        user_input: RecommendationRequest,
        weather: Optional[bool] = Query(None, description="날씨 반영 여부")):
    try:
        print("previous_places 확인:", user_input.previous_places)
        user_mood_keywords = user_input.mood_keywords
        place_category = user_input.place_category
        search_query = user_input.search_query
        
        # weather 기반 추천일 때는 search_query만 필수
        if weather:
            if not search_query:
                raise HTTPException(status_code=400, detail="검색 쿼리를 제공해주세요.")
        else:
            # 기본 추천일 때는 mood_keywords / place_category / search_query 다 필수
            if not user_mood_keywords or not place_category or not search_query:
                raise HTTPException(status_code=400, detail="모든 필드를 제공해주세요.")


        # 1. 주소 지오코딩
        geo_info = await geocode_address(search_query)
        if not geo_info:
            raise HTTPException(status_code=404, detail="지오코딩 결과가 없습니다. 주소를 확인하세요.")
        
        x, y = geo_info["x"], geo_info["y"]

        sgg, dong = extract_sgg_and_optional_dong(geo_info["address"])

        # 구/동 여부에 따라 범위 설정 / 더 세부적으로 하고 싶으면 step_m을 더 작은 사각형으로 쪼개면 됨
        if dong:
            span_m = 10000  # 동 단위이면 더 좁은 범위
            # step_m = 1000
            # sample_tile_count = 30
        else:
            span_m = 20000  # 구 단위이면 넓은 범위
            # step_m = 1000
            # sample_tile_count = 60

        # 2. rect-sweep으로 장소 검색 (기존 로직 대체)
        # places = await search_places_rect_sweep(
        #     center_x=x,
        #     center_y=y,
        #     keyword=place_category,
        #     category_code=None, # 이 예시에서는 keyword만 사용
        #     total_limit=500,    # 몇 개 검색할 건지?
        #     span_m=span_m,
        #     step_m=step_m,
        #     concurrency=8,
        #     restrict_by_query_text=search_query,
        #     sample_tile_count=sample_tile_count
        # )
        # fallback 로직으로 확장

        # 날씨 기반 추천 로직
        if weather:
            # (1) 날씨 API 호출
            weather_status = await geocode_address_weather(x, y)  # clear, Clouds, Rain 중 하나 반환하도록 구현

            # (2) 해당 날씨에 맞는 키워드 목록
            keywords = WEATHER_KEYWORDS.get(weather_status, [])

            base_span = 10000 if dong else 20000
            base_step = 1000
            base_sample = 20 if dong else 30

            # (3) 각 키워드마다 rect-sweep 검색 (fallback X)
            all_candidates = []
            for kw in keywords:
                results = await search_places_rect_sweep(
                    center_x=x,
                    center_y=y,
                    keyword=kw,
                    category_code=None,
                    total_limit=50,
                    span_m=base_span,
                    step_m=base_step,
                    concurrency=8,
                    restrict_by_query_text=search_query,
                    sample_tile_count=base_sample
                )
                # 랜덤 샘플링으로 5개 뽑기
                if len(results) > 5:
                    sampled = random.sample(results, 5)
                else:
                    sampled = results

                all_candidates.extend(sampled)

            # (4) 최종 후보 중에서 랜덤 5개 추출
            if len(all_candidates) > 5:
                all_candidates = random.sample(all_candidates, 5)

            # 결과 정리
            trimmed_response = [
                {
                    "name": p.get("place_name"),
                    "category": p.get("category_name"),
                    "address_road": p.get("road_address_name"),
                    "address_ex": p.get("address_name"),
                    "phone": p.get("phone"),
                    "url": p.get("place_url")
                }
                for p in all_candidates
            ]
            return {"recommendations": trimmed_response}
        

        # 기존 추천 로직    
        places = await fetch_places_with_fallback(
            x=x,
            y=y,
            keyword=place_category,
            restrict_text=search_query,
            dong=(span_m <= 15000)
        )

        # 30개 랜덤 추출 > 랜덤 제외하려면 이 부분 주석
        if len(places) > 30:
            places = random.sample(places, 30)
        
        if not places:
            #raise HTTPException(status_code=404, detail="검색된 장소가 없습니다. 키워드를 변경하거나 범위를 넓혀보세요.")
            return {"recommendations": []}

        # --- 중복 제거 ---
        previous_places = user_input.previous_places or []

        # previous_places가 존재할 때만 중복 제거
        if previous_places:
            #  previous_places 장소들의 도로명 주소를 set 으로 정규화해서 저장
            previous_place_keys = {
                normalize(p.address or "")
                for p in previous_places
                if p.address # address 값이 none 이 아닌 경우만
            }

            original_len = len(places)

            # 이번 검색 결과(places)에서 previous_places에 포함된 주소는 제거
            places = [
                p for p in places
                if normalize(p.get("road_address_name") or "") not in previous_place_keys
            ]

            print(f"중복 필터링: {original_len} → {len(places)}")
            print("previous_place_keys:", previous_place_keys)
            print("places sample:", [normalize(p.get("road_address_name") or "") for p in places[:5]])

            # 중복으로 다 빠진 경우 → 무조건 재탐색
            if not places:
                print("모든 후보가 중복됨 → fallback 재탐색 실행")
                retry_places = await fetch_places_with_fallback(
                    x=x,
                    y=y,
                    keyword=place_category,
                    restrict_text=search_query,
                    dong=(span_m <= 15000)
                )
                # 재탐색 결과에서도 previous_place_keys 에 포함된 건 제거
                places = [
                    p for p in retry_places
                    if normalize(p.get("road_address_name") or "") not in previous_place_keys
                ]

                print(f"재탐색 후 후보 개수: {len(places)}")
                
                if not places:
                    return {"recommendations": []} #재탐색까지 했는데 중복 제거 후 후보가 하나도 없을 때 > 빈 배열 리턴

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
                "road_address_name": place.get("road_address_name"),
                "phone": place.get("phone"),
                "place_url": place.get("place_url"),
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
                "road_address_name": p.get("road_address_name"),
                "phone": p.get("phone"),
                "place_url": p.get("place_url"),
                "description": p["description"],
                "similarity_score": similarity_score
            })
            
        # 6. 유사도 점수 내림차순 정렬 및 상위 5개 반환
        sorted_recommendations = sorted(recommendations, key=lambda x: x["similarity_score"], reverse=True)
        
        # 필요한 필드만 추려서 응답
        trimmed_response = [
            {
                "name": r["place_name"],
                "category": r["category_name"],
                "address_road": r.get("road_address_name"),
                "address_ex": r["address_name"],
                "phone": r.get("phone"),
                "url": r.get("place_url"),
                "score": round(float(r["similarity_score"]), 4)
            }
            for r in sorted_recommendations[:5]
        ]

        return {"recommendations": trimmed_response}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")

@app.post("/model/train")
async def train_model_endpoint(users_data: List[UserBehaviorData]):
    if not users_data:
        raise HTTPException(status_code=400, detail="학습에 필요한 사용자 데이터가 없습니다.")
    train_and_save_model(users_data)
    analyze_clusters() # 학습 후 클러스터 정보 출력
    return {"message": "모델 학습이 완료되었습니다."}

# 모델 파일이 없으면 시작 시 학습 및 저장
if not os.path.exists('user_kmeans_model.pkl'):
    train_and_save_model()

@app.post("/user-cluster")
async def get_user_cluster_endpoint(user_behavior: UserBehaviorData):
    """
    사용자의 행동 데이터를 받아 클러스터를 예측하고 반환합니다.
    """
    # UserBehaviorData 모델의 필드를 CATEGORIES 순서에 맞는 NumPy 배열로 변환
    user_features = np.array([
        user_behavior.family_friendly,
        user_behavior.date_friendly,
        user_behavior.pet_friendly,
        user_behavior.solo_friendly,
        user_behavior.quiet,
        user_behavior.cozy,
        user_behavior.focus,
        user_behavior.noisy,
        user_behavior.lively,
        user_behavior.diverse_menu,
        user_behavior.book_friendly,
        user_behavior.plants,
        user_behavior.trendy,
        user_behavior.photo_friendly,
        user_behavior.good_view,
        user_behavior.spacious,
        user_behavior.aesthetic,
        user_behavior.long_stay,
        user_behavior.good_music,
        user_behavior.exotic
    ])
    
    cluster_id = get_user_cluster(user_behavior)

    if cluster_id is None:
        return {"user_id": user_behavior.user_id, "cluster": "데이터 부족", "message": "사용자 데이터가 부족하여 유형을 분류할 수 없습니다."}
    
    # 클러스터 ID에 따라 사용자 유형 반환
    user_type = {
        0: "소확행",  
        1: "느좋러",  
        2: "입문자",
        3: "인싸"   
    }
    
    return {"user_id": user_behavior.user_id, "cluster": user_type.get(cluster_id, "알 수 없음")}

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
    sample_tile_count: Optional[int] = Query(None, ge=1, le=500, description="타일 중 샘플링 개수 (예: 60개만 선택)")
):
    # 1) 중심 좌표 구하기
    geo = await geocode_address(query)
    if not geo:
        raise HTTPException(status_code=404, detail="지오코딩 결과 없음")
    x, y = geo["x"], geo["y"]

    # # 2) rect 스윕 호출
    # places = await search_places_rect_sweep(
    #     center_x=x,
    #     center_y=y,
    #     keyword=keyword,
    #     category_code=category_code,
    #     total_limit=total_limit,
    #     span_m=span_m,
    #     step_m=step_m,
    #     concurrency=concurrency,
    #     restrict_by_query_text=query,
    #     sample_tile_count=sample_tile_count
    # )
    # 2) fallback 포함된 확장 탐색
    places = await fetch_places_with_fallback(
        x=x,
        y=y,
        keyword=keyword,
        restrict_text=query,
        dong=(span_m <= 15000)  # 또는 동단위 여부 판단 기준
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
#fallback 없는 테스트 api
@app.get("/rect-sweep-test")
async def rect_sweep(
    query: str = Query(..., description="예: 서울 종로구"),
    keyword: Optional[str] = Query(None, description="예: 카페 (없으면 카테고리로만 탐색)"),
    category_code: Optional[str] = Query(None, description="예: 카페 CE7, 편의점 CS2"),
    total_limit: int = Query(200, ge=1, le=1000),
    span_m: int = Query(20000, ge=1000, le=40000, description="전체 박스 크기(미터)"),
    step_m: int = Query(4000, ge=500, le=10000, description="타일 간격(미터)"),
    concurrency: int = Query(8, ge=1, le=32),
    sample_tile_count: Optional[int] = Query(None, ge=1, le=500, description="타일 중 샘플링 개수 (예: 60개만 선택)")
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
        sample_tile_count=sample_tile_count
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

@app.get("/test-geocode")
async def test_geocode(
    query: str = Query(..., description="예: 서울 중랑구 신내동(법정동/행정동/구/시 모두 가능)")
):
    geo = await geocode_address(query)
    if not geo:
        raise HTTPException(status_code=404, detail="지오코딩 결과가 없습니다. 주소를 확인하세요.")
    return {"query": query, "result": geo}


@app.get("/")
def read_root():
    return {"message": "Hello FastAPI!"}

# 날씨 반환 테스트
@app.get("/weather-geocode")
async def weather_geocode(query: str):
    geo = await geocode_address_weather(query)   # 여기 맞춰주기
    if not geo:
        raise HTTPException(status_code=404, detail="지오코딩 결과가 없습니다.")
    return geo



### fallback 기반 장소 검색 함수 추가
#JPN_SYNONYMS = ["일식", "스시", "초밥", "라멘", "돈카츠", "덮밥"]

async def fetch_places_with_fallback(*, x: float, y: float, keyword: str, restrict_text: str, dong: bool) -> List[dict]:
    CATEGORY_KEYWORD_EXPANSION = { #키워드를 더 늘려서 검색
    "술집": ["호프집"],
    "디자인문구": ["소품샵", "문구"],
    # "중식": ["마라탕", "짜장면", "양꼬치", "중국집"],
    # "한식": ["국밥", "비빔밥", "된장찌개", "한정식", "청국장"],
    # "분식": ["떡볶이", "김밥", "순대", "튀김"],
    }
    base_span = 10000 if dong else 20000
    base_step = 1000
    base_sample = 30 if dong else 60
    # expanded_span = 16000 if dong else 28000
    sample_progression = (
        [30, 50, 70, 100, None] if dong else
        [60, 80, 100, 120, None]
    )

    # Step 1: 같은 샘플 개수로 최대 3회 시도
    for _ in range(3):
        places = await search_places_rect_sweep(
        center_x=x,
        center_y=y,
        keyword=keyword,
        category_code=None,
        total_limit=500,
        span_m=base_span,
        step_m=base_step,
        concurrency=8,
        restrict_by_query_text=restrict_text,
        sample_tile_count=base_sample
    )
        if len(places) >= 2:
            return places

    # Step 2: 샘플 개수 늘려가며 시도
    for sample_tile_count in sample_progression[1:]:
        places = await search_places_rect_sweep(
            center_x=x,
            center_y=y,
            keyword=keyword,
            category_code=None,
            total_limit=500,
            span_m=base_span,
            step_m=base_step,
            concurrency=8,
            restrict_by_query_text=restrict_text,
            sample_tile_count=sample_tile_count
        )
        if len(places) >= 1:
            return places

    # # Step n: 범위 확장 > 추후 고려
    # places = await search_places_rect_sweep(
    #     center_x=x,
    #     center_y=y,
    #     keyword=keyword,
    #     category_code=None,
    #     total_limit=700,
    #     span_m=expanded_span,
    #     step_m=base_step,
    #     concurrency=8,
    #     restrict_by_query_text=restrict_text,
    #     sample_tile_count=None
    # )
    # if len(places) >= 5:
    #     return places

    # Step 3: 키워드 확장
    places = []
    seen_ids = set()
    alts = CATEGORY_KEYWORD_EXPANSION.get(keyword, [])

    for alt in alts:
        if alt == keyword:
            continue
        extra = await search_places_rect_sweep(
            center_x=x,
            center_y=y,
            keyword=alt,
            category_code=None,
            total_limit=500,
            span_m=base_span,
            step_m=base_step,
            concurrency=8,
            restrict_by_query_text=restrict_text,
            sample_tile_count=None
        )
        for p in extra:
            pid = p.get("id")
            if pid and pid not in seen_ids:
                places.append(p)
                seen_ids.add(pid)

    # 최종 조건 확인
    if len(places) >= 2:
        return places

    return places

WEATHER_KEYWORDS = {
    "Clear": ["냉면", "삼계탕"],
    "Clouds": ["칼국수", "샤브샤브"],
    "Rain": ["파전", "짬뽕", "감자탕"]
}