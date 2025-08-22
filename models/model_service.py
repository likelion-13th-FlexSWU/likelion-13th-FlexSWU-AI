import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import os
from typing import List, Dict, Any
from models.models import UserBehaviorData

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 20가지 카테고리 목록 (순서 중요)
CATEGORIES = [
    "가족과 가기 좋아요", "데이트하기 좋아요", "반려동물과 함께", "혼밥 하기 편해요", "조용해요",
    "아늑해요", "집중하기 좋아요", "시끌벅적해요", "활기찬 공간이에요", "메뉴가 다양해요",
    "책 읽기 좋아요", "식물이 많아요", "트렌디해요", "사진 찍기 좋아요", "뷰가 좋아요",
    "매장이 넓어요", "인테리어가 감성적이에요", "오래 머물기 좋아요", "음악 선정이 좋아요", "해외같아요"
]

# 클러스터 분석 결과를 바탕으로 클러스터 ID와 사용자 유형 맵핑
user_type_mapping = {
    0: "소확행",
    1: "느좋러",
    2: "입문자",
    3: "인싸"
}

def train_and_save_model(data: List[UserBehaviorData]):
    """
    Spring에서 받은 전체 데이터를 사용하여 모델을 학습하고 저장
    """
    # Spring에서 받은 데이터를 NumPy 배열로 변환
    user_data_np = np.zeros((len(data), len(CATEGORIES)), dtype=int)
    for i, user in enumerate(data):
        user_data_np[i] = np.array([
            user.family_friendly, user.date_friendly, user.pet_friendly, user.solo_friendly,
            user.quiet, user.cozy, user.focus, user.noisy, user.lively, user.diverse_menu,
            user.book_friendly, user.plants, user.trendy, user.photo_friendly, user.good_view,
            user.spacious, user.aesthetic, user.long_stay, user.good_music, user.exotic
        ])
    
    # 데이터 스케일링 및 모델 학습 로직
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_data_np)
    kmeans_model = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=10, batch_size=256)
    kmeans_model.fit(X_scaled)
    
    with open('user_kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    with open('user_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("모델 학습 및 저장 완료: user_scaler.pkl, user_kmeans_model.pkl")

def get_user_cluster(user_behavior: UserBehaviorData, min_surveys: int = 10):
    try:
        # 전달받은 survey_count 값을 직접 확인
        if user_behavior.survey_count < min_surveys:
            return None # 설문 횟수가 10번 미만이면 None 반환
        
        # survey_count를 제외한 나머지 카테고리 값들로 Numpy 배열을 만듦
        user_data = np.array([
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

        total_interactions = np.sum(user_data)
        if total_interactions == 0: # 만약 모든 카운트가 0이면 분석 불가
            return None
        
        with open('user_kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('user_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        user_data_reshaped = user_data.reshape(1, -1)
        user_data_scaled = scaler.transform(user_data_reshaped)
        cluster_id = kmeans_model.predict(user_data_scaled)
        return int(cluster_id[0])
    except FileNotFoundError:
        print("경고: 모델 파일이 존재하지 않습니다. 먼저 /model/train 엔드포인트를 호출하여 학습시키세요.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

app = FastAPI()

# 새로운 학습 엔드포인트: Spring에서 전체 데이터를 받아 모델을 학습
@app.post("/model/train")
async def train_model_endpoint(users_data: List[UserBehaviorData]):
    if not users_data:
        raise HTTPException(status_code=400, detail="학습에 필요한 사용자 데이터가 없습니다.")
    train_and_save_model(users_data)
    return {"message": "모델 학습이 완료되었습니다."}

# 예측 엔드포인트: Spring에서 단일 사용자의 데이터를 받아 유형을 예측
@app.post("/user-cluster")
async def get_user_cluster_endpoint(user_behavior: UserBehaviorData):
    user_features = np.array([
        user_behavior.family_friendly, user_behavior.date_friendly, user_behavior.pet_friendly,
        user_behavior.solo_friendly, user_behavior.quiet, user_behavior.cozy, user_behavior.focus,
        user_behavior.noisy, user_behavior.lively, user_behavior.diverse_menu, user_behavior.book_friendly,
        user_behavior.plants, user_behavior.trendy, user_behavior.photo_friendly, user_behavior.good_view,
        user_behavior.spacious, user_behavior.aesthetic, user_behavior.long_stay, user_behavior.good_music,
        user_behavior.exotic
    ])
    
    cluster_id = get_user_cluster(user_features)
    
    if cluster_id is None:
        raise HTTPException(status_code=400, detail="사용자 데이터가 부족하여 유형을 분류할 수 없습니다.")
    
    user_type = user_type_mapping.get(cluster_id, "알 수 없음")
    
    return {"user_id": user_behavior.user_id, "cluster": user_type}