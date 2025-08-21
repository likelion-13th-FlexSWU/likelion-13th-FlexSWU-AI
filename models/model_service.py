import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

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

def get_user_cluster(user_data: np.ndarray, min_interactions: int = 10):
    """
    주어진 사용자 데이터를 바탕으로 클러스터를 예측하고,
    클러스터 ID에 해당하는 사용자 유형을 반환하는 함수.

    Args:
        user_data (np.ndarray): 20개 카테고리별 선택 횟수를 담은 배열.
        min_interactions (int): 유형 분류를 위한 최소 누적 선택 횟수.
        
    Returns:
        int or None: 예측된 클러스터 ID, 데이터 부족 시 None.
    """
    try:
        # 모델 및 스케일러 로드
        with open('user_kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('user_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # 데이터 형태 변환 및 스케일링
        user_data_reshaped = user_data.reshape(1, -1)
        user_data_scaled = scaler.transform(user_data_reshaped)

        # 클러스터 예측
        cluster_id = kmeans_model.predict(user_data_scaled)
        
        return int(cluster_id[0])

    except FileNotFoundError:
        print("경고: 모델 파일이 존재하지 않습니다. train_and_save_model() 함수를 먼저 실행하세요.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def train_and_save_model():
    """
    더미 데이터를 생성하여 K-Means 모델을 학습하고 저장하는 함수.
    """
    try:
        # 1000개의 더미 데이터 생성
        np.random.seed(42)
        
        # '인싸' 그룹 (시끌벅적, 데이트, 활기찬)
        group1 = np.random.poisson(lam=[5, 10, 0, 0, 0, 0, 0, 30, 25, 5, 0, 0, 10, 15, 10, 15, 10, 5, 15, 15], size=(250, 20))
        # '소확행' 그룹 (혼밥, 조용, 아늑)
        group2 = np.random.poisson(lam=[0, 0, 0, 25, 20, 25, 20, 0, 0, 5, 20, 0, 0, 0, 0, 0, 0, 25, 0, 0], size=(250, 20))
        # '입문자' 그룹 (메뉴 다양, 넓은 매장)
        group3 = np.random.poisson(lam=[5, 5, 5, 5, 5, 5, 5, 5, 5, 25, 5, 5, 5, 5, 5, 25, 5, 5, 5, 5], size=(250, 20))
        # '느좋러' 그룹 (식물, 사진, 뷰, 인테리어)
        group4 = np.random.poisson(lam=[0, 10, 0, 0, 0, 0, 0, 10, 5, 0, 0, 25, 20, 30, 25, 15, 30, 0, 25, 25], size=(250, 20))

        X = np.concatenate([group1, group2, group3, group4])

        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means 모델 학습
        kmeans_model = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=10, batch_size=256)
        kmeans_model.fit(X_scaled)
        
        # 모델과 스케일러 저장
        with open('user_kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans_model, f)
        with open('user_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        print("모델 학습 및 저장 완료: user_scaler.pkl, user_kmeans_model.pkl")

    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")

def analyze_clusters():
    """
    저장된 모델을 로드하여 각 클러스터의 특징을 분석하는 함수.
    """
    try:
        with open('user_kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('user_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # 클러스터 중심점 (역변환하여 원래 스케일로 복원)
        cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
        print("--- 클러스터별 특징 (평균 선택 횟수) ---")
        for i, center in enumerate(cluster_centers):
            print(f"\n클러스터 {i} ({user_type_mapping.get(i, '알 수 없음')}):")
            top_features_indices = np.argsort(center)[::-1][:5]
            for idx in top_features_indices:
                print(f"  - {CATEGORIES[idx]}: {center[idx]:.2f}회")
        print("-" * 30)

    except FileNotFoundError:
        print("경고: 모델 파일이 존재하지 않아 클러스터 분석을 할 수 없습니다. train_and_save_model() 함수를 먼저 실행하세요.")
    except Exception as e:
        print(f"클러스터 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    train_and_save_model()
    
    # 더미 데이터
    sample_user_data = np.zeros(len(CATEGORIES), dtype=int)
    sample_user_data[CATEGORIES.index("가족과 가기 좋아요")] = 30
    sample_user_data[CATEGORIES.index("사진 찍기 좋아요")] = 30
    sample_user_data[CATEGORIES.index("음악 선정이 좋아요")] = 25
    sample_user_data[CATEGORIES.index("인테리어가 감성적이에요")] = 25
    sample_user_data[CATEGORIES.index("조용해요")] = 20
    sample_user_data[CATEGORIES.index("트렌디해요")] = 20

    print("--- 디버깅: 입력될 사용자 데이터 ---")
    for i, count in enumerate(sample_user_data):
        if count > 0:
            print(f"  - {CATEGORIES[i]}: {count}회")
    print("-" * 20)
    
    cluster_id = get_user_cluster(sample_user_data)

    if cluster_id is not None:
        user_type = user_type_mapping.get(cluster_id, "알 수 없음")
        print(f"예시 사용자의 클러스터: {user_type}")
    else:
        print("예시 사용자의 클러스터: 데이터 부족")
    
    analyze_clusters()