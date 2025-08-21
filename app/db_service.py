import os
import numpy as np
from dotenv import load_dotenv
import mysql.connector
from typing import List, Dict, Any

load_dotenv()

# .env 파일에서 DB 접속 정보 불러오기
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DATABASE = os.getenv("DB_DATABASE")

def get_all_user_behavior_data() -> np.ndarray | None:
    """
    모든 사용자의 행동 데이터를 DB에서 가져와 NumPy 배열로 반환
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )
        cursor = conn.cursor(dictionary=False)
        
        # 쿼리문의 컬럼 순서를 models.py의 UserBehaviorData 필드 순서와 정확히 일치시켜야 함
        query = """
        SELECT family_friendly, date_friendly, pet_friendly, solo_friendly, quiet, cozy, focus,
               noisy, lively, diverse_menu, book_friendly, plants, trendy, photo_friendly,
               good_view, spacious, aesthetic, long_stay, good_music, exotic
        FROM user_behaviors
        """
        cursor.execute(query)
        data = cursor.fetchall()
        
        if not data:
            print("경고: 데이터베이스에 사용자 행동 데이터가 없습니다.")
            return None

        # 가져온 데이터를 NumPy 배열로 변환
        data_np = np.array(data, dtype=int)
        return data_np
        
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()

async def get_user_reviews(user_id: str) -> List[Dict[str, Any]]:
    """
    사용자 ID를 기반으로 해당 사용자가 남긴 장소 리뷰 목록을 데이터베이스에서 가져옴
    """
    conn = None
    reviews = []
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )
        cursor = conn.cursor(dictionary=True)
        
        # SQL 쿼리 작성 (테이블명과 컬럼명은 실제 DB에 맞게 수정 필요)
        query = "SELECT place_id, review_text FROM user_reviews WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        
        reviews = cursor.fetchall()

    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return reviews

async def get_all_user_behavior_data() -> List[Dict[str, Any]]:
    """
    모든 사용자의 행동 데이터를 DB에서 가져와 모델 학습에 사용
    """
    conn = None
    data = []
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )
        cursor = conn.cursor(dictionary=True)
        
        # user_behaviors 테이블은 사용자의 행동 데이터를 저장하는 테이블
        # 컬럼명은 app/models.py의 UserBehaviorData 필드와 동일해야 함
        query = "SELECT user_id, family_friendly, date_friendly, ... FROM user_behaviors"
        cursor.execute(query)
        data = cursor.fetchall()
    finally:
        if conn and conn.is_connected():
            conn.close()
    return data