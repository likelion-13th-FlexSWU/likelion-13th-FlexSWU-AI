import asyncio
from app.ai_service import get_gpt_embedding, generate_place_description

# 가상 데이터 생성 (실제 API 응답 형식을 모방)
mock_place_info = {
    "place_name": "서울시립미술관",
    "category_name": "문화, 예술 > 미술관",
    "address_name": "서울특별시 중구 덕수궁길 61"
}

async def test_ai_services():
    print("--- 1. get_gpt_embedding 테스트 ---")
    user_keywords = ["조용한", "감성적인", "힐링"]
    embedding = await get_gpt_embedding(user_keywords)
    print(f"사용자 키워드 임베딩: {embedding[:5]}...") # 벡터의 일부만 출력
    print(f"임베딩 벡터 길이: {len(embedding)}")
    print("-" * 30)

    print("--- 2. generate_place_description 테스트 ---")
    description = await generate_place_description(mock_place_info)
    print(f"생성된 장소 설명: {description}")
    print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_ai_services())