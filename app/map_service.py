# import os
# import httpx
# from dotenv import load_dotenv
# from typing import List

# load_dotenv()
# KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

# async def search_places_kakao(keywords: List[str]):
#     """
#     카카오 지도 API를 사용하여 키워드 기반으로 장소를 비동기적으로 검색합니다.
#     """
#     url = "https://dapi.kakao.com/v2/local/search/keyword.json"
#     headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    
#     places = []
#     async with httpx.AsyncClient() as client:
#         for keyword in keywords:
#             params = {"query": keyword, "size": 30}
#             response = await client.get(url, headers=headers, params=params)
#             if response.status_code == 200:
#                 places.extend(response.json().get('documents', []))
#     return places