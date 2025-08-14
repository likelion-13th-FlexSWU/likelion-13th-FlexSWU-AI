# app/map_service.py
import os
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import pathlib

# .env 명시적 로드
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

# 따옴표/공백 제거
KAKAO_API_KEY = (os.getenv("KAKAO_API_KEY") or "").strip()
HEADERS = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"} if KAKAO_API_KEY else None

KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"

print("[KAKAO] KEY loaded? ", bool(KAKAO_API_KEY), "len=", len(KAKAO_API_KEY) if KAKAO_API_KEY else 0)
print("[KAKAO] .env path used =", (ROOT / ".env"))


async def geocode_address(query: str) -> Optional[Dict[str, Any]]:
    """
    '서울 중랑구' → {'x': 경도, 'y': 위도, 'address': {...}} (실패 시 None)
    """
    if not HEADERS:
        print("[KAKAO] Missing KAKAO_API_KEY")
        return None

    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(KAKAO_ADDR_URL, headers=HEADERS, params={"query": query})

        docs = r.json().get("documents", [])
        if not docs:
            return None

        doc = docs[0]  # 가장 적합한 결과 하나 사용
        return {
            "x": float(doc["x"]),  # 경도
            "y": float(doc["y"]),  # 위도
            "address": doc.get("address") or doc.get("road_address"),
        }
