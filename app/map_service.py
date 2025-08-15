import os
import httpx
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pathlib

# .env 명시적 로드
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

KAKAO_API_KEY = (os.getenv("KAKAO_API_KEY") or "").strip()
HEADERS = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"} if KAKAO_API_KEY else None

KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

TIMEOUT = httpx.Timeout(10.0)

async def geocode_address(query: str) -> Optional[Dict[str, Any]]:
    if not HEADERS:
        print("[KAKAO] Missing KAKAO_API_KEY")
        return None
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(KAKAO_ADDR_URL, headers=HEADERS, params={"query": query})
        r.raise_for_status()
        docs = r.json().get("documents", [])
        if not docs:
            return None
        doc = docs[0]
        addr_obj = doc.get("road_address") or doc.get("address") or {}
        addr_text = addr_obj.get("address_name") or query
        print( float(doc["x"]))
        print( float(doc["y"]))
        return {"x": float(doc["x"]), "y": float(doc["y"]), "address": addr_text}

async def search_places_around(
    x: float, y: float, keyword: str, radius: int, size: int, page: int, sort: str
) -> List[Dict[str, Any]]:
    if not HEADERS:
        print("[KAKAO] Missing KAKAO_API_KEY")
        return []
    params = {
        "query": keyword, "x": x, "y": y,
        "radius": radius, "size": size, "page": page, "sort": sort,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(KAKAO_KEYWORD_URL, headers=HEADERS, params=params)
        r.raise_for_status()
        docs = r.json().get("documents", [])

    return [
        {
            "id": d.get("id"),
            "place_name": d.get("place_name"),
            "category_name": d.get("category_name"),
            "category_group_code": d.get("category_group_code"),
            "category_group_name": d.get("category_group_name"),
            "phone": d.get("phone"),
            "address_name": d.get("address_name"),
            "road_address_name": d.get("road_address_name"),
            "x": float(d["x"]) if d.get("x") else None,
            "y": float(d["y"]) if d.get("y") else None,
            "place_url": d.get("place_url"),
            "distance": int(d["distance"]) if d.get("distance") else None,
        }
        for d in docs
    ]
