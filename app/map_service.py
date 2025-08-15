import os
import httpx
import math
import asyncio
import re
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import pathlib

_GU_PAT = re.compile(r"([가-힣A-Za-z]+구)")

def _extract_gu_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    # 공백 기준 우선
    for tok in str(text).split():
        if tok.endswith("구"):
            return tok
    # 공백 없앨 때도 탐색
    m = _GU_PAT.search(str(text).replace(" ", ""))
    return m.group(1) if m else None

def _place_in_gu(place: Dict[str, Any], target_gu: str) -> bool:
    addr = f"{place.get('road_address_name','')} {place.get('address_name','')}"
    return target_gu in addr

# .env 명시적 로드
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

KAKAO_API_KEY = (os.getenv("KAKAO_API_KEY") or "").strip()
HEADERS = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"} if KAKAO_API_KEY else None

KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"



TIMEOUT = httpx.Timeout(10.0)

def _normalize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        try:
            out.append({
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
            })
        except Exception:
            pass
    return out

# ===== 1) 반경+페이지네이션으로 원하는 개수(total_limit)까지 모으기 =====
async def search_places_paged_around(
    x: float,
    y: float,
    keyword: str,
    total_limit: int = 60,
    radius: int = 2000,
    sort: str = "distance",      # distance 권장
    per_page: int = 15,          # 카카오 하드 리밋
    max_page: int = 45,          # 카카오 최대 페이지
    expand_radius: bool = True,  # 부족하면 반경 키우기
    max_radius: int = 20000,
) -> List[Dict[str, Any]]:
    """
    좌표/반경 기반으로 페이지를 돌며 total_limit 만큼 수집.
    필요시 반경을 넓혀가며 추가 수집.
    """
    if not HEADERS:
        print("[KAKAO] Missing KAKAO_API_KEY")
        return []

    results: List[Dict[str, Any]] = []
    seen = set()
    cur_radius = min(max(radius, 1), max_radius)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        while len(results) < total_limit:
            got_any = False
            for page in range(1, max_page + 1):
                params = {
                    "query": keyword,
                    "x": x, "y": y,
                    "radius": cur_radius,
                    "sort": sort,
                    "size": per_page,
                    "page": page,
                }
                r = await client.get(KAKAO_KEYWORD_URL, headers=HEADERS, params=params)
                r.raise_for_status()
                data = r.json()
                docs = data.get("documents", [])
                meta = data.get("meta", {}) or {}

                if docs:
                    got_any = True
                    for item in _normalize_docs(docs):
                        pid = item.get("id")
                        if pid and pid not in seen:
                            seen.add(pid)
                            results.append(item)
                            if len(results) >= total_limit:
                                break

                # 마지막 페이지면 중단
                if meta.get("is_end", False) or len(docs) < per_page or len(results) >= total_limit:
                    break

            # 충분하면 종료
            if len(results) >= total_limit:
                break
            # 이번 라운드에서 거의 못 얻었고 반경 확장 허용 시
            if expand_radius and cur_radius < max_radius:
                # radius 점진 확대 (1.5배, 최대 20km)
                new_radius = int(cur_radius * 1.5)
                cur_radius = min(max(new_radius, cur_radius + 500), max_radius)
                continue

            # 더 이상 얻을 방법 없음
            break

    return results[:total_limit]

# ===== 2) rect 타일 스윕(사각형으로 지도를 쪼개서 많이 끌어오기) =====
KAKAO_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"
PER_PAGE = 15
MAX_PAGE = 45

def _meters_to_deg(lat: float, meters: float) -> Tuple[float, float]:
    dlat = meters / 111_000.0
    dlon = meters / (111_320.0 * math.cos(math.radians(lat)))
    return dlat, dlon

def _make_rect_tiles(cx: float, cy: float, lat_for_deg: float, span_m: int, step_m: int) -> List[Tuple[float, float, float, float]]:
    dlat_span, dlon_span = _meters_to_deg(lat_for_deg, span_m / 2)
    dlat_step, dlon_step = _meters_to_deg(lat_for_deg, step_m)
    min_y, max_y = cy - dlat_span, cy + dlat_span
    min_x, max_x = cx - dlon_span, cx + dlon_span

    rects = []
    y = min_y
    while y < max_y:
        y2 = min(y + dlat_step, max_y)
        x = min_x
        while x < max_x:
            x2 = min(x + dlon_step, max_x)
            rects.append((x, y, x2, y2))
            x += dlon_step
        y += dlat_step

    # 중앙에 가까운 타일부터
    rects.sort(key=lambda r: abs((r[0]+r[2])/2 - cx) + abs((r[1]+r[3])/2 - cy))
    return rects

async def _paged_fetch(client: httpx.AsyncClient, url: str, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    acc: List[Dict[str, Any]] = []
    for page in range(1, MAX_PAGE + 1):
        params = dict(base_params)
        params["size"] = PER_PAGE
        params["page"] = page
        r = await client.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        data = r.json()
        docs = data.get("documents", [])
        acc.extend(docs)
        meta = data.get("meta", {}) or {}
        if meta.get("is_end", False) or len(docs) < PER_PAGE:
            break
    return _normalize_docs(acc)

async def _fetch_tile(client: httpx.AsyncClient, rect: Tuple[float, float, float, float], keyword: Optional[str], category_code: Optional[str]) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = rect
    rect_str = f"{x1},{y1},{x2},{y2}"
    tasks = []
    if keyword:
        tasks.append(_paged_fetch(client, KAKAO_KEYWORD_URL, {"query": keyword, "rect": rect_str}))
    if category_code:
        tasks.append(_paged_fetch(client, KAKAO_CATEGORY_URL, {"category_group_code": category_code, "rect": rect_str}))
    if not tasks:
        return []
    parts = await asyncio.gather(*tasks)
    merged: List[Dict[str, Any]] = []
    seen = set()
    for part in parts:
        for it in part:
            pid = it.get("id")
            if pid and pid not in seen:
                seen.add(pid)
                merged.append(it)
    return merged

async def search_places_rect_sweep(
    center_x: float,
    center_y: float,
    keyword: Optional[str],
    total_limit: int = 150,
    span_m: int = 20000,        # 전체 박스 크기(예: 20km)
    step_m: int = 4000,         # 타일 간격
    category_code: Optional[str] = None,  # 예: 카페 "CE7"
    concurrency: int = 8,
    restrict_by_query_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    큰 반경에서도 적게 나올 때, rect 타일로 넓은 구역을 병렬로 긁어서 리콜을 크게 확보.
    """
    if not HEADERS:
        print("[KAKAO] Missing KAKAO_API_KEY")
        return []

    rects = _make_rect_tiles(center_x, center_y, center_y, span_m, step_m)
    results: List[Dict[str, Any]] = []
    seen = set()
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async def worker(rect):
            nonlocal results
            if len(results) >= total_limit:
                return
            async with sem:
                items = await _fetch_tile(client, rect, keyword, category_code)
                for it in items:
                    pid = it.get("id")
                    if pid and pid not in seen:
                        seen.add(pid)
                        results.append(it)
                        if len(results) >= total_limit:
                            break

        for rect in rects:
            if len(results) >= total_limit:
                break
            await worker(rect)
    if restrict_by_query_text:
        gu = _extract_gu_from_text(restrict_by_query_text)
        if gu:
            results = [p for p in results if _place_in_gu(p, gu)]

    return results[:total_limit]

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
