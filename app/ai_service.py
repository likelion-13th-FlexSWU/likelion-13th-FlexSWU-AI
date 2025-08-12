import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Union

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_gpt_embedding(text: Union[str, List[str]]):
    """
    GPT를 사용하여 텍스트나 키워드 목록의 임베딩 벡터를 비동기적으로 반환합니다.
    """
    if isinstance(text, list):
        text = " ".join(text)
    
    response = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

async def generate_place_description(place_info: dict):
    """
    장소 정보를 바탕으로 GPT를 사용하여 설명 문장을 비동기적으로 생성합니다.
    카테고리를 프롬프트에 포함시켜 더 정확한 설명을 유도합니다.
    """
    category_name = place_info.get('category_name', '장소')
    
    prompt = (
        f"장소 이름: {place_info.get('place_name')}, "
        f"카테고리: {category_name}. "
        f"이 장소의 분위기나 특징을 감성적인 문장으로 한 줄 설명해 줘. "
        f"예시: '한적한 분위기의 한식 전문점으로 혼자 식사하기 좋아요.'"
    )
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()