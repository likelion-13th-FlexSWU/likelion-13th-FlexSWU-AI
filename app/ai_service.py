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

async def generate_place_description(place_info: dict, user_keywords: List[str]):
    """
    장소 정보를 바탕으로 GPT를 사용하여 설명 문장을 비동기적으로 생성합니다.
    카테고리를 프롬프트에 포함시켜 더 정확한 설명을 유도합니다.
    """
    category_name = place_info.get('category_name', '장소')

    # 사용자 키워드를 프롬프트에 직접 포함
    keywords_str = ', '.join(user_keywords)
    
    prompt = (
        f"장소 이름: {place_info.get('place_name')}, "
        f"카테고리: {category_name}. "
        f"다음 키워드들을 참고하여 이 장소의 특징을 한 문장으로 요약해 줘: {keywords_str} "
    )
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()