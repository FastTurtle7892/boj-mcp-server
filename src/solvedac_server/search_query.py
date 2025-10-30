from fastmcp.prompts.prompt import  PromptMessage, TextContent
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Union
import httpx
import re
from difflib import get_close_matches

class TagDisplayName(BaseModel):
    model_config = ConfigDict(extra="ignore")
    language: Optional[str] = None
    name: str

class TagInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    key: str
    displayNames: Optional[List[TagDisplayName]] = None
    # 리스트가 ["dp", ...] 이거나 [{"alias":"동적계획법"}, ...] 둘 다 올 수 있음
    aliases: Optional[List[Union[str, dict]]] = None

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, v):
        if v is None:
            return []
        out = []
        for it in v:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict):
                a = it.get("alias")
                if isinstance(a, str) and a.strip():
                    out.append(a)
        return out

    @field_validator("displayNames", mode="before")
    @classmethod
    def _coerce_display_names(cls, v):
        if v is None:
            return None
        out = []
        for it in v:
            if isinstance(it, dict) and isinstance(it.get("name"), str):
                out.append(it)
            elif isinstance(it, str):
                out.append({"language": None, "name": it})
        return out

class TagCatalog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tags: List[TagInfo]
    token_to_key: Dict[str, str]  # 다양한 표기 → 공식 key
    
class Problem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    problemId: int
    titleKo: Optional[str] = None
    level: int
    isSolvable: bool

class ProblemSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    count: int
    items: List[Problem]

_WORD = re.compile(r"[A-Za-z0-9가-힣_+\-]+")
_TAG_PATTERN = re.compile(r"(?:#|tag:)\s*([A-Za-z0-9_+\-가-힣 ]+)", re.IGNORECASE)

def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())

def infer_bare_tag_keys(
    query: str, token_to_key: dict[str, str]
) -> list[str]:
    # '#', 'tag:'가 있으면 이미 명시 태그가 있으니 건너뜀
    s = (query or "")
    if "#" in s or "tag:" in s.lower():
        return []
    tokens = set(m.group(0) for m in _WORD.finditer(s))
    keys = []
    seen = set()
    for tok in tokens:
        k = token_to_key.get(_norm(tok)) or token_to_key.get(
            _norm(f"#{tok}")
        )
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys

def build_tag_catalog(raw: List[TagInfo]) -> TagCatalog:
    token_to_key: Dict[str, str] = {}
    for t in raw:
        key = t.key
        for tok in {key, f"#{key}"}:
            token_to_key[_norm(tok)] = key
        for dn in (t.displayNames or []):
            token_to_key[_norm(dn.name)] = key
            token_to_key[_norm(f"#{dn.name}")] = key
        for a in (t.aliases or []):
            token_to_key[_norm(a)] = key
            token_to_key[_norm(f"#{a}")] = key
    return TagCatalog(tags=raw, token_to_key=token_to_key)

def _needs_tag_catalog(query: str, enable_bare_infer: bool) -> bool:
    s = (query or "").lower()
    # #dp, tag:dp 같은 "명시 태그"가 있거나, bare-word 승격을 켜 둔 경우에만 필요
    return ("#" in s) or ("tag:" in s) or bool(enable_bare_infer)

def _augment_query_with_user_filters(
    base: str, handle: Optional[str]
) -> str:
    q = base.strip()
    if handle and handle.strip():
        # 세션 없어도 동작: 특정 핸들이 푼/시도한 문제 제외
        q += f" -@{handle} -t@{handle}"
    else:
        # 세션이 유효할 때만 동작: 현재 로그인 사용자 기준
        q += " -@$me -t@$me"
    return q


def search_workflow_prompt_core(
    natural_request: str = Field(
        ..., description="예: '실버~골드 DP 5문제, 비슷한 태그 제외'"
    ),
) -> list[PromptMessage]:
    sys = PromptMessage(
        role="assistant",
        content=TextContent(
            type="text",
            text=(
                "You are a Solved.ac search assistant.\n"
                "1) Convert the user's request into a precise Solved.ac query string "
                "(e.g., `tier:g5..p5 #dp -#greedy`).\n"
                "2) Do NOT browse the web.\n"
                "3) Call the MCP TOOL `solvedac_search_problems` with {query, limit, handle}.\n"
                "4) Rank top 5 by suitability and show: problemId, titleKo, level."
            ),
        ),
    )
    usr = PromptMessage(
        role="user",
        content=TextContent(
            type="text",
            text=(
                f"요청(자연어): {natural_request}\n"
                f"규칙: 쿼리를 먼저 제시하고, 이어서 리소스를 호출해 결과를 평가하세요."
            ),
        ),
    )
    return [sys, usr]

def normalize_query_tags(
    query: str,
    token_to_key: Dict[str, str],
    *,
    strict: bool = True,
) -> tuple[str, list[str], dict[str, list[str]]]:
    """
    '#dp', 'tag:dp', '#다이나믹 프로그래밍' 같은 토큰을 공식 '#{key}'로 치환합니다.
    strict=True면 모르는 태그를 unknown에 담아 호출부에서 에러 처리할 수 있게 합니다.
    """
    def _n(s: str) -> str:
        return " ".join(s.strip().lower().split())

    q = query
    unknown: list[str] = []
    suggestions: dict[str, list[str]] = {}

    mentioned = set(
        m.group(1).strip() for m in _TAG_PATTERN.finditer(query)
    )
    for tok in mentioned:
        key = token_to_key.get(_n(tok)) or token_to_key.get(
            _n(f"#{tok}")
        )
        if not key:
            unknown.append(tok)
            cand_tokens = list(token_to_key.keys())
            close = get_close_matches(
                _n(tok), cand_tokens, n=5, cutoff=0.75
            )
            suggestions[tok] = sorted(
                {token_to_key[c] for c in close if c in token_to_key}
            )
            continue
        q = re.sub(
            rf"(#|tag:)\s*{re.escape(tok)}\b",
            f"#{key}",
            q,
            flags=re.IGNORECASE,
        )

    return q, unknown, suggestions

# ===== 태그 목록/상세 코어 =====
async def fetch_tag_list_core(client: httpx.AsyncClient) -> List[TagInfo]:
    # 환경에 따라 경로가 다를 수 있어 순차 시도
    for url, params in [
        ("/tag/list", None),
        ("/api/v3/tag/list", None),
    ]:
        resp = await client.get(url, params=params)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        data = resp.json()
        items = (
            data.get("items", data) if isinstance(data, dict) else data
        )
        return [TagInfo.model_validate(t) for t in items]
    raise RuntimeError("태그 목록 엔드포인트를 찾을 수 없습니다.")

async def fetch_tag_by_key_core(
    client: httpx.AsyncClient, key: str
) -> TagInfo:
    # show / path / api/v3 변형 경로 순차 시도
    for url, params in [
        ("/tag/show", {"key": key}),
        (f"/tag/{key}", None),
        ("/api/v3/tag/show", {"key": key}),
    ]:
        resp = await client.get(url, params=params)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        return TagInfo.model_validate(resp.json())
    raise ValueError(f"태그 키 '{key}'를 찾을 수 없습니다.")


async def search_problems_core(
        state: Dict[str, Any],
        query: str,
        page: int = 1,
        *,
        sort: Optional[str] = None,        # 예: "random", "id", "level", ...
        direction: Optional[str] = None    # 예: "asc" | "desc" (필요 시)
) -> ProblemSearchResponse:
    """
    solved.ac 문제 검색 코어 함수.
    - 기존: query, page만 지원
    - 변경: 선택적으로 sort/direction을 추가 지원 (sort="random" 사용 가능)

    Parameters
    ----------
    query : str
        solved.ac 고급검색 쿼리 문자열 (예: "tier:g5..p5 tag:dfs -@$me -t@$me")
    page : int, default 1
        페이지(1부터)
    sort : Optional[str], default None
        정렬 키. 랜덤 정렬을 원할 때 "random"을 지정.
    direction : Optional[str], default None
        정렬 방향("asc" | "desc"). 랜덤 정렬에서는 보통 불필요.
    """
    client: httpx.AsyncClient = state.get("http_client")
    if not client:
        raise RuntimeError("HTTP Client is not available.")

    # 쿼리 파라미터 구성
    params: Dict[str, Any] = {"query": query, "page": page}
    if sort:
        params["sort"] = sort
    if direction:
        # 가벼운 방어적 정규화 (옵션)
        dnorm = direction.lower()
        if dnorm in ("asc", "desc"):
            params["direction"] = dnorm

    try:
        resp = await client.get("/search/problem", params=params)
        resp.raise_for_status()
        return ProblemSearchResponse.model_validate(resp.json())

    except httpx.HTTPStatusError as e:
        s = e.response.status_code
        if s == 429:
            raise RuntimeError(
                "요청이 많습니다(429). 잠시 후 다시 시도하세요."
            ) from e
        if 500 <= s < 600:
            raise RuntimeError(
                "solved.ac 서버 오류가 발생했습니다. 잠시 후 다시 시도하세요."
            ) from e
        raise

    except httpx.RequestError as e:
        raise RuntimeError(
            f"네트워크 오류로 문제 검색에 실패했습니다: {e}"
        ) from e
                
async def _ensure_tag_catalog(
    state: Dict[str, Any],
) -> TagCatalog:
    """
    태그 카탈로그를 state 메모리에 로드하고 반환.
    - 이미 있으면 그대로 반환
    - 없으면 API에서 가져와 build_tag_catalog 후 state에 저장
    """
    cat: Optional[TagCatalog] = state.get("tag_catalog")
    if cat is not None:
        return cat
    client: httpx.AsyncClient = state.get("http_client")
    if not client:
        raise RuntimeError("HTTP Client is not available.")
    raw = await fetch_tag_list_core(client)
    cat = build_tag_catalog(raw)
    state["tag_catalog"] = cat
    return cat