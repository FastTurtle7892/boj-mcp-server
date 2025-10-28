# solvedac_server.py (핵심 부분만)
from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message, PromptMessage, TextContent
import asyncio
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator
from typing import List, Optional, Literal, Dict, Any, Union
import httpx
import re
from difflib import get_close_matches
from contextlib import asynccontextmanager
from smithery.decorators import smithery

TIER_NAMES = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ruby"]
TIER_SYMBOLS = ["V", "IV", "III", "II", "I"]

def get_tier_name_from_level(level: int) -> str:
    """solved.ac의 숫자 레벨을 문자열 티어 이름(예: Platinum V)으로 변환합니다."""
    if level == 0:
        return "Unrated"
    
    if 1 <= level <= 30:
        # (level - 1) // 5 = 0(Bronze) ~ 5(Ruby)
        tier_index = (level - 1) // 5
        # (level - 1) % 5 = 0(V) ~ 4(I)
        sub_tier_index = (level - 1) % 5
        
        main_tier = TIER_NAMES[tier_index]
        symbol = TIER_SYMBOLS[sub_tier_index]
        
        return f"{main_tier} {symbol}"
    
    if level == 31:
        return "Master"

    return "Unknown"


class UserShowResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
 
    handle: str
    tier: int   
    rating: int
    solvedCount: int

    @computed_field
    @property
    def tier_name(self) -> str:
        """
        API에서 받은 숫자 티어(self.tier)를
        헬퍼 함수를 이용해 문자열 이름(예: Platinum V)으로 변환합니다.
        """
        return get_tier_name_from_level(self.tier)

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


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())

SOLVEDAC_API_BASE_URL = "https://solved.ac/api/v3"
state = {"http_client": None}

@asynccontextmanager
async def lifespan(app: FastMCP):
    state["http_client"] = httpx.AsyncClient(
        base_url=SOLVEDAC_API_BASE_URL,
        headers={"X-Solvedac-Language": "ko"},
        timeout=10.0,
        follow_redirects=True,
    )
    try:
        yield
    finally:
        if state["http_client"]:
            await state["http_client"].aclose()
        state["http_client"] = None

@smithery.server()
def create_server():
    app = FastMCP(name="SolvedAcAPI", lifespan=lifespan)

    # ---------- 0) 도구 함수: 모델 호출 함수(실행 버튼) ----------
    # 1) 유저 정보 조회 툴
    @app.tool(
        name="solvedac_get_user_info",
        description="solved.ac 사용자의 레이팅/티어/푼 문제 수 조회"
    )
    async def get_user_info_tool(
        handle: str = Field(..., description="사용자 핸들")
    ):
        return await get_user_info_core(handle)

# --------------------- 태그 함수 --------------------------
    def _needs_tag_catalog(query: str, enable_bare_infer: bool) -> bool:
        s = (query or "").lower()
        # #dp, tag:dp 같은 "명시 태그"가 있거나, bare-word 승격을 켜 둔 경우에만 필요
        return ("#" in s) or ("tag:" in s) or bool(enable_bare_infer)

    _WORD = re.compile(r"[A-Za-z0-9가-힣_+\-]+")

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

    def _norm(s: str) -> str:
        return " ".join(s.strip().lower().split())

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

    # ===== 태그 카탈로그 리소스(메모리 보관) =====
    @app.resource(
        "solvedac://tags/catalog",
        description="solved.ac 공식 태그 목록을 로드하여 서버 메모리에 보관합니다.",
    )
    async def solvedac_tag_catalog_resource():
        client: httpx.AsyncClient = state.get("http_client")
        if not client:
            raise RuntimeError("HTTP Client is not available.")
        cat = state.get("tag_catalog")
        if cat is None:
            raw = await fetch_tag_list_core(client)
            cat = build_tag_catalog(raw)
            state["tag_catalog"] = cat
        return {"count": len(cat.tags), "keys": [t.key for t in cat.tags]}

    @app.tool(
        name="solvedac_refresh_tag_catalog",
        description="solved.ac 태그 카탈로그를 강제로 다시 로드합니다.",
    )
    async def refresh_tag_catalog_tool():
        client: httpx.AsyncClient = state.get("http_client")
        raw = await fetch_tag_list_core(client)
        cat = build_tag_catalog(raw)
        state["tag_catalog"] = cat
        return {"refreshed": True, "count": len(cat.tags)}

    # ===== 쿼리 내 태그를 공식 키로 정규화 =====
    _TAG_PATTERN = re.compile(
        r"(?:#|tag:)\s*([A-Za-z0-9_+\-가-힣 ]+)", re.IGNORECASE
    )

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

    # ----------------------------------------------------------------
    DEFAULT_LIMIT = 5

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

    @app.tool(
        name="solvedac_search_problems",
        description=(
            "난이도/태그/키워드 검색 (예: tier:g5..p5 #dp). "
            "공식 태그 정규화(옵션) 후, -@handle/-t@handle + sort=random."
        ),
    )
    async def search_problems_tool(
        query: str = Field(
            ...,
            description="검색 쿼리 (예: 'tier:g5..p5 #dp' 또는 '완전탐색')",
        ),
        limit: int = Field(
            DEFAULT_LIMIT, ge=1, le=20, description="반환 개수"
        ),
        handle: Optional[str] = Field(
            None, description="사용자 solved.ac 핸들"
        ),
        max_attempts: int = Field(
            3, ge=1, le=5, description="랜덤 정렬 재시도(중복 제거)"
        ),
        strict_tags: bool = Field(
            False, description="모르는 태그는 에러(True)/무시(False)"
        ),
        enable_bare_infer: bool = Field(
            False, description="맨바닥 단어(예: '완전탐색')를 태그로 승격"
        ),
    ):
        q0 = (query or "").strip()

        # 0) 태그가 실제로 필요할 때만 카탈로그 사용
        if _needs_tag_catalog(q0, enable_bare_infer):
            cat = await _ensure_tag_catalog()

            # 1) 명시 태그(#dp, tag:dp ...) 정규화
            q_norm, unknown, _ = normalize_query_tags(
                q0, cat.token_to_key, strict=strict_tags
            )
            if strict_tags and unknown:
                raise ValueError(f"알 수 없는 태그: {', '.join(unknown)}")

            # 2) 요청 시에만 bare-word 승격 (예: '완전탐색' → '#bruteforce')
            if enable_bare_infer:
                inferred = infer_bare_tag_keys(q_norm, cat.token_to_key)
                if inferred:
                    q_norm = (
                        f"{q_norm} " + " ".join(f"#{k}" for k in inferred)
                    )
        else:
            # 태그 필요 없음 → 그대로 진행
            q_norm = q0

        # 3) 푼/시도 제외 필터 부착
        q = _augment_query_with_user_filters(q_norm, handle)

        # 4) 서버 랜덤 정렬 + 중복 제거 + 재시도
        uniq: Dict[int, Any] = {}
        last_resp = None
        for _ in range(max_attempts):
            resp = await search_problems_core(
                query=q, page=1, sort="random"
            )
            last_resp = resp
            for it in (resp.items or []):
                pid = getattr(it, "problemId", None) or getattr(it, "id", None)
                if pid is not None and pid not in uniq:
                    uniq[pid] = it
                    if len(uniq) >= limit:
                        break
            if len(uniq) >= limit:
                break

        if last_resp is None:
            raise RuntimeError("검색 응답이 없습니다.")

        items = list(uniq.values())[:limit]
        last_resp.items = items
        last_resp.count = len(items)  # ✅ 반환 개수로 세팅 (테스트가 기대하는 계약)
        return last_resp


    # ---------- 1) 코어 함수: 실행 함수 ----------
    async def _ensure_tag_catalog() -> TagCatalog:
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
    
    async def get_user_info_core(handle: str) -> UserShowResponse:
        client: httpx.AsyncClient = state.get("http_client")
        if not client:
            raise RuntimeError("HTTP Client is not available.")
        try:
            resp = await client.get("/user/show", params={"handle": handle})
            resp.raise_for_status()
            return UserShowResponse.model_validate(resp.json())
        except httpx.HTTPStatusError as e:
            s = e.response.status_code
            if s == 404:
                raise ValueError(
                    f"사용자 핸들 '{handle}'을(를) 찾을 수 없습니다."
                ) from e
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
                f"네트워크 오류로 사용자 정보를 가져오지 못했습니다: {e}"
            ) from e

    async def search_problems_core(
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

    # ---------- 2) 리소스 (자료실 주소) ----------
    @app.resource(
        "solvedac://tags/catalog",
        description="solved.ac 공식 태그 목록을 로드하여 서버 메모리에 보관합니다.",
    )
    async def solvedac_tag_catalog_resource():
        cat = await _ensure_tag_catalog()  # ← 내부 헬퍼 사용
        return {"count": len(cat.tags), "keys": [t.key for t in cat.tags]}
    
    @app.resource("solvedac://users/{handle}",
                  description="특정 solved.ac 사용자의 기본 정보(레이팅, 티어, 푼 문제 수)를 조회합니다.")
    async def get_user_info(
        handle: str = Field(..., description="조회하려는 사용자의 solved.ac 핸들/아이디"),
    ) -> UserShowResponse:
        return await get_user_info_core(handle)

    @app.resource("solvedac://problems/search/{stub}",
                  description="난이도, 태그, 키워드 등으로 solved.ac 문제를 검색합니다. (예: query='tier:s5..g5 tag:dp')")
    async def search_problems(
        query: str = Field(..., description="문제 검색 쿼리"),
        page: int = Field(1, ge=1, description="페이지 번호(1부터 시작)"),
        stub: str = Field("_", description="템플릿 제약 대응용 더미 세그먼트(무시됨)"),
    ) -> ProblemSearchResponse:
        return await search_problems_core(query=query, page=page)

    # ---------- 3) 프롬프트 (행동 절차 카드) ----------
    @app.prompt(
        name="solvedac.search-workflow",
        description="자연어 조건을 solved.ac 검색 쿼리로 변환하고, 해당 쿼리로 문제 후보를 검토합니다.",
        tags={"solvedac", "search"},
    )
    def search_workflow_prompt(
        natural_request: str = Field(
            ..., description="예: '실버~골드 사이 DP 5문제, 비슷한 태그는 제외'"
        ),
        page: int = Field(1, ge=1, description="검색 페이지"),
    ) -> list[PromptMessage]:
        sys = PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text=(
                    "You are a Solved.ac search assistant.\n"
                    "1) Convert the user's request into a precise Solved.ac query string "
                    "(e.g., `tier:g5..p5 tag:dfs -tag:greedy`).\n"
                    "2) Do NOT browse the web.\n"
                    "3) Call the MCP TOOL `solvedac_search_problems` with {query, page}.\n"
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
                    f"페이지: {page}\n"
                    f"규칙: 쿼리를 먼저 제시하고, 이어서 리소스를 호출해 결과를 평가하세요."
                ),
            ),
        )
        return [sys, usr]

    get_user_info_for_test = get_user_info_core
    search_problems_for_test = search_problems_core
    search_workflow_prompt_for_test = search_workflow_prompt_core
    
    return app
