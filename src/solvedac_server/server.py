from fastmcp import FastMCP
from fastmcp.prompts.prompt import PromptMessage, TextContent
from pydantic import BaseModel, Field, ConfigDict, computed_field
from typing import Optional, Dict, Any
import httpx
from contextlib import asynccontextmanager
from smithery.decorators import smithery
import solvedac_server.search_query as sq

TIER_NAMES = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ruby"]
TIER_SYMBOLS = ["V", "IV", "III", "II", "I"]
DEFAULT_LIMIT = 5
SOLVEDAC_API_BASE_URL = "https://solved.ac/api/v3"
state = {"http_client": None}

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

    # ---------- 1) 도구 함수: 모델 호출 함수(실행 버튼) ----------
    @app.tool(
        name="solvedac_get_user_info",
        description="solved.ac 사용자의 레이팅/티어/푼 문제 수 조회"
    )
    async def get_user_info_tool(
        handle: str = Field(..., description="사용자 핸들")
    ):
        return await get_user_info_core(handle)

    @app.tool(
        name="solvedac_refresh_tag_catalog",
        description="solved.ac 태그 카탈로그를 강제로 다시 로드합니다.",
    )
    async def refresh_tag_catalog_tool():
        client: httpx.AsyncClient = state.get("http_client")
        raw = await sq.fetch_tag_list_core(client)
        cat = sq.build_tag_catalog(raw)
        state["tag_catalog"] = cat
        return {"refreshed": True, "count": len(cat.tags)}

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
        if sq._needs_tag_catalog(q0, enable_bare_infer):
            cat = await sq._ensure_tag_catalog(state)

            # 1) 명시 태그(#dp, tag:dp ...) 정규화
            q_norm, unknown, _ = sq.normalize_query_tags(
                q0, cat.token_to_key, strict=strict_tags
            )
            if strict_tags and unknown:
                raise ValueError(f"알 수 없는 태그: {', '.join(unknown)}")

            # 2) 요청 시에만 bare-word 승격 (예: '완전탐색' → '#bruteforce')
            if enable_bare_infer:
                inferred = sq.infer_bare_tag_keys(q_norm, cat.token_to_key)
                if inferred:
                    q_norm = (
                        f"{q_norm} " + " ".join(f"#{k}" for k in inferred)
                    )
        else:
            # 태그 필요 없음 → 그대로 진행
            q_norm = q0

        # 3) 푼/시도 제외 필터 부착
        q = sq._augment_query_with_user_filters(q_norm, handle)

        # 4) 서버 랜덤 정렬 + 중복 제거 + 재시도
        uniq: Dict[int, Any] = {}
        last_resp = None
        for _ in range(max_attempts):
            resp = await sq.search_problems_core(
                state, query=q, page=1, sort="random"
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
    
    # ---------- 2) 리소스 함수: 모델이 참고할 자료(자료실 주소) ----------
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
            raw = await sq.fetch_tag_list_core(client)
            cat = sq.build_tag_catalog(raw)
            state["tag_catalog"] = cat
        return {"count": len(cat.tags), "keys": [t.key for t in cat.tags]}

    @app.resource(
        "solvedac://tags/catalog",
        description="solved.ac 공식 태그 목록을 로드하여 서버 메모리에 보관합니다.",
    )
    async def solvedac_tag_catalog_resource():
        cat = await sq._ensure_tag_catalog(state)  # ← 내부 헬퍼 사용
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
    ) -> sq.ProblemSearchResponse:
        return await sq.search_problems_core(state, query=query, page=page)


    # ---------- 3) 프롬프트 함수: 모델이 참고할 행동(행동 절차 카드) ----------
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
    search_problems_for_test = sq.search_problems_core
    search_workflow_prompt_for_test = sq.search_workflow_prompt_core
    
    return app