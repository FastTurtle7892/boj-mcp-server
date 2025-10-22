# solvedac_server.py (핵심 부분만)
from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message, PromptMessage, TextContent
import asyncio
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import httpx
from contextlib import asynccontextmanager

class UserShowResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    handle: str
    tier: int
    rating: int
    solvedCount: int

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

# 2) 문제 검색 툴
@app.tool(
    name="solvedac_search_problems",
    description="난이도/태그/키워드 쿼리로 문제 검색 (예: tier:g5..p5 tag:dfs)"
)
async def search_problems_tool(
    query: str = Field(..., description="검색 쿼리 (예: 'tier:g5..p5 tag:dfs')"),
    page: int = Field(1, ge=1, description="페이지(1부터)")
):
    return await search_problems_core(query=query, page=page)


# ---------- 1) 코어 함수: 실행 함수 ----------
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
            raise ValueError(f"사용자 핸들 '{handle}'을(를) 찾을 수 없습니다.") from e
        if s == 429:
            raise RuntimeError("요청이 많습니다(429). 잠시 후 다시 시도하세요.") from e
        if 500 <= s < 600:
            raise RuntimeError("solved.ac 서버 오류가 발생했습니다. 잠시 후 다시 시도하세요.") from e
        raise
    except httpx.RequestError as e:
        raise RuntimeError(f"네트워크 오류로 사용자 정보를 가져오지 못했습니다: {e}") from e

async def search_problems_core(query: str, page: int = 1) -> ProblemSearchResponse:
    client: httpx.AsyncClient = state.get("http_client")
    if not client:
        raise RuntimeError("HTTP Client is not available.")
    try:
        resp = await client.get("/search/problem", params={"query": query, "page": page})
        resp.raise_for_status()
        return ProblemSearchResponse.model_validate(resp.json())
    except httpx.HTTPStatusError as e:
        s = e.response.status_code
        if s == 429:
            raise RuntimeError("요청이 많습니다(429). 잠시 후 다시 시도하세요.") from e
        if 500 <= s < 600:
            raise RuntimeError("solved.ac 서버 오류가 발생했습니다. 잠시 후 다시 시도하세요.") from e
        raise
    except httpx.RequestError as e:
        raise RuntimeError(f"네트워크 오류로 문제 검색에 실패했습니다: {e}") from e
    
def search_workflow_prompt_core(
    natural_request: str = Field(..., description="예: '실버~골드 사이 DP 5문제, 비슷한 태그는 제외'"),
    page: int = Field(1, ge=1, description="검색 페이지"),
) -> list[PromptMessage]:
    sys = PromptMessage(
        role="assistant",
        content=TextContent(
            type="text",
            text= (
                "You are a Solved.ac search assistant.\n"
                "1) Convert the user's request into a precise Solved.ac query string "
                "(e.g., `tier:g5..p5 tag:dfs -tag:greedy`).\n"
                "2) Do NOT browse the web.\n"
                "3) Call the MCP TOOL `solvedac_search_problems` with {query, page}.\n"
                "4) Rank top 5 by suitability and show: problemId, titleKo, level."
            )    
        )
    )
    usr = PromptMessage(
        role="user",
        content=TextContent(
            type="text",
            text=(
                f"요청(자연어): {natural_request}\n"
                f"페이지: {page}\n"
                f"규칙: 쿼리를 먼저 제시하고, 이어서 리소스를 호출해 결과를 평가하세요."
            )
        )
    )
    return [sys, usr]

# ---------- 2) 리소스 (자료실 주소) ----------
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
    tags={"solvedac", "search"}
)
def search_workflow_prompt(
    natural_request: str = Field(..., description="예: '실버~골드 사이 DP 5문제, 비슷한 태그는 제외'"),
    page: int = Field(1, ge=1, description="검색 페이지"),
) -> list[PromptMessage]:
    sys = PromptMessage(
        role="assistant",
        content=TextContent(
            type="text",
            text= (
                "You are a Solved.ac search assistant.\n"
                "1) Convert the user's request into a precise Solved.ac query string "
                "(e.g., `tier:g5..p5 tag:dfs -tag:greedy`).\n"
                "2) Do NOT browse the web.\n"
                "3) Call the MCP TOOL `solvedac_search_problems` with {query, page}.\n"
                "4) Rank top 5 by suitability and show: problemId, titleKo, level."
            )    
        )
    )
    usr = PromptMessage(
        role="user",
        content=TextContent(
            type="text",
            text=(
                f"요청(자연어): {natural_request}\n"
                f"페이지: {page}\n"
                f"규칙: 쿼리를 먼저 제시하고, 이어서 리소스를 호출해 결과를 평가하세요."
            )
        )
    )
    return [sys, usr]

# ---------- 4) context (항상 곁에 있는 기본 정보) ----------
@app.context(name="defaults",
             description="기본 파라미터")
async def default_context():
    return {
        "lang": "ko"
    }

get_user_info_for_test = get_user_info_core
search_problems_for_test = search_problems_core
search_workflow_prompt_for_test = search_workflow_prompt_core

if __name__ == "__main__":
    app.run()
