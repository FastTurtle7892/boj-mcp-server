import pytest
import asyncio
from unittest.mock import MagicMock, patch
from solvedac_server.server import create_server, UserShowResponse, ProblemSearchResponse, Problem
from fastmcp import FastMCP, Client

# ----------------------------------------
# 1. Fixture 정의: FastMCP 서버 및 클라이언트
# ----------------------------------------

@pytest.fixture(scope="module")
def mcp_app():
    """테스트용 FastMCP 서버 인스턴스를 생성하는 픽스처"""
    # create_server() 함수는 FastMCP 앱 인스턴스를 반환합니다.
    return create_server()

@pytest.fixture
async def mcp_client(mcp_app: FastMCP):
    """인메모리 FastMCP 클라이언트를 생성하고, 사용 후 종료하는 픽스처"""
    # FastMCP는 서버 인스턴스를 직접 클라이언트의 transport target으로 사용하여
    # 실제 네트워크 호출 없이 인메모리 테스트를 지원합니다.
    async with Client(mcp_app) as client:
        # lifespan은 서버 실행 시 자동으로 관리되지만,
        # 코어 함수를 직접 호출할 경우를 위해 lifespan 컨텍스트를 수동으로 시작합니다.
        async with mcp_app.lifespan(mcp_app):
             yield client
        # with 블록 종료 시 lifespan도 종료됩니다.

# ----------------------------------------
# 2. 코어 함수 유닛 테스트 (Mocking을 사용하여 외부 HTTP 통신 차단)
# ----------------------------------------

# 실제 HTTP 요청을 Mocking하기 위한 Helper
def mock_response(status_code, json_data=None):
    """httpx 응답 객체를 Mock합니다."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data if json_data is not None else {}
    mock.raise_for_status.side_effect = None if 200 <= status_code < 300 else lambda: pytest.fail(f"HTTPStatusError with code {status_code}")
    if not (200 <= status_code < 300):
        # HTTPStatusError 발생을 위한 더미 응답 객체 설정
        from httpx import HTTPStatusError
        mock.raise_for_status.side_effect = HTTPStatusError(
            message=f"Status code {status_code}", 
            request=MagicMock(), 
            response=mock
        )
    return mock

@pytest.mark.asyncio
async def test_get_user_info_core_success(mcp_app: FastMCP):
    """get_user_info_core 함수 성공 케이스 테스트"""
    expected_data = {"handle": "testuser", "tier": 15, "rating": 1500, "solvedCount": 100}
    
    with patch("httpx.AsyncClient.get") as mock_get:
        # Mocking된 응답 설정
        mock_get.return_value = mock_response(200, expected_data)
        
        # Lifespan 컨텍스트 내에서 테스트 실행
        async with mcp_app.lifespan(mcp_app):
            result = await mcp_app.get_user_info_for_test("testuser")

        assert isinstance(result, UserShowResponse)
        assert result.handle == "testuser"
        assert result.rating == 1500
        mock_get.assert_called_once_with("/user/show", params={"handle": "testuser"})

@pytest.mark.asyncio
async def test_get_user_info_core_404_error(mcp_app: FastMCP):
    """get_user_info_core 함수 404 에러 처리 테스트"""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(404)
        
        async with mcp_app.lifespan(mcp_app):
            with pytest.raises(ValueError, match="사용자 핸들 'nonexistent'을\\(를\\) 찾을 수 없습니다."):
                await mcp_app.get_user_info_for_test("nonexistent")

@pytest.mark.asyncio
async def test_search_problems_core_success(mcp_app: FastMCP):
    """search_problems_core 함수 성공 케이스 테스트"""
    problem_data = {"problemId": 1000, "titleKo": "A+B", "level": 1, "isSolvable": True}
    expected_data = {"count": 1, "items": [problem_data]}
    
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, expected_data)
        
        async with mcp_app.lifespan(mcp_app):
            result = await mcp_app.search_problems_for_test(query="tier:b5", page=1)

        assert isinstance(result, ProblemSearchResponse)
        assert result.count == 1
        assert len(result.items) == 1
        assert result.items[0].problemId == 1000
        mock_get.assert_called_once_with("/search/problem", params={"query": "tier:b5", "page": 1})

# ----------------------------------------
# 3. FastMCP Tool 호출 테스트 (인메모리 클라이언트 사용)
# ----------------------------------------

@pytest.mark.asyncio
async def test_tool_get_user_info(mcp_client: Client):
    """solvedac_get_user_info 툴 호출 테스트 (코어 함수는 이미 목킹이 되어 있다고 가정)"""
    # NOTE: 툴 테스트는 일반적으로 실제 코어 함수를 사용하므로, 코어 함수가 이미 목킹되어 있다고 가정하고
    # 여기서는 FastMCP 클라이언트의 툴 호출 로직이 정상 동작하는지 확인합니다.
    # 하지만 코어 함수에서 실제 HTTP 통신을 하므로, 여기서는 코어 함수 내부의 HTTP 호출을 Mocking합니다.
    expected_data = {"handle": "testtool", "tier": 15, "rating": 1500, "solvedCount": 100}

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, expected_data)

        # 툴 함수를 클라이언트를 통해 호출
        result = await mcp_client.call_tool("solvedac_get_user_info", handle="testtool")
    
        assert isinstance(result, UserShowResponse)
        assert result.handle == "testtool"

@pytest.mark.asyncio
async def test_tool_search_problems_limit(mcp_client: Client):
    """solvedac_search_problems 툴의 limit 기능 테스트"""
    # Mocking된 5개의 문제 데이터
    mock_problems = [{"problemId": i, "titleKo": f"Problem {i}", "level": 10, "isSolvable": True} for i in range(1, 6)]
    mock_data = {"count": 50, "items": mock_problems * 10} # 50개(items)가 반환되었다고 가정
    
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, mock_data)

        # limit=3으로 툴 호출
        result = await mcp_client.call_tool("solvedac_search_problems", query="tier:d1", limit=3)

        assert isinstance(result, ProblemSearchResponse)
        # 결과 목록은 3개로 제한되어야 함
        assert len(result.items) == 3
        # count 값도 3으로 수정되어야 함
        assert result.count == 3
        assert result.items[0].problemId == 1

@pytest.mark.asyncio
async def test_tool_search_problems_default_limit(mcp_client: Client):
    """solvedac_search_problems 툴의 기본 limit=5 테스트"""
    # Mocking된 10개의 문제 데이터
    mock_problems = [{"problemId": i, "titleKo": f"Problem {i}", "level": 10, "isSolvable": True} for i in range(1, 11)]
    mock_data = {"count": 50, "items": mock_problems * 5} # 50개(items)가 반환되었다고 가정

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, mock_data)

        # limit을 지정하지 않고 툴 호출
        result = await mcp_client.call_tool("solvedac_search_problems", query="tier:d1")

        assert isinstance(result, ProblemSearchResponse)
        # 기본 limit인 5개로 제한되어야 함
        assert len(result.items) == 5
        assert result.count == 5

# ----------------------------------------
# 4. Prompt 함수 테스트 (단순히 프롬프트 구조가 올바른지 확인)
# ----------------------------------------

def test_search_workflow_prompt(mcp_app: FastMCP):
    """search_workflow_prompt 함수 반환 구조 테스트"""
    natural_req = "실버~골드 사이 DP 5문제"
    page_num = 2
    
    # 프롬프트 함수를 직접 호출
    result = mcp_app.search_workflow_prompt_for_test(natural_request=natural_req, page=page_num)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].role == "assistant"
    assert "Convert the user's request into a precise Solved.ac query string" in result[0].content.text
    assert result[1].role == "user"
    assert f"요청(자연어): {natural_req}" in result[1].content.text
    assert f"페이지: {page_num}" in result[1].content.text