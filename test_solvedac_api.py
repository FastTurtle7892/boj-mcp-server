import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from solvedac_server.server import create_server, UserShowResponse
from solvedac_server.search_query import ProblemSearchResponse
from fastmcp import FastMCP, Client
from mcp.shared.exceptions import McpError

# 1. Fixtures
@pytest.fixture(scope="module")
def mcp_app():
    """테스트용 FastMCP 서버 인스턴스를 생성하는 픽스처"""
    return create_server()

@pytest_asyncio.fixture
async def mcp_client(mcp_app: FastMCP):
    """앱의 생명주기를 자동으로 관리하는 인메모리 클라이언트를 제공합니다."""
    async with Client(mcp_app._fastmcp) as client:
        yield client

# 2. Mock Helper
def mock_response(status_code, json_data=None):
    """httpx 응답 객체를 Mock합니다."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data if json_data is not None else {}
    if not (200 <= status_code < 300):
        from httpx import HTTPStatusError
        mock.raise_for_status.side_effect = HTTPStatusError(
            message=f"Status code {status_code}", request=MagicMock(), response=mock
        )
    return mock

# 3. Resource Tests
@pytest.mark.asyncio
async def test_resource_get_user_info_success(mcp_client: Client):
    """get_user_info 리소스의 성공 케이스를 테스트합니다."""
    expected_data = {"handle": "testuser", "tier": 15, "rating": 1500, "solvedCount": 100}
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, expected_data)
        result = await mcp_client.read_resource("solvedac://users/testuser")
    
    parsed_result = UserShowResponse.model_validate_json(result[0].text)
    assert isinstance(parsed_result, UserShowResponse)
    assert parsed_result.handle == "testuser"
    mock_get.assert_called_once_with("/user/show", params={"handle": "testuser"})

@pytest.mark.asyncio
async def test_resource_get_user_info_404_error(mcp_client: Client):
    """get_user_info 리소스의 404 에러 케이스를 테스트합니다."""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(404)
        with pytest.raises(McpError, match=r"사용자 핸들 'nonexistent'을\(를\) 찾을 수 없습니다."):
            await mcp_client.read_resource("solvedac://users/nonexistent")

@pytest.mark.xfail(reason="fastmcp.Client.read_resource는 현재 URI 경로 외 파라미터 전달을 지원하지 않는 것으로 보임")
@pytest.mark.asyncio
async def test_resource_search_problems_success(mcp_client: Client):
    """search_problems 리소스의 성공 케이스를 테스트합니다."""
    problem_data = {"problemId": 1000, "titleKo": "A+B", "level": 1, "isSolvable": True}
    expected_data = {"count": 1, "items": [problem_data]}
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, expected_data)
        # 참고: 이 API 호출은 현재 프레임워크에서 동작하지 않음
        result = await mcp_client.read_resource("solvedac://problems/search/_?query=tier:b5&page=1")

    parsed_result = ProblemSearchResponse.model_validate_json(result[0].text)
    assert isinstance(parsed_result, ProblemSearchResponse)
    assert parsed_result.count == 1
    assert parsed_result.items[0].problemId == 1000
    mock_get.assert_called_once_with("/search/problem", params={"query": "tier:b5", "page": 1})

# 4. Tool Tests
@pytest.mark.asyncio
async def test_tool_get_user_info(mcp_client: Client):
    """solvedac_get_user_info 툴 호출을 테스트합니다."""
    expected_data = {"handle": "testtool", "tier": 15, "rating": 1500, "solvedCount": 100}
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, expected_data)
        result = await mcp_client.call_tool("solvedac_get_user_info", {"handle": "testtool"})
    
    # call_tool의 결과는 dict이므로, Pydantic 모델로 변환하여 검증합니다.
    parsed_result = UserShowResponse.model_validate(result.data)
    assert isinstance(parsed_result, UserShowResponse)
    assert parsed_result.handle == "testtool"

@pytest.mark.asyncio
async def test_tool_search_problems_limit(mcp_client: Client):
    """solvedac_search_problems 툴의 limit 기능을 테스트합니다."""
    mock_problems = [{"problemId": i, "titleKo": f"Problem {i}", "level": 10, "isSolvable": True} for i in range(1, 6)]
    mock_data = {"count": 50, "items": mock_problems * 10}
    
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, mock_data)
        result = await mcp_client.call_tool("solvedac_search_problems", {"query": "tier:d1", "limit": 3})

    parsed_result = ProblemSearchResponse.model_validate(result.data)
    assert isinstance(parsed_result, ProblemSearchResponse)
    assert len(parsed_result.items) == 3
    assert parsed_result.count == 3

@pytest.mark.asyncio
async def test_tool_search_problems_default_limit(mcp_client: Client):
    """solvedac_search_problems 툴의 기본 limit 기능을 테스트합니다."""
    mock_problems = [{"problemId": i, "titleKo": f"Problem {i}", "level": 10, "isSolvable": True} for i in range(1, 11)]
    mock_data = {"count": 50, "items": mock_problems * 5}

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = mock_response(200, mock_data)
        result = await mcp_client.call_tool("solvedac_search_problems", {"query": "tier:d1"})

    parsed_result = ProblemSearchResponse.model_validate(result.data)
    assert isinstance(parsed_result, ProblemSearchResponse)
    assert len(parsed_result.items) == 5
    assert parsed_result.count == 5

# 5. Prompt Test
@pytest.mark.asyncio
async def test_prompt_search_workflow(mcp_client: Client):
    """search_workflow_prompt 프롬프트의 구조를 테스트합니다."""
    natural_req = "실버~골드 사이 DP 5문제"
    page_num = 2
    
    result = await mcp_client.get_prompt("solvedac.search-workflow", {"natural_request": natural_req, "page": page_num})
    
    assert isinstance(result.messages, list)
    assert len(result.messages) == 2
    assert result.messages[0].role == "assistant"
    assert "Convert the user's request" in result.messages[0].content.text
    assert result.messages[1].role == "user"
    assert f"요청(자연어): {natural_req}" in result.messages[1].content.text
    assert f"페이지: {page_num}" in result.messages[1].content.text
