# MCP Server for BOJ

### **BOJ를 위한 Model Context Protocol (MCP) 서버입니다.**  
AI 어시스턴트가 solved.ac의 사용자 정보와 문제 데이터를 조회할 수 있도록 해줍니다.

---

### **pytest status**
[![FastMCP CI Tests](https://github.com/Junwoo-Seo-1998/boj-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/Junwoo-Seo-1998/boj-mcp-server/actions/workflows/ci.yml)

---
### **solved.ac 사이트의 MCP  저장소**
[![smithery badge](https://smithery.ai/badge/@Junwoo-Seo-1998/boj-mcp-server)](https://smithery.ai/server/@Junwoo-Seo-1998/boj-mcp-server)

---

## 특징 (Features)

- **Solved.ac API에 빠르게 접근**
- **사용자 프로필 조회** (레이팅, 티어, 푼 문제 수)
- **난이도, 태그, 키워드로 문제 검색**

---

## 설치 (Installation)

### Smithery를 통한 설치

Smithery를 이용해 Claude Desktop 등에 mcp-server를 자동 설치할 수 있습니다.

```bash
npx -y @smithery/cli install @Junwoo-Seo-1998/boj-mcp-server --client claude
```

---

## 사용법 (Usage)

### 라이브러리로 사용하기

`server.py`의 핵심 함수들을 직접 호출하여 파이썬 코드 내에서 사용할 수 있습니다.

```python
import asyncio
from solvedac_server.server import create_server, lifespan
# 테스트용으로 노출된 코어 함수 임포트
from solvedac_server.server import (
    get_user_info_for_test as get_user_info,
    search_problems_for_test as search_problems,
    search_workflow_prompt_for_test,
)

async def main():
    app = create_server()
    async with lifespan(app):
        # 1. 유저 정보 조회
        user = await get_user_info("kyungbaee")
        print("USER:", user.model_dump())

        # 2. 문제 검색 (limit 파라미터 없이 기본 50개)
        res_all = await search_problems(query="tier:s5..g5 tag:dp", page=1)
        print("SEARCH COUNT (Full):", res_all.count)
        
        # 3. 프롬프트 메시지 생성
        msgs = search_workflow_prompt_for_test(
            natural_request="실버~골드 사이 DP 5문제, 그리디 제외",
            page=1,
        )
        print("=== Prompt messages ===")
        for m in msgs:
            print(f"ROLE: {m.role}\n{m.content.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 사용 가능한 도구 (Available Tools)

| 도구 이름 (Tool Name) | 설명 (Description) | 파라미터 (Parameters) |
|-----------------------|--------------------|------------------------|
| **solvedac_get_user_info** | solved.ac 사용자의 레이팅/티어/푼 문제 수 조회 | `handle (string, required)`: 사용자 핸들 |
| **solvedac_search_problems** | 난이도/태그/키워드 쿼리로 문제 검색 | `query (string, required)`: 검색 쿼리 (예: `tier:g5..p5 tag:dfs`)<br>`page (int, default 1)`: 페이지(1부터)<br>`limit (int, default 5)`: 결과를 상위 N개로 제한 (1~20) |

---

## 사용 가능한 리소스 (Available Resources)

| 리소스 경로 | 설명 |
|--------------|------|
| `solvedac://users/{handle}` | 특정 solved.ac 사용자의 기본 정보(레이팅, 티어, 푼 문제 수)를 조회 |
| `solvedac://problems/search/{stub}` | 난이도, 태그, 키워드 등으로 solved.ac 문제 검색<br>쿼리 파라미터: `query (string)`, `page (int)`<br>(경로의 `{stub}` 값은 무시됩니다.) |

---

## 사용 가능한 프롬프트 (Available Prompts)

| 프롬프트 이름 (Prompt Name) | 설명 (Description) | 파라미터 (Parameters) |
|-----------------------------|--------------------|------------------------|
| **solvedac.search-workflow** | 자연어 조건을 solved.ac 검색 쿼리로 변환하고, 해당 쿼리로 문제 후보를 검토 | `natural_request (string)`: 예: `"실버~골드 사이 DP 5문제"`<br>`page (int, default 1)`: 검색 페이지 |

---

## 로컬 개발 (Local Development)

저장소를 클론하고 의존성을 설치합니다.

```bash
git clone https://github.com/Junwoo-Seo-1998/boj-mcp-server.git
cd boj-mcp-server

# (가상 환경 사용 권장)
pip install -e .
```

### 개발 서버 실행

```bash
smithery dev
```

### Smithery 플레이그라운드 실행

```bash
smithery playground
```

---

## 관련 프로젝트 (Related Projects)

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) — MCP 명세 및 문서  
- [Smithery.ai](https://smithery.ai) — MCP 서버 빌드 및 배포 도구  
- [Solved.ac](https://solved.ac) — 백준(BOJ) 문제 아카이브  

---

## 감사 인사 (Acknowledgements)

이 프로젝트는 **solved.ac API**를 기반으로 합니다.
