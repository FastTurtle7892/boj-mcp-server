# client_resources.py
import asyncio
from solvedac_server import app, lifespan
from solvedac_server import get_user_info_for_test as get_user_info
from solvedac_server import search_problems_for_test as search_problems
from solvedac_server import search_workflow_prompt_for_test

async def main():
    async with lifespan(app):
        user = await get_user_info("kyungbaee")
        print("USER:", user.model_dump())

        res = await search_problems(query="tier:s5..g5 tag:dp", page=1)
        print("SEARCH COUNT:", res.count)
        if res.items:
            first = res.items[0]
            print("FIRST PROBLEM:", first.problemId, first.titleKo, first.level, first.isSolvable)

            
        # 1) 프롬프트 실행 → system/user 메시지 시퀀스 생성
        msgs = search_workflow_prompt_for_test(
            natural_request="실버~골드 사이 DP 5문제, 그리디 제외",
            page=1,
        )

        print("=== Prompt messages ===")
        for i, m in enumerate(msgs, 1):
            role = getattr(m, "role", type(m).__name__)
            content = getattr(m, "content", m)
            print(f"{i}. role={role}\n{content}\n")

        # 2) (데모) 실제 MCP 클라이언트라면, 모델이 쿼리를 만들어 리소스를 호출합니다.
        # 여기서는 수동으로 쿼리를 넣어 리소스 호출만 재연합니다.
        query = "tier:s5..g5 tag:dp -tag:greedy"
        res = await search_problems(query=query, page=1)
        print(f"SEARCH COUNT: {res.count}")

        for p in res.items[:5]:
            print(f"{p.problemId} | {p.titleKo} | level={p.level}")


if __name__ == "__main__":
    asyncio.run(main())
