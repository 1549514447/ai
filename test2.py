import asyncio
from core.orchestrator.intelligent_qa_orchestrator import IntelligentQAOrchestrator

async def test_qa_query(orchestrator, question):
    print(f"问题：{question}")
    try:
        result = await orchestrator.answer_question(question)
        print("返回：", result)
    except Exception as e:
        print("出错：", e)
    print("-" * 60)

async def main():
    # 实例化智能编排器（如需传入AI client可在此处传入）
    orchestrator = IntelligentQAOrchestrator()
    
    # 你可以根据实际情况替换/扩展这些问题
    test_questions = [
        "6月1日有多少产品到期，总资金多少？",
        "5月28日入金是多少？",
        "6月1日至6月30日产品到期金额是多少？如果使用25%复投，7月1日剩余资金有多少？",
        "6月3日到期产品有哪些？",
        "本周到期金额和今日到期金额有什么变化趋势？",
        "目前公司活跃会员有多少？",
        "5月29日至6月30日会有多少产品到期，公司需要准备多少资金？",
        "上个星期的出金平均每日增长多少？",
        "给我5月5日的数据",
        "假设现在是没入金的情况公司还能运行多久？"
    ]
    for q in test_questions:
        await test_qa_query(orchestrator, q)

if __name__ == "__main__":
    asyncio.run(main())