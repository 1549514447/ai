# core/models/openai_client.py
import openai
from typing import Dict, Any, Optional, List
import json
import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, api_key: str = None):
        """
        初始化GPT-4o客户端

        Args:
            api_key: API密钥，如果不传入则从环境变量OPENAI_API_KEY获取
        """
        # 如果没有传入api_key，从环境变量获取
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")

        # 创建一个不带代理的 httpx 异步客户端
        import httpx

        # 直接创建不带代理的 httpx 异步客户端
        custom_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            follow_redirects=True
        )

        # 使用自定义 http 客户端初始化 OpenAI
        self.client = openai.AsyncClient(
            api_key=api_key,
            http_client=custom_http_client
        )

        self.model = "gpt-4o"
        self.max_tokens = 4000

        logger.info("OpenAIClient initialized successfully")

    async def process_direct_query(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        直接查询处理 - GPT-4o的快速响应优势
        """
        try:
            system_prompt = """你是一个精确高效的数据分析助手。

你的核心能力：
1. 快速准确的数据查询和展示
2. 精确的数值计算和统计分析
3. 标准格式的数据整理和呈现
4. 直接明了的问题回答

你的特点：
- 响应速度快，准确性高
- 数据处理精确，计算无误
- 格式规范，易于理解
- 避免过度分析，专注于用户的直接需求

回答要求：
- 直接回答用户问题
- 数据准确，格式清晰
- 包含必要的计算过程
- 简洁专业，避免冗余"""

            user_content = f"""
用户查询：{query}

可用数据：
{json.dumps(data, ensure_ascii=False, indent=2)}

请直接回答用户的问题，提供准确的数据和计算结果。
"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=self.max_tokens
            )

            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model_used": "gpt-4o",
                "query_type": "direct_query",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"GPT-4o直接查询失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "gpt-4o",
                "timestamp": datetime.now().isoformat()
            }

    async def execute_calculations(self, calculation_requests: List[Dict[str, Any]], data: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        精确计算执行 - GPT-4o的计算优势
        """
        try:
            calc_prompt = f"""
执行以下计算请求，确保数值精确和逻辑正确：

计算请求：
{json.dumps(calculation_requests, ensure_ascii=False, indent=2)}

可用数据：
{json.dumps(data, ensure_ascii=False, indent=2)}

要求：
1. 每个计算都要显示详细步骤
2. 确保数值精确性
3. 标注数据来源
4. 验证计算逻辑

返回JSON格式：
{{
    "calculations": [
        {{
            "request_id": "计算请求ID",
            "description": "计算描述",
            "steps": [
                {{
                    "step": 1,
                    "description": "步骤描述",
                    "formula": "计算公式",
                    "input_values": {{"值1": 123, "值2": 456}},
                    "result": 结果数值
                }}
            ],
            "final_result": 最终结果,
            "formatted_result": "格式化显示",
            "data_sources": ["数据来源1", "数据来源2"],
            "confidence": "计算置信度评估"
        }}
    ],
    "summary": {{
        "total_calculations": 计算数量,
        "all_successful": true/false,
        "key_results": {{"关键结果1": 值1, "关键结果2": 值2}}
    }}
}}
"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": calc_prompt}],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            calculations = json.loads(response.choices[0].message.content)

            return {
                "success": True,
                "calculations": calculations,
                "model_used": "gpt-4o",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"计算执行失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "gpt-4o"
            }

    async def format_data_response(self, raw_data: Dict[str, Any], query: str, format_type: str = "comprehensive") -> \
    Dict[str, Any]:
        """
        数据格式化响应 - 修复JSON强制要求问题
        """
        try:
            format_prompt = f"""
    将以下原始数据格式化为用户友好的回答：

    用户查询：{query}
    原始数据：{json.dumps(raw_data, ensure_ascii=False, indent=2)}
    格式类型：{format_type}

    格式化要求：
    - 直接回答用户问题
    - 数据清晰易读
    - 包含关键数值
    - 使用表格或列表组织信息
    - 添加必要的单位和说明

    返回格式化的文本回答。
    """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": format_prompt}],
                temperature=0.2,
                max_tokens=2000
                # 🔥 移除 response_format={"type": "json_object"} - 这是问题根源
            )

            return {
                "success": True,
                "formatted_response": response.choices[0].message.content,
                "model_used": "gpt-4o",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"数据格式化失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "gpt-4o"
            }

    async def validate_and_verify(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        结果验证 - 确保回答的准确性和完整性
        """
        try:
            validation_prompt = f"""
验证以下分析结果的准确性和完整性：

原始查询：{original_query}
分析结果：{json.dumps(result, ensure_ascii=False, indent=2)}

验证要点：
1. 数值计算是否正确
2. 逻辑推理是否合理
3. 是否完整回答了用户问题
4. 数据引用是否准确
5. 结论是否基于证据

返回JSON格式：
{{
    "validation_status": "passed/failed/warning",
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "issues": [
        {{
            "type": "calculation_error/logic_error/incomplete_answer",
            "description": "问题描述",
            "severity": "high/medium/low",
            "suggestion": "修正建议"
        }}
    ],
    "strengths": ["优点1", "优点2"],
    "overall_assessment": "整体评估"
}}
"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            validation = json.loads(response.choices[0].message.content)

            return {
                "success": True,
                "validation": validation,
                "model_used": "gpt-4o",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"结果验证失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "gpt-4o"
            }

    # ============= 为 intelligent_router.py 提供的方法别名 =============

    async def process_simple_query(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        简单查询处理 - 为 intelligent_router.py 提供的方法别名
        """
        return await self.process_direct_query(query, data)

    async def precise_calculation(self, calculation_request: Dict[str, Any],
                                  data: Dict[str, Any]) -> Dict[str, Any]:
        """
        精确计算 - 为 intelligent_router.py 提供的方法别名
        单个计算请求的处理
        """
        try:
            # 包装单个计算请求为列表格式
            if isinstance(calculation_request, dict):
                calculation_requests = [calculation_request]
            else:
                calculation_requests = calculation_request

            result = await self.execute_calculations(calculation_requests, data)

            # 返回单个计算结果（如果原本是单个请求）
            if isinstance(calculation_request, dict) and result["success"]:
                calculations = result.get("calculations", {}).get("calculations", [])
                if calculations:
                    return {
                        "success": True,
                        "calculation": calculations[0],
                        "model_used": "gpt-4o",
                        "timestamp": result.get("timestamp")
                    }

            return result

        except Exception as e:
            logger.error(f"精确计算失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "gpt-4o"
            }