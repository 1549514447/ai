# data/processors/current_data_processor.py
"""
📊 当前数据处理器 (优化版)
专门处理已获取的实时状态数据，进行AI辅助分析、格式化并构建响应。
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass, field  # 确保导入 field
from enum import Enum
import json
from functools import lru_cache  # 保留用于AI分析结果缓存
import hashlib

# AI 客户端导入 (假设从顶层 core.models 导入)
from core.models.claude_client import ClaudeClient, CustomJSONEncoder  # 假设ClaudeClient在core.models
from core.models.openai_client import OpenAIClient  # 假设OpenAIClient在core.models

# QueryAnalysisResult 导入，因为它现在是输入参数
from core.analyzers.query_parser import QueryAnalysisResult  # 使用别名
from core.analyzers.query_parser import QueryType as QueryParserQueryType

logger = logging.getLogger(__name__)


class CurrentDataQueryType(Enum):
    """当前数据查询的细化类型 (由本处理器内部AI识别或从QueryAnalysisResult映射)"""
    SYSTEM_OVERVIEW = "system_overview"
    BALANCE_CHECK = "balance_check"
    USER_STATUS = "user_status"
    TODAY_EXPIRY = "today_expiry"
    CASH_FLOW = "cash_flow"
    PRODUCT_STATUS = "product_status"
    QUICK_METRICS = "quick_metrics"
    UNKNOWN_CURRENT_QUERY = "unknown_current_query"  # 新增，用于无法细化的情况


class ResponseFormat(Enum):
    """响应格式类型"""
    SIMPLE_TEXT = "simple_text"
    DETAILED_SUMMARY = "detailed_summary"
    METRICS_FOCUSED = "metrics_focused"
    BUSINESS_ORIENTED = "business_oriented"


@dataclass
class CurrentDataResponse:
    """当前数据处理器的响应结果"""
    query_type: CurrentDataQueryType
    response_format: ResponseFormat
    main_answer: str
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    formatted_data: Dict[str, str] = field(default_factory=dict)  # <--- 这里期望 Dict[str, str]
    related_metrics: Dict[str, Any] = field(default_factory=dict)
    business_context: str = ""
    quick_insights: List[str] = field(default_factory=list)
    data_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_confidence: float = 0.0
    data_sources_references: List[str] = field(default_factory=list)  # 指向编排器提供的数据源
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)  # 可存储AI耗时等


class CurrentDataProcessor:
    """
    📊 当前数据处理器 (优化版)
    专注于基于已获取的当前系统数据，进行AI辅助的分析、格式化和响应构建。
    """

    def __init__(self, claude_client: Optional[ClaudeClient] = None,
                 gpt_client: Optional[OpenAIClient] = None):
        """
        初始化当前数据处理器。
        Args:
            claude_client: Claude客户端，用于业务理解和洞察。
            gpt_client: GPT客户端，用于数据格式化和计算辅助。
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client  # GPT目前在此处理器中用途较少，但保留以便未来扩展

        self.processing_config = self._load_processing_config()
        # self.query_patterns = self._load_current_data_patterns() # 简单关键词匹配可能不再需要，依赖传入的QueryAnalysisResult

        self.processing_stats = {
            'total_queries_processed': 0,
            'queries_by_sub_type': {},
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'ai_cache_hits': 0,  # 专门统计本处理器AI缓存
            'ai_cache_misses': 0
        }
        logger.info("CurrentDataProcessor initialized (optimized for external data input and AI assistance)")

    def _get_query_hash(self, text_to_hash: str) -> str:
        """为文本生成MD5哈希值，用于缓存键。"""
        return hashlib.md5(text_to_hash.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=50)  # 缓存最近50个AI分析结果
    async def _cached_ai_analysis(self, query_or_prompt_hash: str, prompt: str,
                                  ai_client_type: str = "claude") -> Dict[str, Any]:
        """
        带缓存的AI分析调用。
        Args:
            query_or_prompt_hash: 基于查询或完整提示生成的哈希，用作缓存键。
            prompt: 发送给AI的完整提示词。
            ai_client_type: "claude" 或 "gpt"。
        Returns:
            AI的响应字典，或包含错误的字典。
        """
        # 注意：lru_cache 对于异步方法需要特殊处理或使用异步缓存库。
        # 为简单起见，这里的lru_cache主要作用于同步部分（参数哈希），实际AI调用是异步的。
        # 更优的异步缓存方案可以使用例如 'async-cache' 库。
        # 当前的 lru_cache 装饰器可能不会按预期对异步函数的不同调用缓存，
        # 因为它缓存的是协程对象本身。我们将依赖 prompt_hash 来实现手动检查。

        # 手动模拟缓存检查 (因为lru_cache对async方法的限制)
        # if query_or_prompt_hash in manual_cache and (time.time() - manual_cache[query_or_prompt_hash]['timestamp'] < TTL):
        #     return manual_cache[query_or_prompt_hash]['data']

        self.processing_stats['ai_cache_misses'] += 1
        logger.debug(f"AI cache MISS for hash: {query_or_prompt_hash}. Executing AI call with {ai_client_type}.")

        client_to_use = self.claude_client if ai_client_type == "claude" else self.gpt_client

        if not client_to_use:
            logger.error(f"{ai_client_type.capitalize()}Client 未初始化。")
            return {"success": False, "error": f"{ai_client_type.capitalize()} 服务不可用",
                    "response": f"AI分析服务（{ai_client_type}）不可用"}

        try:
            # 统一调用AI客户端的方法签名 (假设都有一个通用的文本生成/分析方法)
            # 以下是基于 ClaudeClient 的 messages API 示例
            if isinstance(client_to_use, ClaudeClient) and hasattr(client_to_use, 'messages') and callable(
                    getattr(client_to_use.messages, 'create', None)):
                logger.debug(f"使用 ClaudeClient (messages API) 进行分析。Prompt hash: {query_or_prompt_hash}")
                # Claude SDK 的 messages.create 是同步的，需要用 asyncio.to_thread 运行
                response_raw = await asyncio.to_thread(
                    client_to_use.messages.create,
                    model=getattr(client_to_use, 'model', "claude-3-sonnet-20240229"),  # 从客户端实例获取模型或用默认
                    max_tokens=1024,  # 根据任务调整
                    messages=[{"role": "user", "content": prompt}]
                )
                content_text = ""
                if hasattr(response_raw, 'content') and response_raw.content:
                    for content_item in response_raw.content:
                        if hasattr(content_item, 'text'): content_text += content_item.text
                return {"success": True, "response": content_text, "model_used": ai_client_type}

            # 可以添加对OpenAIClient或其他调用方式的适配逻辑
            elif isinstance(client_to_use, OpenAIClient) and hasattr(client_to_use, 'chat') and hasattr(
                    client_to_use.chat, 'completions') and callable(
                    getattr(client_to_use.chat.completions, 'create', None)):
                logger.debug(f"使用 OpenAIClient (chat.completions API) 进行分析。Prompt hash: {query_or_prompt_hash}")
                response_raw = await asyncio.to_thread(
                    client_to_use.chat.completions.create,
                    model=getattr(client_to_use, 'model', "gpt-4o"),
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                if response_raw.choices and response_raw.choices[0].message:
                    return {"success": True, "response": response_raw.choices[0].message.content,
                            "model_used": ai_client_type}

            logger.error(f"无法找到合适的AI调用方法 для {ai_client_type}Client。")
            return {"success": False, "error": "AI客户端方法不兼容", "response": "AI分析服务调用失败"}

        except Exception as e:
            logger.error(f"AI分析调用时发生错误 ({ai_client_type}): {str(e)}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e), "response": f"AI分析出错: {str(e)}"}

    async def _determine_current_data_sub_type(self, user_query: str,
                                               query_analysis: QueryAnalysisResult) -> CurrentDataQueryType:
        """
        根据 SmartQueryParser 的结果或通过AI进一步细化当前数据查询的子类型。
        """
        # 优先使用 SmartQueryParser 的结果进行映射
        # QueryParserQueryType 定义在 query_parser.py
        parser_query_type = query_analysis.query_type

        if parser_query_type == QueryParserQueryType.DATA_RETRIEVAL:
            # 需要更细致的逻辑来从 query_analysis.data_requirements 或 entities 中判断
            # 例如，如果实体是“余额”，则为 BALANCE_CHECK
            # 为简化，这里直接进行AI细化或基于关键词（如果需要的话）
            pass  # 继续下面的AI细化
        elif parser_query_type == QueryParserQueryType.SYSTEM_COMMAND and "概览" in user_query:  # 特殊处理
            return CurrentDataQueryType.SYSTEM_OVERVIEW
        # 如果 SmartQueryParser 已经给出了非常明确的、可直接映射的类型，可以在此处理

        # 如果 SmartQueryParser 的结果不够细化，或需要二次确认，再调用AI
        logger.debug(
            f"SmartQueryParser type '{parser_query_type.value}' not specific enough or needs confirmation for CurrentDataProcessor. Using AI to refine sub-type.")

        if not self.claude_client:
            logger.warning("ClaudeClient未初始化，无法细化当前数据查询子类型。基于关键词进行简单判断。")
            # 简单关键词匹配作为后备
            query_lower = user_query.lower()
            if any(
                kw in query_lower for kw in ["余额", "总资金", "有多少钱"]): return CurrentDataQueryType.BALANCE_CHECK
            if any(
                kw in query_lower for kw in ["用户数", "会员数", "活跃用户"]): return CurrentDataQueryType.USER_STATUS
            if any(kw in query_lower for kw in ["今天到期", "今日到期"]): return CurrentDataQueryType.TODAY_EXPIRY
            if any(kw in query_lower for kw in ["入金", "出金", "资金流动"]): return CurrentDataQueryType.CASH_FLOW
            if any(kw in query_lower for kw in ["产品情况", "理财状态"]): return CurrentDataQueryType.PRODUCT_STATUS
            if any(kw in query_lower for kw in ["关键指标", "核心数据"]): return CurrentDataQueryType.QUICK_METRICS
            return CurrentDataQueryType.SYSTEM_OVERVIEW  # 默认

        # 中文提示词
        prompt = f"""
        请仔细分析以下用户针对“当前系统状态”的查询，并将其精确分类到预定义的子类型中。
        用户查询: "{user_query}"

        预定义查询子类型及其典型关键词或意图:
        - "system_overview": 询问整体系统状况、概览、今日总结。
        - "balance_check": 明确询问账户余额、总资金、可用资金等。
        - "user_status": 询问当前用户数量、活跃用户、新增用户等用户统计。
        - "today_expiry": 明确询问今日到期的产品数量或金额。
        - "cash_flow": 关注今日或当前的入金、出金、净流入等资金流动情况。
        - "product_status": 询问当前产品总体情况、在售产品数量等。
        - "quick_metrics": 要求快速获取一些关键性能指标的即时数据。
        - "unknown_current_query": 如果查询与当前数据相关但无法明确归入以上类型。

        请只返回最匹配的子类型ID（例如："balance_check"）。
        如果用户查询意图模糊或不属于以上任何一种，请返回 "unknown_current_query"。
        """
        prompt_hash = self._get_query_hash(f"identify_subtype_{user_query}")
        ai_result = await self._cached_ai_analysis(prompt_hash, prompt, ai_client_type="claude")

        if ai_result.get('success'):
            type_id_str = ai_result.get('response', '').strip().lower().replace('"', '').replace("'", "")
            try:
                return CurrentDataQueryType(type_id_str)
            except ValueError:
                logger.warning(f"AI返回了无法识别的当前数据查询子类型: '{type_id_str}'。将使用 UNKNOWN_CURRENT_QUERY。")
                return CurrentDataQueryType.UNKNOWN_CURRENT_QUERY
        else:
            logger.error(f"AI细化查询子类型失败: {ai_result.get('error')}")
            return CurrentDataQueryType.SYSTEM_OVERVIEW  # 出错时默认

    async def _process_by_query_type(self,
                                     query_sub_type: CurrentDataQueryType,
                                     user_query: str,
                                     current_data: Dict[str, Any],  # 这是从 Orchestrator 传入的已获取数据
                                     user_context: Optional[Dict[str, Any]] = None
                                     ) -> Dict[str, Any]:  # 返回包含指标和AI响应的字典
        """
        根据细化的查询子类型，使用AI从已有的`current_data`中提取信息、计算并格式化。
        """
        logger.info(f"根据子类型 '{query_sub_type.value}' 处理当前数据。")
        if not self.claude_client and not self.gpt_client:  # 至少需要一个AI
            logger.error("没有可用的AI客户端来处理当前数据。")
            return {"key_metrics": {"错误": "AI服务不可用"}, "response_format": "simple_text", "confidence": 0.1}

        # 选择一个AI客户端，优先Claude进行理解和组织，GPT进行精确提取（如果适用）
        # 对于当前数据处理，Claude的综合能力可能更适合直接生成所需信息
        primary_ai_client = self.claude_client if self.claude_client else self.gpt_client

        # 构建针对特定子类型的中文提示词
        # current_data 是 Orchestrator 通过 SmartDataFetcher 获取的，例如 /api/sta/system 的内容
        # 我们需要确保 current_data 的结构是已知的，或者提示词足够通用

        # 提取 current_data 中的关键部分用于提示词，避免过长
        data_summary_for_prompt = {
            "总余额": current_data.get("total_balance", current_data.get("总余额")),  # 尝试不同可能的键名
            "今日入金": current_data.get("daily_inflow", current_data.get("入金")),
            "今日出金": current_data.get("daily_outflow", current_data.get("出金")),
            "今日新增用户": current_data.get("new_registrations", current_data.get("注册人数")),
            "今日活跃用户": current_data.get("active_users_today", current_data.get("持仓人数")),  # 假设 '持仓人数' 代表活跃
            "今日到期金额": current_data.get("today_expiry_details", {}).get("到期金额",
                                                                             current_data.get("今日到期金额"))
        }
        # 移除值为None或0的项以简化提示
        data_summary_for_prompt = {k: v for k, v in data_summary_for_prompt.items() if v is not None and v != 0}

        prompt = f"""
        请根据以下提供的“当前系统数据快照”和用户的具体查询，提取或计算出用户所需的信息。

        用户的具体查询是：“{user_query}”
        系统已将此查询细化分类为：“{query_sub_type.value}”

        当前系统数据快照（部分关键指标）:
        ```json
        {json.dumps(data_summary_for_prompt, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```

        根据细化的查询类型 '{query_sub_type.value}'，请专注于回答用户的问题。
        例如：
        - 如果是 "balance_check"，请主要关注余额相关数据。
        - 如果是 "user_status"，请主要关注用户统计数据。
        - 如果是 "today_expiry"，请明确今日到期金额和数量（如果数据中有）。
        - 如果是 "system_overview" 或 "quick_metrics"，可以提供一个简要的关键指标汇总。

        您的任务是返回一个JSON对象，包含以下字段：
        - "key_metrics": 一个字典，包含与用户查询最相关的1-3个核心指标及其值。请确保数值准确。
                         例如: {{"总余额": 12345.67, "今日净流入": 1000.00}}
        - "related_metrics": (可选) 一个字典，包含用户可能感兴趣的1-2个相关指标。
        - "main_answer_text": (可选) 一句简短的、直接回答用户核心问题的中文文本。如果指标本身就是答案，则此项可省略。
        - "response_format_suggestion": 建议的响应格式，从 ["simple_text", "detailed_summary", "metrics_focused", "business_oriented"] 中选择一个。
        - "confidence": 您对本次提取和计算结果的置信度 (0.0 到 1.0 之间)。

        请确保所有数值都从提供的“当前系统数据快照”中获取或基于其计算。如果数据不足以回答，请在 key_metrics 中指明。
        例如，如果问今日到期产品数量，但数据快照中只有到期金额，则可以返回 {{"今日到期金额": XXX, "今日到期产品数量": "数据未提供"}}。
        """
        prompt_hash = self._get_query_hash(
            f"process_subtype_{query_sub_type.value}_{user_query}_{json.dumps(data_summary_for_prompt, sort_keys=True)}")
        ai_result = await self._cached_ai_analysis(prompt_hash, prompt, ai_client_type="claude")  # 优先 Claude

        processed_result_dict = {}
        if ai_result.get('success'):
            try:
                # AI应该返回JSON字符串，我们在此解析
                response_content = ai_result.get('response', '{}')
                processed_result_dict = json.loads(response_content)
                if not isinstance(processed_result_dict.get('key_metrics'), dict):  # 确保格式正确
                    processed_result_dict['key_metrics'] = {"解析错误": "AI未按预期返回key_metrics"}
                logger.debug(
                    f"AI successfully processed current data for subtype {query_sub_type.value}. Metrics: {list(processed_result_dict.get('key_metrics', {}).keys())}")
            except json.JSONDecodeError:
                logger.error(
                    f"无法解析AI为子类型 {query_sub_type.value} 返回的JSON: {ai_result.get('response', '')[:200]}")
                processed_result_dict['key_metrics'] = {"错误": "AI响应格式不正确"}
                processed_result_dict['confidence'] = 0.3
        else:
            logger.error(f"AI处理当前数据子类型 {query_sub_type.value} 失败: {ai_result.get('error')}")
            processed_result_dict['key_metrics'] = {"错误": f"AI服务调用失败: {ai_result.get('error')}"}
            processed_result_dict['confidence'] = 0.2

        # 确保默认值
        processed_result_dict.setdefault('key_metrics', {})
        processed_result_dict.setdefault('related_metrics', {})
        processed_result_dict.setdefault('response_format', ResponseFormat.SIMPLE_TEXT.value)  # 使用枚举值
        processed_result_dict.setdefault('confidence', 0.5)  # 如果AI未提供，给个默认

        return processed_result_dict

    async def _generate_quick_insights(self, query_sub_type: CurrentDataQueryType,
                                       key_metrics: Dict[str, Any],
                                       current_data: Dict[str, Any]) -> List[str]:  # 返回字符串洞察列表
        """
        基于关键指标和当前数据，使用AI生成1-2条简洁的中文业务洞察。
        """
        if not self.claude_client or not self.config.get('enable_insights_for_current_data', True):  # 假设配置项控制
            logger.debug("ClaudeClient不可用或当前数据洞察已禁用，跳过快速洞察生成。")
            return []

        if not key_metrics:  # 如果没有关键指标，则不生成洞察
            return []

        # 构建提示词
        prompt = f"""
        以下是针对用户关于“{query_sub_type.value}”类型查询所提取的关键指标和相关的当前系统数据快照。

        关键指标:
        ```json
        {json.dumps(key_metrics, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```

        部分当前系统数据快照:
        ```json
        {json.dumps({k: current_data.get(k) for k in list(current_data.keys())[:5]}, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)} 
        ```

        请基于以上信息，用中文生成1到2条最相关的、简洁的业务洞察或观察点。
        每条洞察应直接、精炼，不超过60个汉字。
        如果数据正常或无特别之处，可以指出“指标在预期范围内”或“暂无明显异常的快速洞察”。
        请以JSON字符串数组的形式返回，例如：["洞察点一：XXXX。", "洞察点二：YYYY。"]
        """
        prompt_hash = self._get_query_hash(
            f"quick_insights_{query_sub_type.value}_{json.dumps(key_metrics, sort_keys=True)}")
        ai_result = await self._cached_ai_analysis(prompt_hash, prompt, ai_client_type="claude")

        insights: List[str] = []
        if ai_result.get('success'):
            try:
                response_content = ai_result.get('response', '[]')
                parsed_insights = json.loads(response_content)
                if isinstance(parsed_insights, list) and all(isinstance(item, str) for item in parsed_insights):
                    insights = parsed_insights[:2]  # 最多取两条
                else:
                    logger.warning(f"AI快速洞察返回的不是字符串列表: {response_content[:200]}")
                    if isinstance(response_content, str) and response_content.strip():  # 如果返回的是单个字符串洞察
                        insights = [response_content.strip()]

            except json.JSONDecodeError:
                logger.error(f"无法解析AI快速洞察的JSON: {ai_result.get('response', '')[:200]}")
                if isinstance(ai_result.get('response'), str) and ai_result.get('response').strip():
                    insights = [ai_result.get('response').strip()]  # 尝试直接使用文本
        else:
            logger.error(f"AI生成快速洞察失败: {ai_result.get('error')}")

        logger.debug(f"Generated {len(insights)} quick insights.")
        return insights

    async def _format_current_data_response(self, query_sub_type: CurrentDataQueryType,
                                            user_query: str,
                                            processed_result_dict: Dict[str, Any],  # 来自 _process_by_query_type 的结果
                                            quick_insights: List[str]) -> Dict[
        str, Any]:  # 包含 main_answer, formatted_data, business_context
        """
        使用AI（优先GPT进行格式化）将处理结果和洞察整合成用户友好的响应。
        """
        logger.debug(f"Formatting response for query subtype: {query_sub_type.value}")
        # 优先使用GPT进行结构化输出和格式化，如果GPT不可用，则尝试Claude或基础模板
        formatting_ai_client = self.gpt_client if self.gpt_client else self.claude_client

        if not formatting_ai_client:
            logger.warning("没有可用的AI客户端进行响应格式化，使用基础格式化。")
            # 构建一个非常简单的 main_answer
            simple_main_answer = f"关于您的查询“{user_query[:30]}...”，以下是相关数据："
            metrics_str_parts = [f"{k}: {v}" for k, v in processed_result_dict.get('key_metrics', {}).items()]
            if metrics_str_parts:
                simple_main_answer += "\n关键指标:\n" + "\n".join(metrics_str_parts)
            if quick_insights:
                simple_main_answer += "\n初步观察:\n" + "\n".join(quick_insights)

            return {
                "main_answer": simple_main_answer,
                "formatted_data": processed_result_dict.get('key_metrics', {}),  # 直接用 key_metrics 作为格式化数据
                "business_context": "这是根据当前系统数据进行的快速查询。"
            }

        # 从 processed_result_dict 中提取AI建议的响应格式
        response_format_enum_val = processed_result_dict.get('response_format', ResponseFormat.SIMPLE_TEXT.value)
        try:
            response_format_for_prompt = ResponseFormat(response_format_enum_val).name  # 获取枚举的名称，如 'SIMPLE_TEXT'
        except ValueError:
            response_format_for_prompt = ResponseFormat.SIMPLE_TEXT.name  # 默认

        # 中文提示词
        prompt = f"""
        请将以下分析结果和洞察整合成一段通顺流畅、用户友好的中文回复。

        用户的原始查询：“{user_query}”
        系统识别的查询子类型：“{query_sub_type.value}”
        建议的响应格式偏好：“{response_format_for_prompt}”

        核心数据指标：
        ```json
        {json.dumps(processed_result_dict.get('key_metrics', {}), ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```

        相关的其他指标（可选）：
        ```json
        {json.dumps(processed_result_dict.get('related_metrics', {}), ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```

        系统生成的初步洞察/观察点：
        {chr(10).join([f"- {qi}" for qi in quick_insights]) if quick_insights else "无特别的初步观察。"}

        请根据以上信息，生成一个JSON对象，包含以下三个字段：
        1.  `"main_answer"`: (字符串) 对用户查询的直接、核心的回答。应简洁明了。如果指标本身就是答案，可以直接陈述。
        2.  `"formatted_data"`: (字典) 将最重要的1-3个关键指标及其值，以键值对形式呈现，值应为已格式化好的字符串（例如，货币带单位，百分比带%）。
                           例如：{{"今日总余额": "¥12,345,678.90", "较昨日变动率": "+1.25%"}}
        3.  `"business_context"`: (字符串, 可选) 对这些数据或回答提供1-2句简短的业务背景说明或解读，帮助用户理解。

        请严格遵循建议的响应格式偏好 '{response_format_for_prompt}' 来组织您的回答：
        - 'SIMPLE_TEXT': `main_answer` 简短直接，`formatted_data` 只包含最核心的1个指标。`business_context` 可省略。
        - 'DETAILED_SUMMARY': `main_answer` 可以稍长，包含更多细节。`formatted_data` 可包含2-3个核心指标。`business_context` 应提供。
        - 'METRICS_FOCUSED': `main_answer` 简洁，重点突出 `formatted_data` 中的多个指标。`business_context` 简要。
        - 'BUSINESS_ORIENTED': `main_answer` 和 `business_context` 强调业务含义和洞察的解读。

        如果数据不足或存在不确定性，请在 `main_answer` 或 `business_context` 中婉转表达。
        确保所有数值都已恰当格式化（例如，货币使用元，大数字使用千分位分隔符，百分比带%）。
        """
        prompt_hash = self._get_query_hash(
            f"format_resp_{query_sub_type.value}_{json.dumps(processed_result_dict.get('key_metrics', {}), sort_keys=True)}_{response_format_for_prompt}")
        ai_result = await self._cached_ai_analysis(prompt_hash, prompt,
                                                   ai_client_type="gpt" if self.gpt_client else "claude")  # 优先GPT进行格式化

        formatted_response_dict: Dict[str, Any] = {}
        if ai_result.get('success'):
            try:
                response_content = ai_result.get('response', '{}')
                formatted_response_dict = json.loads(response_content)
                if not isinstance(formatted_response_dict.get('main_answer'), str) or \
                        not isinstance(formatted_response_dict.get('formatted_data'), dict):
                    raise ValueError("AI返回的格式化响应结构不符合预期。")
                logger.debug(
                    f"AI successfully formatted response. Main answer: {formatted_response_dict.get('main_answer', '')[:50]}...")
            except json.JSONDecodeError:
                logger.error(f"无法解析AI格式化响应的JSON: {ai_result.get('response', '')[:200]}")
                formatted_response_dict['main_answer'] = "抱歉，系统在组织回复时遇到问题。" + \
                                                         (" 详细数据请参考指标部分。" if processed_result_dict.get(
                                                             'key_metrics') else "")
            except ValueError as ve:
                logger.error(f"AI格式化响应结构错误: {ve}")
                formatted_response_dict['main_answer'] = "AI未能按预期结构组织回复。"
        else:
            logger.error(f"AI格式化响应失败: {ai_result.get('error')}")
            formatted_response_dict['main_answer'] = "AI服务在格式化最终答复时出现故障。"

        # 确保默认值
        formatted_response_dict.setdefault('main_answer', "已获取相关数据，请查看指标详情。")
        formatted_response_dict.setdefault('formatted_data',
                                           processed_result_dict.get('key_metrics', {}))  # 默认用原始key_metrics
        formatted_response_dict.setdefault('business_context', "这是基于当前最新数据的快速查询结果。")

        return formatted_response_dict

    async def process_current_data_query(
            self,
            user_query: str,
            query_analysis: QueryAnalysisResult,  # 从编排器接收
            current_data: Dict[str, Any],  # 从编排器接收 (已获取的 /api/sta/system 数据)
            user_context: Optional[Dict[str, Any]] = None
    ) -> CurrentDataResponse:
        """
        处理关于当前系统状态的查询。
        数据已由编排器获取并传入。
        """
        method_start_time = datetime.now()
        logger.info(f"CurrentDataProcessor: Processing query '{user_query[:50]}...' with pre-fetched current_data.")

        # 更新内部统计 (total_queries_processed)
        self.processing_stats['total_queries_processed'] = self.processing_stats.get('total_queries_processed', 0) + 1

        try:
            if not current_data:  # 检查传入的数据是否有效
                logger.warning("CurrentDataProcessor: Received empty or None current_data. Cannot proceed effectively.")
                return self._create_error_response(user_query, "无法获取必要的当前系统数据来回答您的问题。")

            # 1. 进一步细化查询子类型 (基于 user_query 和 query_analysis)
            #    _determine_current_data_sub_type 内部使用AI，并基于 user_query
            #    它可以选择性地使用 query_analysis 来辅助判断
            current_query_sub_type: CurrentDataQueryType = await self._determine_current_data_sub_type(user_query,
                                                                                                       query_analysis)
            logger.info(f"Determined current data query sub-type: {current_query_sub_type.value}")

            # 2. 根据细化的子类型，从传入的 current_data 中提取/计算指标
            #    _process_by_query_type 现在使用传入的 current_data
            processed_result_dict: Dict[str, Any] = await self._process_by_query_type(
                query_sub_type=current_query_sub_type,
                user_query=user_query,
                current_data=current_data,  # 使用传入的数据
                user_context=user_context
            )
            logger.info(
                f"Data processed by sub-type. Key metrics found: {list(processed_result_dict.get('key_metrics', {}).keys())}")

            # 3. 生成快速洞察 (基于已提取的指标和传入的 current_data)
            quick_insights: List[str] = await self._generate_quick_insights(
                query_sub_type=current_query_sub_type,
                key_metrics=processed_result_dict.get('key_metrics', {}),
                current_data=current_data  # 使用传入的数据
            )
            logger.info(f"Generated {len(quick_insights)} quick insights.")

            # 4. 格式化最终响应 (基于处理结果和洞察)
            formatted_response_dict: Dict[str, Any] = await self._format_current_data_response(
                query_sub_type=current_query_sub_type,
                user_query=user_query,
                processed_result_dict=processed_result_dict,
                quick_insights=quick_insights
            )
            logger.info("Response formatted.")

            # 5. 构建并返回 CurrentDataResponse 对象
            processing_time_seconds = (datetime.now() - method_start_time).total_seconds()

            # 从AI处理结果中获取建议的响应格式，或使用默认
            response_format_str = processed_result_dict.get('response_format', ResponseFormat.SIMPLE_TEXT.value)
            try:
                final_response_format = ResponseFormat(response_format_str)
            except ValueError:
                logger.warning(
                    f"Invalid response_format string '{response_format_str}' from AI. Defaulting to SIMPLE_TEXT.")
                final_response_format = ResponseFormat.SIMPLE_TEXT

            response = CurrentDataResponse(
                query_type=current_query_sub_type,
                response_format=final_response_format,
                main_answer=formatted_response_dict.get('main_answer', "未能生成主要回答。"),
                key_metrics=processed_result_dict.get('key_metrics', {}),
                formatted_data=formatted_response_dict.get('formatted_data', {}),  # 确保这是 Dict[str, str]
                business_context=formatted_response_dict.get('business_context', ""),
                quick_insights=quick_insights,
                related_metrics=processed_result_dict.get('related_metrics', {}),
                data_timestamp=current_data.get('timestamp', datetime.now().isoformat()),  # 使用传入数据的timestamp
                response_confidence=float(processed_result_dict.get('confidence', 0.75)),  # AI给出的置信度
                data_sources_references=current_data.get('api_source_names', ["pre_fetched_system_data"]),  # 指明数据来源
                processing_time=processing_time_seconds,
                metadata={  # 可以包含此处理器内部的AI耗时等
                    'ai_processing_time': processed_result_dict.get('metadata', {}).get('ai_processing_time', 0.0)
                    # 假设_process_by_query_type的meta包含
                }
            )

            self._update_processing_stats(current_query_sub_type, processing_time_seconds,
                                          response.response_confidence)  # 假设此方法已实现
            logger.info(f"✅ CurrentDataProcessor processed query successfully in {processing_time_seconds:.3f}s.")
            return response

        except Exception as e:
            logger.error(
                f"❌ Error in CurrentDataProcessor.process_current_data_query for query '{user_query[:50]}...': {str(e)}\n{traceback.format_exc()}")
            # 调用内部的错误响应构建方法
            return self._create_error_response(user_query, f"处理当前数据查询时发生内部错误: {str(e)}")
    # 移除了 process_multiple_queries 和 _process_query_with_data，
    # 因为批量处理的复杂性最好放在 Orchestrator 层面或专门的 BatchProcessor 中。
    # CurrentDataProcessor 应专注于单个查询的处理。

    def _update_processing_stats(self, query_type: CurrentDataQueryType,
                                 processing_time: float, confidence: float) -> None:
        """更新本处理器的内部统计信息。"""
        try:
            type_value = query_type.value
            self.processing_stats['queries_by_sub_type'][type_value] = \
                self.processing_stats['queries_by_sub_type'].get(type_value, 0) + 1

            total_processed = self.processing_stats['total_queries_processed']  # 在 process_current_data_query 中增加
            if total_processed == 0: return  # 避免除零

            current_avg_time = self.processing_stats['avg_response_time']
            self.processing_stats['avg_response_time'] = \
                (current_avg_time * (total_processed - 1) + processing_time) / total_processed

            current_avg_confidence = self.processing_stats['avg_confidence']
            self.processing_stats['avg_confidence'] = \
                (current_avg_confidence * (total_processed - 1) + confidence) / total_processed

            total_ai_cache_lookups = self.processing_stats['ai_cache_hits'] + self.processing_stats['ai_cache_misses']
            if total_ai_cache_lookups > 0:
                self.processing_stats['ai_cache_hit_rate'] = self.processing_stats[
                                                                 'ai_cache_hits'] / total_ai_cache_lookups

        except Exception as e:
            logger.error(f"更新CurrentDataProcessor统计信息失败: {str(e)}")

    def _create_error_response(self, user_query: str, error_message: str) -> CurrentDataResponse:
        """创建一个标准的错误响应对象。"""
        logger.error(f"Creating error response for query '{user_query[:50]}...': {error_message}")
        return CurrentDataResponse(
            query_type=CurrentDataQueryType.UNKNOWN_CURRENT_QUERY,  # 或一个特定的ERROR类型
            response_format=ResponseFormat.SIMPLE_TEXT,
            main_answer=f"处理您的即时查询“{user_query[:30]}...”时遇到错误: {error_message}",
            key_metrics={"错误详情": error_message},
            formatted_data={"错误": error_message},
            business_context="系统无法完成您的当前数据请求。",
            quick_insights=["建议检查查询或稍后重试。"],
            related_metrics={},
            data_timestamp=datetime.now().isoformat(),
            response_confidence=0.0,  # 错误发生，置信度为0
            data_sources_references=["N/A"],
            processing_time=0.01,  # 象征性的处理时间
            metadata={"error_flag": True, "error_message": error_message}
        )

    def _load_processing_config(self) -> Dict[str, Any]:
        """加载当前数据处理器的特定配置。"""
        return {
            "max_response_time_ms": 3000,  # 本处理器自身的目标响应时间
            "data_freshness_threshold_sec": 120,  # 对传入数据的“新鲜度”要求
            "enable_insights_for_current_data": True,  # 是否为此处理器生成快速洞察
            "max_quick_insights_count": 2,
            "ai_cache_ttl_sec": 600  # AI分析结果的缓存时间 (10分钟)
        }

    # _load_current_data_patterns 可能不再需要，因为类型识别更多依赖AI或传入的QueryAnalysisResult
    # def _load_current_data_patterns(self) -> Dict[str, List[str]]: ...

    # 外部接口，用于获取本处理器的统计信息
    def get_processor_stats(self) -> Dict[str, Any]:
        return self.processing_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """执行本处理器的健康检查。"""
        # 检查AI客户端是否可用
        claude_ok = self.claude_client is not None  # and await self.claude_client.is_healthy() # 假设AI客户端有健康检查
        gpt_ok = self.gpt_client is not None  # and await self.gpt_client.is_healthy()
        status = "healthy"
        issues = []
        if not claude_ok: issues.append("ClaudeClient not available/healthy.")
        if not gpt_ok: issues.append("GPTClient not available/healthy.")

        if issues:
            status = "degraded" if (claude_ok or gpt_ok) else "unhealthy"

        return {
            "status": status,
            "component_name": self.__class__.__name__,
            "dependencies_status": {
                "claude_client": "ok" if claude_ok else "error",
                "gpt_client": "ok" if gpt_ok else "error",
            },
            "internal_stats": self.get_processor_stats(),
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }