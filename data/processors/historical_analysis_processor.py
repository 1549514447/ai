# data/processors/historical_analysis_processor.py
"""
📈 AI驱动的历史分析处理器 (优化版)
专门处理历史数据分析查询，基于已获取的数据进行深度分析。
"""

import logging
import statistics
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union  # Union 需要导入
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field  # 导入 field
from enum import Enum
import json
import hashlib
from functools import lru_cache  # 保留用于AI分析结果缓存

from core.analyzers.financial_data_analyzer import FinancialDataAnalyzer
# 导入所需类型
from core.analyzers.query_parser import (
    QueryAnalysisResult,
    QueryType as QueryParserQueryType,        # 导入 QueryType 并别名为 QueryParserQueryType
    QueryComplexity as QueryParserComplexity  # 导入 QueryComplexity 并别名为 QueryParserComplexity

)

from core.data_orchestration.smart_data_fetcher import (
    SmartDataFetcher, create_smart_data_fetcher,
    ExecutionResult as FetcherExecutionResult, # <--- 修改这里：导入 ExecutionResult 并别名为 FetcherExecutionResult
    DataQualityLevel as FetcherDataQualityLevel,
    ExecutionStatus as FetcherExecutionStatus # 确保这个也已正确导入和别名
)
from core.models.claude_client import ClaudeClient, CustomJSONEncoder  # AI 客户端
from core.models.openai_client import OpenAIClient  # AI 客户端
from data.connectors.api_connector import APIConnector
from utils.calculators.financial_calculator import FinancialCalculator
from utils.helpers.date_utils import DateUtils, DateParseResult

# 这些工具类如果本处理器需要直接使用，则应由Orchestrator注入
# from utils.helpers.date_utils import DateUtils
# from utils.calculators.financial_calculator import FinancialCalculator
# from core.analyzers.financial_data_analyzer import FinancialDataAnalyzer # 通常不直接依赖，而是接收其分析结果

logger = logging.getLogger(__name__)


class HistoricalQueryType(Enum):
    TREND_ANALYSIS = "trend_analysis"
    GROWTH_ANALYSIS = "growth_analysis"
    COMPARISON_ANALYSIS = "comparison_analysis"
    PATTERN_ANALYSIS = "pattern_analysis"
    PERIOD_SUMMARY = "period_summary"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    UNKNOWN_HISTORICAL = "unknown_historical"  # 新增用于无法分类的情况


class AnalysisDepth(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"


@dataclass
class HistoricalAnalysisResponse:
    query_type: HistoricalQueryType
    analysis_depth: AnalysisDepth
    main_findings: List[str] = field(default_factory=list)
    trend_summary: Dict[str, Any] = field(default_factory=dict)
    key_metrics: Dict[str, Any] = field(default_factory=dict)  # 值可以是数值或格式化字符串
    business_insights: List[str] = field(default_factory=list)
    pattern_discoveries: List[str] = field(default_factory=list)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    data_completeness: float = 0.0
    analysis_confidence: float = 0.0
    data_sources_used: List[str] = field(default_factory=list)  # 来源于 SmartDataFetcher 结果
    analysis_period: str = ""
    processing_time: float = 0.0  # 本处理器耗时
    methodology_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 存储AI耗时等


class HistoricalAnalysisProcessor:
    """
    📈 AI驱动的历史分析处理器 (优化版)
    专注于基于已获取的真实历史数据进行深度分析，提供专业的趋势洞察和业务建议。
    """

    def __init__(self, claude_client: Optional[ClaudeClient] = None,
                 gpt_client: Optional[OpenAIClient] = None):
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.analysis_config = self._load_analysis_config()
        # self.historical_patterns = self._load_historical_patterns() # 可选，用于辅助
        self.processing_stats = self._default_processing_stats()

        # 依赖注入的组件，由 Orchestrator 设置
        self.api_connector: Optional[APIConnector] = None  # 通常不直接用，数据来自Fetcher
        self.financial_data_analyzer: Optional[FinancialDataAnalyzer] = None  # 如果需要调用更底层的分析
        self.financial_calculator: Optional[FinancialCalculator] = None  # 如果需要直接进行特定计算
        self.date_utils: Optional[DateUtils] = None  # 如果需要日期处理

        logger.info("HistoricalAnalysisProcessor (Optimized) initialized for AI-driven historical analysis")

    def _default_processing_stats(self) -> Dict[str, Any]:
        return {
            'total_analyses': 0, 'analyses_by_type': {},
            'avg_processing_time': 0.0, 'avg_confidence': 0.0,
            'ai_cache_hits': 0, 'ai_cache_misses': 0, 'ai_cache_hit_rate': 0.0
        }

    def _load_analysis_config(self) -> Dict[str, Any]:
        return {
            'default_analysis_period_days': 30,
            'max_analysis_period_days': 365,
            'min_data_points_for_trend': 7,
            'claude_max_retries': 2,  # 与Orchestrator的配置保持一致或独立
            'gpt_max_retries': 2,
            'cache_ttl_seconds': 1800,  # 30分钟
            'max_cache_size': 50,  # 调整缓存大小
            # 可以定义不同分析深度所需的数据片段键名 (与SmartDataFetcher约定)
            'data_keys_for_depth': {
                AnalysisDepth.BASIC: ['daily_summary_short_term'],
                AnalysisDepth.STANDARD: ['daily_summary_medium_term', 'key_product_trends_short_term'],
                AnalysisDepth.COMPREHENSIVE: ['daily_summary_long_term', 'all_product_trends_medium_term',
                                              'user_segment_summary_medium_term'],
                AnalysisDepth.EXPERT: ['all_historical_daily', 'all_historical_product', 'all_historical_user',
                                       'all_historical_expiry']
            }
        }

    def _get_query_hash(self, text_to_hash: str) -> str:
        return hashlib.md5(text_to_hash.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=50)
    async def _cached_ai_analysis(self, query_or_prompt_hash: str, prompt: str,
                                  ai_client_type: str = "claude") -> Dict[str, Any]:
        ai_call_start_time = time.time()
        client_to_use: Union[ClaudeClient, OpenAIClient, None] = None
        model_name_for_log = "Unknown AI"
        response_data: Dict[str, Any] = {"success": False, "error": "AI client not configured or type unsupported."}

        if ai_client_type.lower() == "claude" and self.claude_client:
            client_to_use = self.claude_client
            model_name_for_log = "Claude"
        elif ai_client_type.lower() == "gpt" and self.gpt_client:
            client_to_use = self.gpt_client
            model_name_for_log = "GPT"

        if not client_to_use:
            logger.error(f"AI客户端 '{ai_client_type}' 不可用。")
            return response_data

        # 统计缓存未命中 (只有在实际调用AI时才算未命中)
        self.processing_stats['ai_cache_misses'] = self.processing_stats.get('ai_cache_misses', 0) + 1
        logger.debug(f"AI cache MISS for hash: {query_or_prompt_hash}. Calling {model_name_for_log}.")

        try:
            ai_raw_response = None
            if isinstance(client_to_use, ClaudeClient):
                if hasattr(client_to_use, 'messages') and callable(getattr(client_to_use.messages, 'create', None)):
                    ai_raw_response = await asyncio.to_thread(
                        client_to_use.messages.create,
                        model=getattr(client_to_use, 'model', "claude-3-sonnet-20240229"),
                        max_tokens=2048, messages=[{"role": "user", "content": prompt}]
                    )
                    content_text = ""
                    if hasattr(ai_raw_response, 'content') and ai_raw_response.content:
                        for item in ai_raw_response.content:
                            if hasattr(item, 'text'): content_text += item.text
                    response_data = {"success": True, "response": content_text}
                # Add other Claude call methods if needed, e.g., older SDKs
                else:
                    response_data = {"success": False, "error": "ClaudeClient method not found."}

            elif isinstance(client_to_use, OpenAIClient):
                if hasattr(client_to_use, 'chat') and hasattr(client_to_use.chat, 'completions') and callable(
                        getattr(client_to_use.chat.completions, 'create', None)):
                    ai_raw_response = await asyncio.to_thread(
                        client_to_use.chat.completions.create,
                        model=getattr(client_to_use, 'model', "gpt-4o"),
                        messages=[{"role": "user", "content": prompt}], max_tokens=2048
                    )
                    if ai_raw_response.choices and ai_raw_response.choices[0].message:
                        response_data = {"success": True, "response": ai_raw_response.choices[0].message.content}
                elif hasattr(client_to_use, 'process_direct_query'):  # Your custom method
                    response_data = await client_to_use.process_direct_query(user_query=prompt, data={})
                else:
                    response_data = {"success": False, "error": "OpenAIClient method not found."}

            response_data["model_used"] = model_name_for_log
            response_data["ai_call_duration"] = time.time() - ai_call_start_time
            return response_data

        except Exception as e:
            logger.error(f"AI分析调用 ({model_name_for_log}) 时发生错误: {str(e)}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e), "response": f"AI分析出错: {str(e)}",
                    "ai_call_duration": time.time() - ai_call_start_time}

    async def process_historical_analysis_query(self, user_query: str,
                                                user_context: Optional[Dict[str, Any]] = None
                                                ) -> HistoricalAnalysisResponse:
        """
        🎯 处理历史分析查询的主入口 (优化版)
        数据从 user_context['data_acquisition_result'] 获取。
        """
        method_start_time = datetime.now()
        logger.info(f"📈 开始历史分析查询: '{user_query[:100]}...'")
        self.processing_stats['total_analyses'] = self.processing_stats.get('total_analyses', 0) + 1

        internal_ai_processing_time = 0.0  # 累加本处理器内部的AI耗时

        # 1. 从 user_context 中提取 QueryAnalysisResult 和 FetcherExecutionResult
        if not user_context or not isinstance(user_context, dict):
            logger.error("HistoricalAnalysisProcessor: user_context 未提供或格式不正确。")
            return self._create_error_response(user_query, "内部错误：缺少必要的处理上下文。")

        query_analysis: Optional[QueryAnalysisResult] = user_context.get('query_analysis')
        fetcher_result: Optional[FetcherExecutionResult] = user_context.get('data_acquisition_result')

        if not query_analysis or not isinstance(query_analysis, QueryAnalysisResult):
            logger.error("HistoricalAnalysisProcessor: 'query_analysis' 未在 user_context 中提供或类型不正确。")
            return self._create_error_response(user_query, "历史分析失败：缺少查询解析结果。")

        if not fetcher_result or not isinstance(fetcher_result, FetcherExecutionResult):
            logger.error("HistoricalAnalysisProcessor: 'data_acquisition_result' 未在 user_context 中提供或类型不正确。")
            return self._create_error_response(user_query, "历史分析失败：缺少必要的数据获取结果。")

        try:
            # Step 2: (AI识别查询类型和分析深度) - 现在从 query_analysis 中获取，或由本处理器细化
            # 假设 query_analysis.query_type 可以映射到 HistoricalQueryType
            # 并且 query_analysis.analysis_depth (如果存在) 或偏好可以映射到 AnalysisDepth
            # 为简化，我们假设Orchestrator已做好初步分类，这里可以进一步细化或直接使用

            # 假设调用AI进行细化分类，如果 Orchestrator 的分类不够具体
            ai_call_start = time.time()
            determined_query_type, determined_analysis_depth = await self._determine_historical_query_details(
                user_query, query_analysis)
            internal_ai_processing_time += (time.time() - ai_call_start)
            logger.info(f"历史查询细化类型: {determined_query_type.value}, 分析深度: {determined_analysis_depth.value}")

            # Step 3: (AI提取时间范围和分析参数) - 现在从 query_analysis.time_requirements 获取
            # 或调用AI/DateUtils进行更精确的提取（如果需要）
            ai_call_start = time.time()
            time_params = await self._extract_time_params_from_query_analysis(user_query, query_analysis)
            internal_ai_processing_time += (time.time() - ai_call_start)
            logger.info(f"提取并确认时间参数: 从 {time_params.get('start_date')} 到 {time_params.get('end_date')}")

            # Step 4: 从传入的 fetcher_result 准备历史数据
            prep_data_start = time.time()
            historical_data_payload = self._prepare_historical_data_from_context(
                time_params, determined_analysis_depth, fetcher_result
            )
            logger.info(
                f"历史数据准备完成 ({time.time() - prep_data_start:.2f}s). 数据质量评估: {historical_data_payload.get('data_quality_assessment', {}).get('overall_quality', 'N/A')}")

            if historical_data_payload.get('error'):
                logger.error(f"历史数据准备失败: {historical_data_payload.get('error')}")
                return self._create_error_response(user_query, f"数据准备错误: {historical_data_payload.get('error')}")

            # Step 5: AI驱动的历史数据分析 (核心分析逻辑)
            analysis_main_start = time.time()
            analysis_results_dict = await self._ai_analyze_historical_data(
                historical_data_payload, determined_query_type, determined_analysis_depth, user_query
            )
            internal_ai_processing_time += analysis_results_dict.get('metadata', {}).get(
                'ai_processing_time_for_analysis_step', 0.0)  # 从子方法获取AI耗时
            logger.info(f"AI历史数据分析完成 ({time.time() - analysis_main_start:.2f}s).")

            # Step 6: AI生成深度业务洞察
            insights_gen_start = time.time()
            business_insights_dict = await self._ai_generate_business_insights(
                analysis_results_dict, determined_query_type, user_query
            )
            internal_ai_processing_time += business_insights_dict.get('metadata', {}).get(
                'ai_processing_time_for_insight_step', 0.0)
            logger.info(f"AI业务洞察生成完成 ({time.time() - insights_gen_start:.2f}s).")

            # Step 7: 构建最终响应
            processing_time_wall = (datetime.now() - method_start_time).total_seconds()
            response = self._build_historical_analysis_response(
                determined_query_type, determined_analysis_depth,
                analysis_results_dict, business_insights_dict,
                historical_data_payload, time_params, processing_time_wall
            )
            response.metadata['ai_processing_time'] = internal_ai_processing_time  # 记录本处理器内部总AI耗时

            self._update_processing_stats(determined_query_type, processing_time_wall, response.analysis_confidence)
            logger.info(
                f"✅ 历史分析成功完成，总耗时{processing_time_wall:.2f}s, 内部AI耗时: {internal_ai_processing_time:.2f}s")
            return response

        except Exception as e:
            processing_time_wall_on_error = (datetime.now() - method_start_time).total_seconds()
            logger.error(f"❌ 历史分析处理主流程失败: {str(e)}\n{traceback.format_exc()}")
            err_resp = self._create_error_response(user_query, f"历史分析内部错误: {str(e)}")
            err_resp.processing_time = processing_time_wall_on_error  # 记录实际耗时
            err_resp.metadata['ai_processing_time'] = internal_ai_processing_time  # 记录出错前的AI耗时
            return err_resp

    async def _determine_historical_query_details(self, user_query: str, query_analysis: QueryAnalysisResult) -> Tuple[
        HistoricalQueryType, AnalysisDepth]:
        """
        根据 QueryAnalysisResult 或通过 AI 进一步细化历史查询的类型和分析深度。
        """
        # 优先使用 QueryAnalysisResult 中的信息进行映射
        parser_qt: QueryParserQueryType = query_analysis.query_type
        parser_qc: QueryParserComplexity = query_analysis.complexity

        # 明确 QueryParserQueryType 到 HistoricalQueryType 的映射规则
        qt_map: Dict[QueryParserQueryType, HistoricalQueryType] = {
            QueryParserQueryType.TREND_ANALYSIS: HistoricalQueryType.TREND_ANALYSIS,
            QueryParserQueryType.GROWTH_ANALYSIS: HistoricalQueryType.GROWTH_ANALYSIS,  # 假设 QueryParser 有此类型
            QueryParserQueryType.COMPARISON: HistoricalQueryType.COMPARISON_ANALYSIS,
            QueryParserQueryType.PREDICTION: HistoricalQueryType.TREND_ANALYSIS,  # 预测通常基于趋势
            QueryParserQueryType.SCENARIO_SIMULATION: HistoricalQueryType.PATTERN_ANALYSIS,  # 场景分析可能涉及模式识别
            QueryParserQueryType.RISK_ASSESSMENT: HistoricalQueryType.VOLATILITY_ANALYSIS,  # 风险评估可能关注波动性
            QueryParserQueryType.DATA_RETRIEVAL: HistoricalQueryType.PERIOD_SUMMARY,  # 获取历史数据可能视为期间总结
            QueryParserQueryType.CALCULATION: HistoricalQueryType.PERIOD_SUMMARY,  # 历史计算也可能是一种总结
            QueryParserQueryType.DEFINITION_EXPLANATION: HistoricalQueryType.UNKNOWN_HISTORICAL,  # 历史分析不处理定义
            QueryParserQueryType.OPTIMIZATION: HistoricalQueryType.UNKNOWN_HISTORICAL,  # 历史分析不直接做优化建议
            QueryParserQueryType.GENERAL_KNOWLEDGE: HistoricalQueryType.UNKNOWN_HISTORICAL,  # 通用知识通常不是历史分析范畴
            QueryParserQueryType.SYSTEM_COMMAND: HistoricalQueryType.UNKNOWN_HISTORICAL,
            QueryParserQueryType.UNKNOWN: HistoricalQueryType.UNKNOWN_HISTORICAL,
        }
        # 如果 QueryParserQueryType 中有您在 Orchestrator 中假设的 GENERAL_KNOWLEDGE，也需要在这里映射
        if hasattr(QueryParserQueryType, 'GENERAL_KNOWLEDGE'):
            qt_map[QueryParserQueryType.GENERAL_KNOWLEDGE] = HistoricalQueryType.UNKNOWN_HISTORICAL

        determined_query_type = qt_map.get(parser_qt, HistoricalQueryType.TREND_ANALYSIS)  # 默认趋势分析

        # 明确 QueryParserComplexity 到 AnalysisDepth 的映射规则
        depth_map: Dict[QueryParserComplexity, AnalysisDepth] = {
            QueryParserComplexity.SIMPLE: AnalysisDepth.BASIC,
            QueryParserComplexity.MEDIUM: AnalysisDepth.STANDARD,
            QueryParserComplexity.COMPLEX: AnalysisDepth.COMPREHENSIVE,
            QueryParserComplexity.EXPERT: AnalysisDepth.EXPERT,
        }
        determined_analysis_depth = depth_map.get(parser_qc, AnalysisDepth.STANDARD)  # 默认标准深度

        logger.debug(
            f"初步确定类型: {determined_query_type.value}, 深度: {determined_analysis_depth.value} (基于QueryParser结果)")

        # (后续的AI细化逻辑保持不变，但应使用上面确定的 determined_query_type 和 determined_analysis_depth 作为AI的输入或参考)
        # ...
        if self.claude_client:  # AI 细化逻辑
            # ... (构建中文提示词，让AI从HistoricalQueryType和AnalysisDepth的选项中选择)
            # 提示词应包含 HistoricalQueryType 和 AnalysisDepth 的所有枚举值作为选项
            historical_type_options = [ht.value for ht in HistoricalQueryType if
                                       ht != HistoricalQueryType.UNKNOWN_HISTORICAL]
            analysis_depth_options = [ad.value for ad in AnalysisDepth]

            prompt = f"""
            请基于用户原始查询和系统初步的查询分析结果，进一步精确判断历史数据分析的具体“查询子类型”和“分析深度”。

            用户原始查询: "{user_query}"
            系统初步分析:
            - 主要意图类型: {query_analysis.query_type.value}
            - 复杂度评估: {query_analysis.complexity.value}
            - 初步建议的历史查询子类型: {determined_query_type.value}
            - 初步建议的分析深度: {determined_analysis_depth.value}

            “历史查询子类型”选项 (请严格从此列表选择最相关的一个，仅返回其英文ID):
            {json.dumps(historical_type_options, ensure_ascii=False)}

            “分析深度”选项 (请严格从此列表选择最合适的一个，仅返回其英文ID):
            {json.dumps(analysis_depth_options, ensure_ascii=False)}

            返回一个JSON对象，包含 "query_type" (历史查询子类型) 和 "analysis_depth" 两个键。
            例如: {{"query_type": "growth_analysis", "analysis_depth": "comprehensive"}}
            如果初步建议已经很准确，可以直接采纳。如果需要调整，请给出调整后的选择。
            """
            query_hash = self._get_query_hash(
                f"hist_subtype_depth_refined_{user_query}_{query_analysis.query_type.value}")
            ai_result = await self._cached_ai_analysis(query_hash, prompt, ai_client_type="claude")

            if ai_result.get('success'):
                try:
                    analysis = json.loads(ai_result.get('response', '{}'))
                    qt_str = analysis.get('query_type')
                    ad_str = analysis.get('analysis_depth')
                    if qt_str: determined_query_type = HistoricalQueryType(qt_str)  # 使用AI的判断覆盖
                    if ad_str: determined_analysis_depth = AnalysisDepth(ad_str)  # 使用AI的判断覆盖
                    logger.info(
                        f"AI细化后的类型: {determined_query_type.value}, 深度: {determined_analysis_depth.value}")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"AI细化历史查询类型失败(解析/ValueError): {e}. AI原始返回: {ai_result.get('response', '')[:200]}. 保持规则判断结果。")
            else:
                logger.warning(f"AI细化调用失败: {ai_result.get('error')}. 保持规则判断结果。")

        return determined_query_type, determined_analysis_depth

        # data/processors/historical_analysis_processor.py

    async def _extract_time_params_from_query_analysis(self, user_query: str,
                                                       query_analysis: QueryAnalysisResult) -> Dict[str, Any]:
        logger.debug(f"Extracting time parameters from QueryAnalysisResult for query: '{user_query[:50]}...'")

        time_req = query_analysis.time_requirements if query_analysis else {}  # query_analysis 可能为 None

        final_time_params: Dict[str, Any] = {
            'has_explicit_time': False,
            'start_date': None, 'end_date': None, 'time_range_days': None,
            'relative_time': None, 'analysis_granularity': 'daily'
        }

        # 优先使用 QueryParser 解析出的时间信息
        if time_req and isinstance(time_req, dict):
            if time_req.get('start_date') and time_req.get('end_date'):
                final_time_params['start_date'] = time_req['start_date']
                final_time_params['end_date'] = time_req['end_date']
                final_time_params['has_explicit_time'] = True
                if time_req.get('time_range_days'):
                    try:
                        final_time_params['time_range_days'] = int(time_req['time_range_days'])
                    except:
                        pass
                else:
                    try:
                        sd = datetime.strptime(time_req['start_date'], '%Y-%m-%d')
                        ed = datetime.strptime(time_req['end_date'], '%Y-%m-%d')
                        final_time_params['time_range_days'] = (ed - sd).days + 1
                    except Exception as e_calc_days:
                        logger.warning(f"Could not calculate time_range_days from QueryParser dates: {e_calc_days}")

            if time_req.get('relative_period'):
                final_time_params['relative_time'] = time_req['relative_period']
                final_time_params['has_explicit_time'] = True
            if time_req.get('granularity'):
                final_time_params['analysis_granularity'] = time_req['granularity']

        # 如果 QueryParser 未能提供完整的时间范围，则调用 DateUtils 进行二次解析
        if not final_time_params.get('start_date') or not final_time_params.get('end_date'):
            logger.info(
                f"QueryParser did not yield full time range. Attempting DateUtils.parse_dates_from_query for: '{user_query[:50]}...'")
            if not self.date_utils:
                logger.error(
                    "DateUtils not initialized in HistoricalAnalysisProcessor. Cannot perform detailed date parsing.")
            else:
                try:
                    # ++++++++++++++ 修改此处调用 ++++++++++++++
                    date_util_parse_result: DateParseResult = await self.date_utils.parse_dates_from_query(
                        query=user_query)
                    # +++++++++++++++++++++++++++++++++++++++++++

                    if date_util_parse_result and date_util_parse_result.ranges:
                        # DateParseResult.ranges 是 List[DateRange]
                        # 我们取第一个最相关的范围（或者需要更复杂的逻辑来选择）
                        # 假设 DateUtils 的 AI 解析会把最主要的范围放在前面或只返回一个主要范围
                        main_range = date_util_parse_result.ranges[0]
                        if main_range.start_date and main_range.end_date:
                            final_time_params['start_date'] = main_range.start_date
                            final_time_params['end_date'] = main_range.end_date
                            final_time_params['has_explicit_time'] = True  # 标记为已解析出

                            # 从 DateRange 对象计算 time_range_days
                            try:
                                sd_du = datetime.strptime(main_range.start_date, '%Y-%m-%d')
                                ed_du = datetime.strptime(main_range.end_date, '%Y-%m-%d')
                                final_time_params['time_range_days'] = (ed_du - sd_du).days + 1
                            except Exception as e_calc_days_du:
                                logger.warning(
                                    f"Could not calculate time_range_days from DateUtils range: {e_calc_days_du}")
                                # 如果 QueryParser 之前有天数，尝试使用
                                if time_req and time_req.get('time_range_days'):
                                    final_time_params['time_range_days'] = int(time_req['time_range_days'])

                            # 尝试获取原始表述和粒度 (如果 DateUtils 能提供)
                            # DateParseResult.ranges[DateRange] 没有 original_expression 和 granularity 字段
                            # 但 DateParseResult.relative_terms 可能包含一些信息
                            if date_util_parse_result.relative_terms:
                                final_time_params['relative_time'] = ", ".join(
                                    date_util_parse_result.relative_terms)
                            # 粒度可能需要 DateUtils 更明确地返回，或在这里基于范围长度推断
                            # final_time_params['analysis_granularity'] = ...

                            logger.info(
                                f"DateUtils successfully parsed time range: {final_time_params['start_date']} to {final_time_params['end_date']}")
                        else:
                            logger.warning(
                                f"DateUtils.parse_dates_from_query returned a range with missing start/end dates: {main_range}")
                    elif date_util_parse_result and date_util_parse_result.dates:  # 如果没有范围但有具体日期
                        # 如果只有一个日期，则开始和结束是同一天
                        if len(date_util_parse_result.dates) == 1:
                            final_time_params['start_date'] = date_util_parse_result.dates[0]
                            final_time_params['end_date'] = date_util_parse_result.dates[0]
                            final_time_params['time_range_days'] = 1
                            final_time_params['has_explicit_time'] = True
                        # 如果有多个日期，可以取最早和最晚的构成范围 (需要排序)
                        elif len(date_util_parse_result.dates) > 1:
                            try:
                                sorted_dates = sorted(
                                    [datetime.strptime(d, '%Y-%m-%d') for d in date_util_parse_result.dates])
                                final_time_params['start_date'] = sorted_dates[0].strftime('%Y-%m-%d')
                                final_time_params['end_date'] = sorted_dates[-1].strftime('%Y-%m-%d')
                                final_time_params['time_range_days'] = (sorted_dates[-1] - sorted_dates[0]).days + 1
                                final_time_params['has_explicit_time'] = True
                            except Exception as e_sort_dates:
                                logger.warning(f"Error processing specific dates from DateUtils: {e_sort_dates}")
                    else:
                        logger.warning(
                            f"DateUtils.parse_dates_from_query did not return usable date ranges or specific dates. Result: {date_util_parse_result}")
                except Exception as e_du_call:
                    logger.error(f"Error calling DateUtils.parse_dates_from_query: {e_du_call}")

        # 最终的默认值设置（如果所有解析都失败，或者没有显式时间）
        if not final_time_params.get('start_date') or not final_time_params.get('end_date'):
            default_days = self.analysis_config.get('default_analysis_period_days', 30)
            end_dt = datetime.now()  # 使用当前日期作为结束
            start_dt = end_dt - timedelta(days=default_days - 1)
            final_time_params.update({
                'start_date': start_dt.strftime('%Y-%m-%d'),
                'end_date': end_dt.strftime('%Y-%m-%d'),
                'time_range_days': default_days,
                'has_explicit_time': False,  # 标记为默认
                'relative_time': final_time_params.get('relative_time') or f"最近{default_days}天 (系统默认)"
            })
            logger.info(
                f"Using default time range: {final_time_params['start_date']} to {final_time_params['end_date']}")

        # 再次确保 time_range_days 与 start/end_date 一致
        if final_time_params.get('start_date') and final_time_params.get('end_date'):
            if not final_time_params.get('time_range_days') or not final_time_params.get(
                    'has_explicit_time'):  # 如果天数未设置或之前是默认
                try:
                    sd = datetime.strptime(final_time_params['start_date'], '%Y-%m-%d')
                    ed = datetime.strptime(final_time_params['end_date'], '%Y-%m-%d')
                    calculated_days = (ed - sd).days + 1
                    if calculated_days > 0:  # 确保是有效的天数
                        final_time_params['time_range_days'] = calculated_days
                except Exception as e_recalc_days:
                    logger.warning(f"Final attempt to calculate time_range_days failed: {e_recalc_days}")

        logger.debug(f"Final time parameters for historical analysis: {final_time_params}")
        return final_time_params

    def _prepare_historical_data_from_context(self,
                                              time_params: Dict[str, Any],
                                              analysis_depth: AnalysisDepth,
                                              fetcher_result: FetcherExecutionResult
                                              ) -> Dict[str, Any]:
        """
        从已获取的 FetcherExecutionResult 中提取和组织历史分析所需的数据。
        不再直接调用API。
        """
        logger.info(
            f"Preparing historical data from FetcherExecutionResult. Time range: {time_params.get('start_date')} to {time_params.get('end_date')}")

        historical_data_payload: Dict[str, Any] = {
            'time_range': time_params,
            'data_sources': {},  # 将从 fetcher_result.processed_data 中填充
            'data_quality_assessment': {  # 存储数据质量的详细评估
                'overall_quality': getattr(fetcher_result, 'data_quality', FetcherDataQualityLevel.POOR).value,
                'completeness': getattr(fetcher_result, 'data_completeness', 0.0),
                'accuracy': getattr(fetcher_result, 'accuracy_score', 0.0),
                'freshness': getattr(fetcher_result, 'freshness_score', 0.0),
                'fetcher_confidence': getattr(fetcher_result, 'confidence_level', 0.0),
            },
            'error': None  # 初始化错误信息
        }

        if not fetcher_result or not fetcher_result.processed_data:
            error_msg = "FetcherExecutionResult is missing or does not contain processed_data."
            logger.error(error_msg)
            historical_data_payload['error'] = error_msg
            historical_data_payload['data_quality_assessment'][
                'overall_quality'] = FetcherDataQualityLevel.INSUFFICIENT.value
            return historical_data_payload

        # 根据分析深度，从 fetcher_result.processed_data 中提取数据
        # 'data_keys_for_depth' 中的键名应与 SmartDataFetcher 在 processed_data 中使用的键名一致
        required_data_keys = self.analysis_config.get('data_keys_for_depth', {}).get(analysis_depth, [])

        logger.debug(
            f"Analysis depth '{analysis_depth.value}' requires data keys: {required_data_keys} from processed_data.")

        found_any_data = False
        for data_key in required_data_keys:
            if data_key in fetcher_result.processed_data:
                historical_data_payload['data_sources'][data_key] = fetcher_result.processed_data[data_key]
                logger.debug(f"Successfully extracted data segment: '{data_key}' for historical analysis.")
                found_any_data = True
            else:
                logger.warning(
                    f"Required data segment '{data_key}' for depth '{analysis_depth.value}' not found in SmartDataFetcher's processed_data. Available keys: {list(fetcher_result.processed_data.keys())}")
                # 可以考虑降低数据完整性或质量评分
                historical_data_payload['data_quality_assessment']['completeness'] = \
                    min(historical_data_payload['data_quality_assessment']['completeness'], 0.5)

        if not found_any_data:
            warn_msg = "No relevant historical data segments found in pre-fetched data for the current analysis depth."
            logger.warning(warn_msg)
            # 即使没有找到特定键，也尝试传递整个 processed_data，让下游分析步骤自己判断
            # 但这通常表明数据契约或流程有问题
            # historical_data_payload['data_sources']['fallback_full_processed_data'] = fetcher_result.processed_data
            # 或者更严格：
            historical_data_payload['error'] = warn_msg
            historical_data_payload['data_quality_assessment'][
                'overall_quality'] = FetcherDataQualityLevel.INSUFFICIENT.value

        # 还可以附加原始获取的数据，如果某些深层分析需要它
        # historical_data_payload['raw_fetched_api_data_snapshot'] = fetcher_result.fetched_data # 可能非常大，谨慎使用

        return historical_data_payload

    # ... (其他方法的实现，如 _ai_analyze_historical_data, _gpt_statistical_analysis, 等，
    #      这些方法现在会接收通过 _prepare_historical_data_from_context 准备好的数据)

    async def _ai_analyze_historical_data(self, historical_data_payload: Dict[str, Any],
                                          query_type: HistoricalQueryType,
                                          analysis_depth: AnalysisDepth,
                                          user_query: str) -> Dict[str, Any]:
        logger.info(f"🔬 AI分析历史数据: 查询类型='{query_type.value}', 分析深度='{analysis_depth.value}'")
        ai_processing_time_this_step = 0.0

        # 从 historical_data_payload 中获取 data_sources
        data_for_analysis = historical_data_payload.get('data_sources', {})
        if not data_for_analysis:
            logger.warning("No data sources found in historical_data_payload for AI analysis.")
            return {'error': 'No data provided for AI analysis', 'statistical_results': {}, 'pattern_insights': {},
                    'metadata': {'ai_processing_time': 0.0}}

        # Step 1: GPT-4o进行数值计算和统计分析 (如果配置了GPT且需要)
        statistical_analysis = {}
        if self.gpt_client and self.analysis_config.get('ai_analysis', {}).get('use_gpt_for_calculations', True):
            gpt_start_time = time.time()
            # _gpt_statistical_analysis 现在接收的是已准备好的数据字典
            statistical_analysis = await self._gpt_statistical_analysis(data_for_analysis, query_type,
                                                                        historical_data_payload.get('time_range', {}))
            ai_processing_time_this_step += (time.time() - gpt_start_time)
            logger.debug(f"GPT统计分析完成。结果键: {list(statistical_analysis.keys())}")
        else:
            logger.info("GPT客户端不可用或配置禁用，将使用基础统计计算。")
            # 提取用于基础计算的日数据部分，例如：
            daily_data_for_basic_stats = data_for_analysis.get('daily_aggregates', data_for_analysis.get('daily', []))
            statistical_analysis = self._basic_statistical_calculation(daily_data_for_basic_stats)  # 同步方法
            logger.debug(f"基础统计计算完成。结果键: {list(statistical_analysis.keys())}")

        # Step 2: Claude进行模式识别和趋势分析
        pattern_analysis = {}
        if self.claude_client and self.analysis_config.get('ai_analysis', {}).get('use_claude_for_insights', True):
            claude_start_time = time.time()
            # _claude_pattern_analysis 接收已准备数据和统计结果
            pattern_analysis = await self._claude_pattern_analysis(data_for_analysis, statistical_analysis, user_query,
                                                                   historical_data_payload.get('time_range', {}))
            ai_processing_time_this_step += (time.time() - claude_start_time)
            logger.debug(
                f"Claude模式分析完成。主要趋势: {pattern_analysis.get('trend_patterns', {}).get('primary_trend', 'N/A')}")
        else:
            logger.info("Claude客户端不可用或配置禁用，模式分析和深度洞察将受限。")
            pattern_analysis = {"trend_patterns": {"primary_trend": "AI模式分析未执行"}, "pattern_confidence": 0.5,
                                "note": "Claude client unavailable"}

        return {
            'statistical_results': statistical_analysis,
            'pattern_insights': pattern_analysis,
            'analysis_type_performed': query_type.value,
            'analysis_depth_applied': analysis_depth.value,
            'data_quality_snapshot': historical_data_payload.get('data_quality_assessment', {}),
            'metadata': {'ai_processing_time_for_analysis_step': ai_processing_time_this_step}  # 记录此步骤AI耗时
        }

    async def _gpt_statistical_analysis(self,
                                        data_for_analysis: Dict[str, Any],
                                        # 从 _prepare_historical_data_from_context 来的数据
                                        query_type: HistoricalQueryType,
                                        time_range: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"GPT进行统计分析，查询类型: {query_type.value}")
        if not self.gpt_client:
            logger.warning("GPTClient未初始化，无法执行GPT统计分析。回退到基础计算。")
            # 提取用于基础计算的日数据部分
            daily_data_for_stats = data_for_analysis.get('daily_aggregates', data_for_analysis.get('daily', []))
            return self._basic_statistical_calculation(daily_data_for_stats)

        # 准备给GPT的提示词 (中文)
        # 需要从 data_for_analysis 中选择合适的数据片段发送给GPT
        # 例如，如果 'daily_aggregates' 包含列表数据：
        daily_aggregates_sample = []
        if 'daily_aggregates' in data_for_analysis and isinstance(data_for_analysis['daily_aggregates'], list):
            daily_aggregates_sample = data_for_analysis['daily_aggregates'][:5]  # 发送前5条作为样本

        prompt = f"""
        您是一位专业的金融数据统计师。请对以下提供的历史金融数据样本进行精确的统计分析。
        用户的查询类型是关于：“{query_type.value}”

        数据样本（例如，日聚合数据的前几条记录）:
        ```json
        {json.dumps(daily_aggregates_sample, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```
        总数据点数量（如果适用）: {len(data_for_analysis.get('daily_aggregates', []))}
        分析的时间范围: 从 {time_range.get('start_date', '未知')} 到 {time_range.get('end_date', '未知')}

        请严格按照以下要求进行分析，并以JSON对象格式返回结果：
        1.  **主要指标统计**: 针对数据中的核心数值列（例如 '入金', '出金', '注册人数', '持仓人数'，根据样本判断存在哪些），计算并返回它们的：
            * `mean` (平均值), `median` (中位数), `std_dev` (标准差), `min_value` (最小值), `max_value` (最大值), `total_sum` (总和)。
        2.  **增长率分析**: 如果数据是时间序列，计算主要指标的整体增长率（例如，末期对比初期）。
            * `overall_growth_rate`: (例如：0.15 表示增长15%)。
        3.  **趋势简述**: 对主要指标的整体趋势给出一个简短描述 (例如："波动上升", "持续下降", "保持稳定")。
            * `trend_description`。
        4.  **波动性评估**: 计算一个波动性指标（例如，变异系数 CV = std_dev / mean）。
            * `volatility_coefficient`。

        返回的JSON对象结构应类似（根据实际包含的指标调整）：
        ```json
        {{
          "inflow_analysis": {{ "mean": ..., "overall_growth_rate": ..., "trend_description": "...", "volatility_coefficient": ... }},
          "user_registration_analysis": {{ "mean": ..., "overall_growth_rate": ..., "trend_description": "..." }},
          "summary_confidence": 0.85
        }}
        ```
        请确保所有数值计算精确。如果某些指标无法计算，请在该指标的分析中注明或返回null。
        """
        query_hash = self._get_query_hash(
            f"gpt_stats_{query_type.value}_{json.dumps(daily_aggregates_sample, sort_keys=True, cls=CustomJSONEncoder)}")
        ai_result = await self._cached_ai_analysis(query_hash, prompt, ai_client_type="gpt")

        if ai_result.get('success'):
            try:
                response_str = ai_result.get('response', '{}')
                parsed_stats = json.loads(response_str)
                if isinstance(parsed_stats, dict):
                    parsed_stats['ai_model_used'] = 'gpt'
                    parsed_stats['processing_time_reported_by_ai'] = ai_result.get('ai_call_duration', 0.0)
                    return parsed_stats
                else:
                    logger.error(f"GPT统计分析返回的不是有效JSON字典: {response_str[:200]}")
            except json.JSONDecodeError:
                logger.error(f"无法解析GPT统计分析的JSON响应: {ai_result.get('response', '')[:200]}")
        else:
            logger.error(f"GPT统计分析API调用失败: {ai_result.get('error')}")

        logger.warning("GPT统计分析失败或返回格式不正确，回退到基础计算。")
        return self._basic_statistical_calculation(data_for_analysis.get('daily_aggregates', []))

    async def _claude_pattern_analysis(self, data_for_analysis: Dict[str, Any],
                                       statistical_analysis: Dict[str, Any],
                                       user_query: str,
                                       time_range: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Claude进行模式分析，用户查询: {user_query[:50]}")
        if not self.claude_client:
            logger.warning("ClaudeClient未初始化，无法执行Claude模式分析。")
            return {"trend_patterns": {"primary_trend": "AI模式分析未执行 (无客户端)"},
                    "overall_pattern_confidence": 0.3}

        stats_summary_for_claude = {}  # (与之前版本一致的统计摘要逻辑)
        if 'calculated_metrics' in statistical_analysis:
            stats_summary_for_claude = statistical_analysis['calculated_metrics']
        elif isinstance(statistical_analysis, dict):
            for key, value_dict in statistical_analysis.items():
                if isinstance(value_dict, dict) and key.endswith("_analysis"):  # 来自GPT的复杂key
                    stats_summary_for_claude[key.replace("_analysis", "")] = {
                        "mean": value_dict.get('mean'), "trend": value_dict.get('trend_description'),
                        "growth_rate": value_dict.get('overall_growth_rate'),
                        "volatility": value_dict.get('volatility_coefficient')
                    }
        if not stats_summary_for_claude and statistical_analysis and 'error' not in statistical_analysis:
            stats_summary_for_claude = {"generic_stats_available": True, "details_omitted_for_prompt": True,
                                        "raw_stats_keys": list(statistical_analysis.keys())}

        prompt = f"""
        您是一位顶级的金融数据模式分析专家。请基于以下提供的历史数据统计摘要和时间范围，进行深入的模式分析。
        用户的原始查询是：“{user_query}”

        历史数据的时间范围：从 {time_range.get('start_date', '未知')} 到 {time_range.get('end_date', '未知')} ({time_range.get('time_range_days', '未知')}天)。
        数据质量评估：{data_for_analysis.get('data_quality_assessment', {}).get('overall_quality', '一般')}

        核心统计指标摘要：
        ```json
        {json.dumps(stats_summary_for_claude, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```
        请您重点分析并识别以下方面，并以JSON对象格式返回结果：
        1.  `"trend_patterns"`: (对象) 描述主要的趋势模式。
            * `"primary_trend"`: (字符串) 对整体趋势的描述。
            * `"trend_strength"`: (浮点数, 0.0-1.0) 趋势的强度。
            * `"trend_sustainability"`: (字符串) 趋势可持续性评估。
        2.  `"business_cycles"`: (对象) 描述可能的业务周期性。
            * `"cycle_detected"`: (布尔值) 是否检测到周期。
            * `"cycle_period"`: (字符串, 可选) 周期长度。
            * `"cycle_phase"`: (字符串, 可选) 当前周期阶段。
        3.  `"anomaly_detection_summary"`: (对象) 异常情况总结。
            * `"anomalies_found_description"`: (字符串) 主要异常描述。
            * `"anomaly_impact_assessment"`: (字符串) 异常的潜在业务影响。
        4.  `"correlation_patterns_summary"`: (对象) 指标间潜在关联。
            * `"strong_correlations_identified"`: (字符串列表) 观察到的强相关性。
            * `"correlation_insights_text"`: (字符串) 相关性的业务解读。
        5.  `"early_warning_signals_identified"`: (字符串列表) 潜在风险或预警信号。
        6.  `"overall_pattern_confidence"`: (浮点数, 0.0-1.0) 对本次模式分析结果的整体置信度。

        如果数据不足以进行某项分析，请在该项中明确指出“数据不足无法分析”。
        """
        query_hash = self._get_query_hash(
            f"claude_pattern_{user_query}_{json.dumps(stats_summary_for_claude, sort_keys=True, cls=CustomJSONEncoder)}")
        ai_result = await self._cached_ai_analysis(query_hash, prompt, ai_client_type="claude")

        if ai_result.get('success'):
            try:
                response_str = ai_result.get('response', '{}')
                parsed_patterns = json.loads(response_str)
                if isinstance(parsed_patterns, dict) and "trend_patterns" in parsed_patterns:
                    parsed_patterns['ai_model_used'] = 'claude'
                    parsed_patterns['processing_time_reported_by_ai'] = ai_result.get('ai_call_duration', 0.0)
                    return parsed_patterns
                else:
                    logger.error(f"Claude模式分析返回JSON结构不正确: {response_str[:300]}")
            except json.JSONDecodeError:
                logger.error(f"无法解析Claude模式分析JSON: {ai_result.get('response', '')[:300]}")
        else:
            logger.error(f"Claude模式分析API调用失败: {ai_result.get('error')}")

        return {  # 降级返回
            'trend_patterns': {'primary_trend': '趋势分析AI调用失败，请参考基础统计', 'trend_strength': 0.3,
                               'trend_sustainability': '未知'},
            'business_cycles': {'cycle_detected': False},
            'anomaly_detection_summary': {'anomalies_found_description': 'AI异常检测未执行'},
            'correlation_patterns_summary': {'strong_correlations_identified': [],
                                             'correlation_insights_text': 'AI相关性分析未执行'},
            'early_warning_signals_identified': ["AI模式分析失败，请关注基础数据"],
            'overall_pattern_confidence': 0.3, 'note': 'AI模式分析失败，结果基于有限推断。'
        }

    async def _ai_generate_business_insights(self, analysis_results_dict: Dict[str, Any],
                                             # 来自 _ai_analyze_historical_data
                                             query_type: HistoricalQueryType,
                                             user_query: str) -> Dict[str, Any]:
        # (与之前版本类似，确保中文提示词，并使用self.claude_client)
        logger.debug(f"Claude生成业务洞察，查询类型: {query_type.value}")
        if not self.claude_client:
            logger.warning("ClaudeClient未初始化，无法生成AI业务洞察。")
            return {
                'key_findings': ['历史数据基础分析已完成。'], 'business_insights': ["AI洞察服务当前不可用。"],
                'risk_warnings': [], 'opportunities': [],
                'recommendations': ['请结合具体业务场景解读数据。'], 'insight_confidence': 0.4,
                'metadata': {'ai_processing_time': 0.0, 'note': "AI洞察未执行(客户端缺失)"}
            }

        stats_summary = analysis_results_dict.get('statistical_results', {})
        pattern_summary = analysis_results_dict.get('pattern_insights', {})

        condensed_summary_for_claude = {
            "统计发现概要": {k: v for k, v_dict in stats_summary.items() if isinstance(v_dict, dict) for k, v in
                             v_dict.items() if
                             not isinstance(v, dict) and k in ['mean', 'overall_growth_rate', 'trend_description',
                                                               'volatility_coefficient']},
            "模式分析概要": {
                "主要趋势": pattern_summary.get('trend_patterns', {}).get('primary_trend'),
                "业务周期": "检测到" if pattern_summary.get('business_cycles', {}).get(
                    'cycle_detected') else "未检测到",
                "异常摘要": pattern_summary.get('anomaly_detection_summary', {}).get('anomalies_found_description')
            },
            "数据质量快照": analysis_results_dict.get('data_quality_snapshot', {})
        }

        prompt = f"""
        作为一位经验丰富的金融策略分析师和业务顾问，请基于以下提供的历史数据分析摘要，为用户生成具有深度和可操作性的中文业务洞察。
        用户的原始查询是：“{user_query}”
        系统识别的查询类型为：“{query_type.value}”

        历史数据分析核心摘要：
        ```json
        {json.dumps(condensed_summary_for_claude, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        ```
        请您重点从以下几个方面进行思考和阐述，并以JSON对象格式返回结果：
        1.  `"key_findings"`: (字符串列表) 总结出3-5个最核心、最重要的业务发现。
        2.  `"business_insights"`: (字符串列表) 对这些发现进行深入的业务解读，阐释它们背后的商业含义、可能的原因以及对当前业务状态的指示。每个解读是一个独立的字符串。
        3.  `"risk_warnings"`: (字符串列表) 基于分析，明确指出1-3个最值得关注的潜在业务风险或预警信号。如果无明显风险，返回空列表。
        4.  `"opportunities"`: (字符串列表) 识别并列出1-3个潜在的业务增长机会或改进点。如果无明显机会，返回空列表。
        5.  `"recommendations"`: (字符串列表) 针对上述风险和机会，提供2-3条具体的、可操作的行动建议。
        6.  `"insight_confidence"`: (浮点数, 0.0-1.0) 您对本次洞察分析的整体专业判断置信度。
        """
        query_hash = self._get_query_hash(
            f"claude_biz_insights_{user_query}_{json.dumps(condensed_summary_for_claude, sort_keys=True, cls=CustomJSONEncoder)}")
        ai_result = await self._cached_ai_analysis(query_hash, prompt, ai_client_type="claude")
        ai_call_duration = ai_result.get('ai_call_duration', 0.0)  # 获取AI调用耗时

        if ai_result.get('success'):
            try:
                response_str = ai_result.get('response', '{}')
                parsed_insights = json.loads(response_str)
                if isinstance(parsed_insights, dict) and \
                        all(k in parsed_insights for k in
                            ["key_findings", "business_insights", "risk_warnings", "opportunities", "recommendations",
                             "insight_confidence"]):
                    parsed_insights['metadata'] = {'ai_model_used': 'claude',
                                                   'ai_processing_time_for_insight_step': ai_call_duration}
                    return parsed_insights
                else:
                    logger.error(f"Claude业务洞察返回JSON结构不完整: {response_str[:300]}")
            except json.JSONDecodeError:
                logger.error(f"无法解析Claude业务洞察JSON: {ai_result.get('response', '')[:300]}")
        else:
            logger.error(f"Claude业务洞察API调用失败: {ai_result.get('error')}")

        return {
            'key_findings': ['AI洞察生成失败，请参考统计和模式分析结果。'], 'business_insights': [],
            'risk_warnings': [], 'opportunities': [], 'recommendations': ['建议人工审核数据。'],
            'insight_confidence': 0.3, 'metadata': {'ai_processing_time': ai_call_duration, 'note': "AI洞察生成失败"}
        }

    # _build_historical_analysis_response, _create_error_response, _update_processing_stats,
    # get_processing_stats, health_check, batch_analyze_periods 保持不变或做微小调整

        # 在 HistoricalAnalysisProcessor 类中添加此方法：
    def _basic_statistical_calculation(self, daily_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        (私有) 执行基础的统计计算，作为AI统计分析的降级方案。
        daily_data_list: 例如，包含 {'日期': 'YYYY-MM-DD', '入金': 123.45, '出金': 50.0 ...} 的字典列表。
        """
        logger.info(f"Executing basic statistical calculation for {len(daily_data_list)} data points.")
        if not daily_data_list or not isinstance(daily_data_list, list):
            logger.warning("Basic statistical calculation: No data or invalid data format provided.")
            return {'error': '无有效数据进行基础统计计算', 'summary_confidence': 0.1}

        stats_summary: Dict[str, Any] = {
            'calculated_metrics': {},
            'summary_confidence': 0.5,  # 基础计算的置信度通常低于AI增强的
            'note': "结果基于基础统计规则，未经过AI深度分析。"
        }

        # 识别数据中可能存在的数值列 (这里假设一些常见的列名)
        # 在真实场景中，您可能需要更动态地识别数值列或从元数据中获取
        possible_numeric_keys = ["入金", "出金", "注册人数", "持仓人数", "购买产品数量", "到期产品数量", "净流入"]

        # 尝试为每个可能的数值列计算统计数据
        for key in possible_numeric_keys:
            values = []
            dates = []  # 如果需要，也可以收集日期
            for item in daily_data_list:
                if isinstance(item, dict) and item.get(key) is not None:
                    try:
                        values.append(float(item[key]))
                        if '日期' in item: dates.append(item['日期'])
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping non-numeric value for key '{key}': {item[key]}")

            if len(values) >= 2:  # 至少需要两个数据点才能进行一些有意义的统计
                mean_val = sum(values) / len(values)
                median_val = sorted(values)[len(values) // 2]
                std_dev_val = statistics.stdev(values) if len(values) > 1 else 0.0
                min_val = min(values)
                max_val = max(values)
                sum_val = sum(values)
                growth_rate_val = (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else (
                    1.0 if values[-1] > 0 else 0.0)
                trend_desc = '上升' if values[-1] > values[0] else ('下降' if values[-1] < values[0] else '平稳')
                volatility_coeff = std_dev_val / abs(mean_val) if mean_val != 0 else float('inf')

                stats_summary['calculated_metrics'][f"{key}_stats"] = {
                    'mean': round(mean_val, 2),
                    'median': round(median_val, 2),
                    'std_dev': round(std_dev_val, 2),
                    'min_value': round(min_val, 2),
                    'max_value': round(max_val, 2),
                    'total_sum': round(sum_val, 2),
                    'count': len(values),
                    'overall_growth_rate': round(growth_rate_val, 4),
                    'trend_description': trend_desc,
                    'volatility_coefficient': round(volatility_coeff, 4) if volatility_coeff != float(
                        'inf') else "N/A (mean is zero)"
                }
            elif len(values) == 1:
                stats_summary['calculated_metrics'][f"{key}_stats"] = {'value': values[0], 'count': 1}

        # 示例：计算总的净流入（如果入金和出金都存在）
        if 'inflow_stats' in stats_summary['calculated_metrics'] and 'outflow_stats' in stats_summary[
            'calculated_metrics']:
            total_inflow = stats_summary['calculated_metrics']['inflow_stats'].get('sum', 0.0)
            total_outflow = stats_summary['calculated_metrics']['outflow_stats'].get('sum', 0.0)
            stats_summary['calculated_metrics']['total_net_flow'] = round(total_inflow - total_outflow, 2)
            if daily_data_list:
                stats_summary['calculated_metrics']['average_daily_net_flow'] = round(
                    (total_inflow - total_outflow) / len(daily_data_list), 2)

        if not stats_summary['calculated_metrics']:
            logger.warning("Basic statistical calculation: No numeric metrics found or calculable.")
            stats_summary['error'] = "未能从提供的数据中提取可计算的数值指标。"
            stats_summary['summary_confidence'] = 0.2

        return stats_summary
    def _build_historical_analysis_response(self, query_type: HistoricalQueryType,
                                            analysis_depth: AnalysisDepth,
                                            analysis_results_dict: Dict[str, Any],
                                            business_insights_dict: Dict[str, Any],
                                            historical_data_payload: Dict[str, Any],
                                            time_params: Dict[str, Any],
                                            processing_time_wall: float) -> HistoricalAnalysisResponse:
        logger.debug("Building final HistoricalAnalysisResponse...")

        key_metrics_ext: Dict[str, Any] = {}  # 值可以是格式化的字符串
        stats_res_for_metrics = analysis_results_dict.get('statistical_results', {})
        if isinstance(stats_res_for_metrics, dict):
            for analysis_key, metrics_val_dict in stats_res_for_metrics.items():
                if isinstance(metrics_val_dict, dict) and not any(
                        k in analysis_key for k in ['metadata', 'ai_model_used', 'note', 'confidence']):  # 排除元数据
                    for metric_name, metric_val in metrics_val_dict.items():
                        # 简单提取，实际可能需要根据指标类型格式化
                        key_metrics_ext[f"{analysis_key}_{metric_name}"] = metric_val

        pattern_res_for_metrics = analysis_results_dict.get('pattern_insights', {})
        if isinstance(pattern_res_for_metrics, dict) and isinstance(pattern_res_for_metrics.get('trend_patterns'),
                                                                    dict):
            key_metrics_ext['primary_trend_description'] = pattern_res_for_metrics['trend_patterns'].get(
                'primary_trend', 'N/A')
            key_metrics_ext['trend_strength_score'] = pattern_res_for_metrics['trend_patterns'].get('trend_strength',
                                                                                                    0.0)

        analysis_confidence = float(business_insights_dict.get('insight_confidence', 0.65))
        data_quality_info = historical_data_payload.get('data_quality_assessment', {})

        response = HistoricalAnalysisResponse(
            query_type=query_type,
            analysis_depth=analysis_depth,
            main_findings=business_insights_dict.get('key_findings', []),
            trend_summary=pattern_res_for_metrics.get('trend_patterns', {}),  # 使用模式分析中的趋势总结
            key_metrics=key_metrics_ext,  # 现在是 Dict[str, Any]
            business_insights=business_insights_dict.get('business_insights', []),
            pattern_discoveries=pattern_res_for_metrics.get('early_warning_signals_identified', []),
            comparative_analysis=pattern_res_for_metrics.get('correlation_patterns_summary', {}),  # 假设模式分析包含相关性
            risk_warnings=business_insights_dict.get('risk_warnings', []),
            opportunities=business_insights_dict.get('opportunities', []),
            recommendations=business_insights_dict.get('recommendations', []),
            data_completeness=float(data_quality_info.get('completeness', 0.0)),
            analysis_confidence=analysis_confidence,
            data_sources_used=list(historical_data_payload.get('data_sources', {}).keys()),  # 实际使用的数据片段键名
            analysis_period=f"{time_params.get('start_date', 'N/A')} 至 {time_params.get('end_date', 'N/A')}",
            processing_time=processing_time_wall,
            methodology_notes=[
                f"分析深度级别: {analysis_depth.value}",
                f"数据时间范围: {time_params.get('time_range_days', 'N/A')} 天",
                "基于AI双模型协作分析，结合统计学方法。",
                f"数据质量评估: {data_quality_info.get('overall_quality', 'N/A')} (评分: {data_quality_info.get('quality_score', 0.0):.2f})"
            ],
            metadata=business_insights_dict.get('metadata', {})  # 从洞察生成步骤获取AI耗时等
        )
        # 将 statistical_results 和 pattern_insights 中的详细内容也放入 metadata，供调试或前端深度展示
        response.metadata['full_statistical_analysis'] = stats_res_for_metrics
        response.metadata['full_pattern_analysis'] = pattern_res_for_metrics
        return response

    def _create_error_response(self, user_query: str, error: str) -> HistoricalAnalysisResponse:
        # (与之前版本一致)
        return HistoricalAnalysisResponse(
            query_type=HistoricalQueryType.UNKNOWN_HISTORICAL, analysis_depth=AnalysisDepth.BASIC,
            main_findings=[f"历史数据分析过程中发生错误: {error}"],
            trend_summary={'primary_trend': '错误', 'error_details': error}, key_metrics={"error_message": error},
            analysis_confidence=0.0, processing_time=0.01,
            methodology_notes=[f"错误详情: {error}"]
        )

    def _update_processing_stats(self, query_type: HistoricalQueryType,
                                 processing_time: float, confidence: float):
        # (与之前版本一致)
        try:
            type_key = query_type.value
            self.processing_stats.setdefault('analyses_by_type', {})
            self.processing_stats['analyses_by_type'][type_key] = self.processing_stats['analyses_by_type'].get(
                type_key, 0) + 1

            total = self.processing_stats['total_analyses']
            if total == 0: total = 1  # 避免首次调用时 total 为 0

            current_avg_time = self.processing_stats.get('avg_processing_time', 0.0)
            self.processing_stats['avg_processing_time'] = (current_avg_time * (total - 1) + processing_time) / total

            current_avg_conf = self.processing_stats.get('avg_confidence', 0.0)
            self.processing_stats['avg_confidence'] = (current_avg_conf * (total - 1) + confidence) / total

            total_cache = self.processing_stats.get('ai_cache_hits', 0) + self.processing_stats.get('ai_cache_misses',
                                                                                                    0)
            if total_cache > 0:
                self.processing_stats['ai_cache_hit_rate'] = self.processing_stats['ai_cache_hits'] / total_cache
        except Exception as e:
            logger.error(f"HistoricalAnalysisProcessor 统计信息更新失败: {str(e)}")

    def get_processing_stats(self) -> Dict[str, Any]:
        stats_copy = json.loads(json.dumps(self.processing_stats, cls=CustomJSONEncoder))
        return stats_copy

    async def health_check(self) -> Dict[str, Any]:
        # (与之前版本一致)
        claude_ok = self.claude_client is not None
        gpt_ok = self.gpt_client is not None
        # 更精细的健康检查可以实际调用AI客户端的测试接口
        status = "healthy"
        issues = []
        if not claude_ok: issues.append("ClaudeClient not configured.")
        if not gpt_ok: issues.append("GPTClient not configured.")
        if issues: status = "degraded" if (claude_ok or gpt_ok) else "unhealthy"
        return {
            "status": status, "component_name": self.__class__.__name__,
            "ai_clients_status": {"claude_available": claude_ok, "gpt_available": gpt_ok},
            "issues": issues, "timestamp": datetime.now().isoformat()
        }

    async def batch_analyze_periods(self, periods_info: List[Dict[str, Any]],
                                    user_context: Optional[Dict[str, Any]] = None) -> List[HistoricalAnalysisResponse]:
        # (与之前版本一致)
        if not periods_info: return []
        # 确保 user_context 传递给每个 process_historical_analysis_query 调用
        tasks = [self.process_historical_analysis_query(p_info['user_query'], user_context=user_context) for p_info in
                 periods_info]
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for i, res_or_exc in enumerate(results_or_exceptions):
            if isinstance(res_or_exc, Exception):
                logger.error(f"批量分析失败 for query '{periods_info[i]['user_query']}': {res_or_exc}")
                final_results.append(self._create_error_response(periods_info[i]['user_query'], str(res_or_exc)))
            else:
                final_results.append(res_or_exc)
        return final_results


# --- 工厂函数 (保持不变) ---
def create_historical_analysis_processor(claude_client=None, gpt_client=None) -> HistoricalAnalysisProcessor:
    return HistoricalAnalysisProcessor(claude_client, gpt_client)

# --- 移除了 async def main() ---