# core/orchestrator/intelligent_qa_orchestrator.py - 重构简化版
"""
🧠 Claude驱动的智能问答编排器 - 简化版
职责明确：Claude理解 + API获取 + GPT计算 + Claude回答

重构改进：
- 删除冗余的 processor
- 简化流程：Claude → API → 计算 → 回答
- 使用统一计算器
- 保持对话管理功能
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import traceback

# 🎯 简化后的核心组件导入
from core.analyzers.query_parser import (
    SmartQueryParser, create_smart_query_parser, QueryAnalysisResult,
    QueryComplexity, QueryType, BusinessScenario
)
from core.data_orchestration.smart_data_fetcher import (
    SmartDataFetcher, create_smart_data_fetcher,
    ExecutionResult as FetcherExecutionResult,
    DataQualityLevel as FetcherDataQualityLevel,
    ExecutionStatus as FetcherExecutionStatus
)
from core.analyzers.insight_generator import (
    InsightGenerator, create_insight_generator, BusinessInsight
)

# 🆕 使用你的统一计算器
from utils.calculators.statistical_calculator import (
    UnifiedCalculator, create_unified_calculator, UnifiedCalculationResult,
    CalculationType
)

# 工具类和其他必要组件
from utils.helpers.date_utils import DateUtils, create_date_utils
from utils.formatters.financial_formatter import FinancialFormatter, create_financial_formatter
from utils.formatters.chart_generator import ChartGenerator, create_chart_generator, ChartType
from utils.formatters.report_generator import ReportGenerator, create_report_generator
from data.models.conversation import ConversationManager, create_conversation_manager
from data.connectors.database_connector import DatabaseConnector, create_database_connector

# AI 客户端导入
from core.models.claude_client import ClaudeClient, CustomJSONEncoder
from core.models.openai_client import OpenAIClient
from config import Config as AppConfig

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """简化的处理策略"""
    SIMPLE_DATA = "simple_data"  # 简单数据获取
    DATA_WITH_CALC = "data_with_calc"  # 数据获取 + 计算
    COMPREHENSIVE = "comprehensive"  # 全面分析
    ERROR_HANDLING = "error_handling"  # 错误处理


@dataclass
class ProcessingResult:
    """处理结果 - 保持兼容性"""
    session_id: str
    query_id: str
    success: bool
    response_text: str
    insights: List[BusinessInsight] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    processing_strategy: ProcessingStrategy = ProcessingStrategy.SIMPLE_DATA
    processors_used: List[str] = field(default_factory=list)
    ai_collaboration_summary: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    data_quality_score: float = 0.0
    response_completeness: float = 0.0
    total_processing_time: float = 0.0
    ai_processing_time: float = 0.0
    data_fetching_time: float = 0.0
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    query_analysis_snapshot: Optional[Dict[str, Any]] = None


class IntelligentQAOrchestrator:
    """🧠 简化版智能问答编排器"""

    _instance: Optional['IntelligentQAOrchestrator'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(IntelligentQAOrchestrator, cls).__new__(cls)
        return cls._instance

    def __init__(self, claude_client_instance: Optional[ClaudeClient] = None,
                 gpt_client_instance: Optional[OpenAIClient] = None,
                 db_connector_instance: Optional[DatabaseConnector] = None,
                 app_config_instance: Optional[AppConfig] = None):

        if hasattr(self, 'initialized') and self.initialized:
            return  # 避免重复初始化

        # 基础配置
        self.claude_client = claude_client_instance
        self.gpt_client = gpt_client_instance
        self.db_connector = db_connector_instance
        self.app_config = app_config_instance if app_config_instance is not None else AppConfig()
        self.config = self._load_orchestrator_config()

        self.initialized = False
        self._initialize_component_placeholders()

        # 统计和缓存
        self.orchestrator_stats = self._default_stats()
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = self.config.get('cache_ttl_seconds', 1800)

        logger.info("简化版 IntelligentQAOrchestrator 创建完成")

    def _initialize_component_placeholders(self):
        """初始化组件占位符 - 🎯 大幅简化"""
        # 🎯 核心组件 - 只保留必要的
        self.query_parser: Optional[SmartQueryParser] = None
        self.data_fetcher: Optional[SmartDataFetcher] = None
        self.statistical_calculator: Optional[UnifiedCalculator] = None  # 🆕 使用统一计算器
        self.insight_generator: Optional[InsightGenerator] = None

        # 工具组件
        self.date_utils: Optional[DateUtils] = None
        self.financial_formatter: Optional[FinancialFormatter] = None
        self.chart_generator: Optional[ChartGenerator] = None
        self.report_generator: Optional[ReportGenerator] = None
        self.conversation_manager: Optional[ConversationManager] = None

        # ❌ 删除的冗余组件
        # self.data_requirements_analyzer = None  # 功能已合并到query_parser
        # self.financial_data_analyzer = None     # 功能已合并到statistical_calculator
        # self.current_data_processor = None      # 完全冗余
        # self.historical_analysis_processor = None  # 计算逻辑已提取到statistical_calculator
        # self.prediction_processor = None        # 计算逻辑已提取到statistical_calculator

    def _load_orchestrator_config(self) -> Dict[str, Any]:
        """加载配置 - 保持不变"""
        cfg = {
            'max_processing_time': getattr(self.app_config, 'MAX_PROCESSING_TIME', 120),
            'enable_intelligent_caching': getattr(self.app_config, 'ENABLE_INTELLIGENT_CACHING', True),
            'min_confidence_threshold': getattr(self.app_config, 'MIN_CONFIDENCE_THRESHOLD', 0.6),
            'cache_ttl_seconds': getattr(self.app_config, 'CACHE_TTL', 1800),
            'claude_timeout': getattr(self.app_config, 'CLAUDE_TIMEOUT', 60),
            'gpt_timeout': getattr(self.app_config, 'GPT_TIMEOUT', 40),
            'DEBUG': getattr(self.app_config, 'DEBUG', False),
            'version': getattr(self.app_config, 'VERSION', '2.2.0-simplified')
        }

        # API配置
        if hasattr(self.app_config, 'CLAUDE_API_KEY'):
            cfg['CLAUDE_API_KEY'] = self.app_config.CLAUDE_API_KEY
        if hasattr(self.app_config, 'OPENAI_API_KEY'):
            cfg['OPENAI_API_KEY'] = self.app_config.OPENAI_API_KEY
        if hasattr(self.app_config, 'DATABASE_CONFIG'):
            cfg['DATABASE_CONFIG'] = self.app_config.DATABASE_CONFIG

        # API连接器配置
        api_connector_cfg = {}
        if hasattr(self.app_config, 'FINANCE_API_BASE_URL'):
            api_connector_cfg['base_url'] = self.app_config.FINANCE_API_BASE_URL
        if hasattr(self.app_config, 'FINANCE_API_KEY'):
            api_connector_cfg['api_key'] = self.app_config.FINANCE_API_KEY
        if api_connector_cfg:
            cfg['api_connector_config'] = api_connector_cfg

        return cfg

    async def initialize(self):
        """🎯 简化版初始化"""
        if self.initialized:
            logger.debug("Orchestrator already initialized.")
            return

        logger.info("🚀 开始初始化简化版智能编排器...")
        start_init_time = time.time()

        try:
            # 初始化AI客户端
            if not self.claude_client and self.config.get('CLAUDE_API_KEY'):
                self.claude_client = ClaudeClient(api_key=self.config['CLAUDE_API_KEY'])
            if not self.gpt_client and self.config.get('OPENAI_API_KEY'):
                self.gpt_client = OpenAIClient(api_key=self.config['OPENAI_API_KEY'])

            # 数据库连接器
            if not self.db_connector and self.config.get('DATABASE_CONFIG'):
                db_cfg = self.config['DATABASE_CONFIG']
                if all(key in db_cfg for key in ['user', 'password', 'host', 'database']):
                    self.db_connector = create_database_connector(db_cfg)
                    logger.info("DatabaseConnector 初始化完成")

            # 🎯 核心组件初始化 - 大幅简化
            self.query_parser = create_smart_query_parser(self.claude_client, self.gpt_client)

            fetcher_config = self.config.get('api_connector_config', {})
            self.data_fetcher = create_smart_data_fetcher(self.claude_client, self.gpt_client, fetcher_config)

            # 🆕 使用统一计算器
            self.statistical_calculator = create_unified_calculator(self.gpt_client, precision=6)

            self.insight_generator = create_insight_generator(self.claude_client, self.gpt_client)

            # 工具组件
            self.date_utils = create_date_utils(self.claude_client)
            self.financial_formatter = create_financial_formatter()
            self.chart_generator = create_chart_generator()
            self.report_generator = create_report_generator()

            # 对话管理器
            if self.db_connector:
                self.conversation_manager = create_conversation_manager(self.db_connector)
                logger.info("ConversationManager 初始化完成")
            else:
                logger.warning("数据库连接器不可用，ConversationManager 使用内存模式")
                self.conversation_manager = ConversationManager(database_connector=None)

            self.initialized = True
            init_duration = time.time() - start_init_time
            logger.info(f"✅ 简化版智能编排器初始化完成 (耗时: {init_duration:.2f}s)")

        except Exception as e:
            self.initialized = False
            logger.error(f"❌ 智能编排器初始化失败: {str(e)}\n{traceback.format_exc()}")

    # ================================================================
    # 🎯 核心方法：简化版智能查询处理
    # ================================================================

    async def process_intelligent_query(self, user_query: str, user_id: int = 0,
                                        conversation_id: Optional[str] = None,
                                        preferences: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """🎯 简化版智能查询处理 - 核心流程"""

        if not self.initialized:
            await self.initialize()
            if not self.initialized:
                return self._create_error_result("系统初始化失败", user_query)

        session_id = str(uuid.uuid4())
        query_id = f"q_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{hashlib.md5(user_query.encode('utf-8')).hexdigest()[:6]}"
        start_time = time.time()

        logger.info(f"🎯 QueryID: {query_id} - 开始简化版处理: '{user_query[:50]}...'")
        self.orchestrator_stats['total_queries'] += 1

        # 处理对话ID
        conversation_id_for_db = self._parse_conversation_id(conversation_id)

        # 保存用户输入
        user_message_saved = await self._save_user_message_if_needed(
            conversation_id_for_db, user_query, query_id)

        try:
            # 🎯 简化的处理流程
            timing = {}

            # 1️⃣ Claude 理解查询
            start_t = time.time()
            query_analysis = await self._claude_understand_query(user_query, conversation_id_for_db)
            timing['parsing'] = time.time() - start_t

            # 2️⃣ 获取数据
            start_t = time.time()
            data_result = await self._execute_data_acquisition(query_analysis)
            timing['data_fetching'] = time.time() - start_t

            # 3️⃣ 计算处理 (如果需要)
            start_t = time.time()
            calculation_result = None
            if query_analysis.needs_calculation:
                calculation_result = await self._execute_calculation(query_analysis, data_result)
            timing['calculation'] = time.time() - start_t

            # 4️⃣ 生成洞察
            start_t = time.time()
            insights = await self._generate_insights(data_result, calculation_result, query_analysis)
            timing['insights'] = time.time() - start_t

            # 5️⃣ Claude 生成最终回答
            start_t = time.time()
            response_text = await self._claude_generate_response(
                user_query, query_analysis, data_result, calculation_result, insights)
            timing['response_generation'] = time.time() - start_t

            # 6️⃣ 生成可视化
            start_t = time.time()
            visualizations = await self._generate_visualizations(data_result, calculation_result)
            timing['visualization'] = time.time() - start_t

            # 构建结果
            total_processing_time = time.time() - start_time
            confidence = self._calculate_confidence(query_analysis, data_result, calculation_result, insights)

            result = ProcessingResult(
                session_id=session_id,
                query_id=query_id,
                success=True,
                response_text=response_text,
                insights=insights,
                key_metrics=self._extract_key_metrics(data_result, calculation_result),
                visualizations=visualizations,
                processing_strategy=self._determine_strategy(query_analysis),
                processors_used=self._get_processors_used(query_analysis, calculation_result),
                ai_collaboration_summary=self._get_ai_summary(timing),
                confidence_score=confidence,
                data_quality_score=getattr(data_result, 'confidence_level', 0.8),
                response_completeness=self._calculate_completeness(data_result, calculation_result, insights),
                total_processing_time=total_processing_time,
                ai_processing_time=timing.get('parsing', 0) + timing.get('response_generation', 0),
                data_fetching_time=timing.get('data_fetching', 0),
                processing_metadata={
                    'query_complexity': query_analysis.complexity.value,
                    'query_type': query_analysis.query_type.value,
                    'step_times': timing,
                    'simplified_architecture': True
                },
                conversation_id=conversation_id,
                query_analysis_snapshot=query_analysis.to_dict()
            )

            # 更新统计和缓存
            self._update_stats(result)
            await self._cache_result_if_appropriate(result)

            # 保存AI响应
            if conversation_id_for_db and user_message_saved:
                await self._save_ai_response_if_needed(conversation_id_for_db, result, query_id)

            logger.info(f"✅ QueryID: {query_id} - 处理成功，耗时: {total_processing_time:.2f}s")
            return result

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"❌ QueryID: {query_id} - 处理失败: {str(e)}\n{traceback.format_exc()}")
            return await self._handle_error(session_id, query_id, user_query, str(e), error_time, conversation_id)

    # ================================================================
    # 🎯 简化后的核心处理方法
    # ================================================================

    async def _claude_understand_query(self, user_query: str,
                                       conversation_id_for_db: Optional[int]) -> QueryAnalysisResult:
        """1️⃣ Claude 理解查询"""
        logger.debug(f"🧠 Claude 理解查询: {user_query[:50]}...")

        # 获取对话上下文
        context = {}
        if conversation_id_for_db and self.conversation_manager:
            try:
                context = self.conversation_manager.get_context(conversation_id_for_db)
            except Exception as e:
                logger.warning(f"获取对话上下文失败: {e}")

        # 调用重构后的 query_parser
        query_analysis = await self.query_parser.parse_complex_query(user_query, context)

        if not query_analysis or query_analysis.confidence_score < 0.3:
            logger.warning("Claude 理解失败，使用降级解析")
            # query_parser 内部已有降级逻辑

        return query_analysis

    async def _execute_data_acquisition(self, query_analysis: QueryAnalysisResult) -> FetcherExecutionResult:
        """2️⃣ 执行数据获取"""
        logger.debug(f"📊 执行数据获取: {len(query_analysis.api_calls_needed)} 个API调用")

        if not self.data_fetcher:
            raise RuntimeError("SmartDataFetcher 未初始化")

        # 构建数据获取计划 (简化版)
        api_calls = query_analysis.api_calls_needed

        if not api_calls:
            # 默认获取系统数据
            api_calls = [{"method": "get_system_data", "params": {}, "reason": "默认数据获取"}]

        # 执行API调用
        data_result = await self.data_fetcher.execute_api_calls_batch(api_calls)

        return data_result

    async def _execute_calculation(self, query_analysis: QueryAnalysisResult,
                                   data_result: FetcherExecutionResult) -> Optional[Dict[str, Any]]:
        """3️⃣ 执行计算处理"""
        if not query_analysis.needs_calculation:
            return None

        logger.debug(f"🧮 执行计算: {query_analysis.calculation_type}")

        if not self.statistical_calculator:
            logger.error("统一计算器未初始化")
            return None

        try:
            # 准备计算数据
            calc_data = self._prepare_calculation_data(data_result)
            calc_params = self._extract_calculation_params(query_analysis)

            # 🆕 使用统一计算器
            calc_result = await self.statistical_calculator.calculate(
                calculation_type=query_analysis.calculation_type,
                data=calc_data,
                params=calc_params
            )

            return {
                'calculation_result': calc_result,
                'calculation_type': query_analysis.calculation_type,
                'success': calc_result.success if calc_result else False,
                'confidence': calc_result.confidence if calc_result else 0.5
            }

        except Exception as e:
            logger.error(f"计算执行失败: {e}")
            return {
                'calculation_result': None,
                'calculation_type': query_analysis.calculation_type,
                'success': False,
                'error': str(e)
            }

    async def _generate_insights(self, data_result: FetcherExecutionResult,
                                 calculation_result: Optional[Dict[str, Any]],
                                 query_analysis: QueryAnalysisResult) -> List[BusinessInsight]:
        """4️⃣ 生成业务洞察"""
        logger.debug("💡 生成业务洞察")

        if not self.insight_generator:
            return []

        try:
            # 准备分析结果
            analysis_results = []

            if data_result:
                analysis_results.append(data_result)

            if calculation_result and calculation_result.get('success'):
                analysis_results.append(calculation_result['calculation_result'])

            if not analysis_results:
                return []

            # 生成洞察
            insights, _ = await self.insight_generator.generate_comprehensive_insights(
                analysis_results=analysis_results,
                user_context={},
                focus_areas=self._determine_focus_areas(query_analysis)
            )

            return insights

        except Exception as e:
            logger.error(f"洞察生成失败: {e}")
            return []

    async def _claude_generate_response(self, user_query: str, query_analysis: QueryAnalysisResult,
                                        data_result: FetcherExecutionResult,
                                        calculation_result: Optional[Dict[str, Any]],
                                        insights: List[BusinessInsight]) -> str:
        """5️⃣ Claude 生成最终回答"""
        logger.debug("✍️ Claude 生成最终回答")

        if not self.claude_client:
            return self._generate_fallback_response(user_query, data_result, calculation_result, insights)

        try:
            # 构建给Claude的上下文
            context_for_claude = {
                "user_query": user_query,
                "query_analysis": {
                    "type": query_analysis.query_type.value,
                    "complexity": query_analysis.complexity.value,
                    "confidence": query_analysis.confidence_score
                },
                "data_summary": self._summarize_data_for_claude(data_result),
                "calculation_summary": self._summarize_calculation_for_claude(calculation_result),
                "insights_summary": [
                    {"title": getattr(insight, 'title', ''), "summary": getattr(insight, 'summary', '')}
                    for insight in insights[:3]
                ]
            }

            prompt = f"""
作为专业的AI金融分析助手，请根据以下信息为用户生成一个全面、准确、易懂的中文回答。

用户查询："{user_query}"

分析上下文：
{json.dumps(context_for_claude, ensure_ascii=False, indent=2)}

请生成一个结构清晰的回答，包括：
1. 直接回答用户的核心问题
2. 关键数据和发现
3. 重要的业务洞察
4. 如果有计算结果，清楚地解释数字的含义
5. 必要的建议或后续行动

要求：
- 使用专业但易懂的语言
- 突出最重要的信息
- 如果数据有限或不确定，请诚实说明
- 回答长度控制在300-800字
"""

            # 调用Claude
            response = await self.claude_client.generate_text(prompt, max_tokens=2000)

            if response and response.get('success'):
                return response.get('text', '').strip()
            else:
                logger.warning("Claude 回答生成失败，使用降级方案")
                return self._generate_fallback_response(user_query, data_result, calculation_result, insights)

        except Exception as e:
            logger.error(f"Claude 回答生成异常: {e}")
            return self._generate_fallback_response(user_query, data_result, calculation_result, insights)

    # ================================================================
    # 🛠️ 辅助方法
    # ================================================================

    def _prepare_calculation_data(self, data_result: FetcherExecutionResult) -> Dict[str, Any]:
        """准备计算数据"""
        if not data_result:
            return {}

        calc_data = {}

        # 从 processed_data 提取
        if hasattr(data_result, 'processed_data') and data_result.processed_data:
            calc_data.update(data_result.processed_data)

        # 从 fetched_data 提取
        if hasattr(data_result, 'fetched_data') and data_result.fetched_data:
            calc_data.update(data_result.fetched_data)

        return calc_data

    def _extract_calculation_params(self, query_analysis: QueryAnalysisResult) -> Dict[str, Any]:
        """提取计算参数"""
        params = {}

        # 时间范围
        if query_analysis.time_range:
            params.update(query_analysis.time_range)

        # 处理元数据
        if query_analysis.processing_metadata:
            params.update(query_analysis.processing_metadata)

        return params

    def _summarize_data_for_claude(self, data_result: FetcherExecutionResult) -> Dict[str, Any]:
        """为Claude总结数据"""
        if not data_result:
            return {"status": "无数据"}

        summary = {
            "status": "数据获取成功" if getattr(data_result, 'execution_status', None) else "数据获取失败",
            "data_quality": getattr(data_result, 'confidence_level', 0.5),
            "data_sources": getattr(data_result, 'data_sources_used', [])
        }

        # 添加关键数据摘要
        if hasattr(data_result, 'processed_data') and data_result.processed_data:
            summary["key_data_points"] = len(data_result.processed_data)

        return summary

    def _summarize_calculation_for_claude(self, calculation_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """为Claude总结计算结果"""
        if not calculation_result:
            return {"status": "无计算"}

        if not calculation_result.get('success'):
            return {"status": "计算失败", "error": calculation_result.get('error')}

        calc_res = calculation_result.get('calculation_result')
        if not calc_res:
            return {"status": "计算结果为空"}

        return {
            "status": "计算成功",
            "calculation_type": calculation_result.get('calculation_type'),
            "confidence": calculation_result.get('confidence', 0.5),
            "has_detailed_results": bool(getattr(calc_res, 'detailed_results', None))
        }

    def _generate_fallback_response(self, user_query: str, data_result: FetcherExecutionResult,
                                    calculation_result: Optional[Dict[str, Any]],
                                    insights: List[BusinessInsight]) -> str:
        """生成降级回答"""
        response_parts = ["根据系统分析：\n"]

        # 数据状态
        if data_result and hasattr(data_result, 'execution_status'):
            response_parts.append(f"数据获取状态：{'成功' if data_result.execution_status else '部分成功'}")

        # 计算结果
        if calculation_result and calculation_result.get('success'):
            calc_res = calculation_result.get('calculation_result')
            if calc_res and hasattr(calc_res, 'primary_result'):
                response_parts.append(f"计算结果：{calc_res.primary_result}")

        # 洞察
        if insights:
            response_parts.append("\n关键洞察：")
            for i, insight in enumerate(insights[:2], 1):
                title = getattr(insight, 'title', f'洞察{i}')
                summary = getattr(insight, 'summary', '...')
                response_parts.append(f"{i}. {title}: {summary}")

        if len(response_parts) == 1:  # 只有开头
            response_parts.append("抱歉，暂时无法提供详细分析。请稍后重试或联系技术支持。")

        return "\n".join(response_parts)

    def _determine_strategy(self, query_analysis: QueryAnalysisResult) -> ProcessingStrategy:
        """确定处理策略"""
        if query_analysis.needs_calculation:
            return ProcessingStrategy.DATA_WITH_CALC
        elif query_analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            return ProcessingStrategy.COMPREHENSIVE
        else:
            return ProcessingStrategy.SIMPLE_DATA

    def _get_processors_used(self, query_analysis: QueryAnalysisResult,
                             calculation_result: Optional[Dict[str, Any]]) -> List[str]:
        """获取使用的处理器"""
        processors = ["Claude", "SmartDataFetcher"]

        if query_analysis.needs_calculation and calculation_result:
            processors.append("UnifiedCalculator")

        return processors

    def _calculate_confidence(self, query_analysis: QueryAnalysisResult,
                              data_result: FetcherExecutionResult,
                              calculation_result: Optional[Dict[str, Any]],
                              insights: List[BusinessInsight]) -> float:
        """计算整体置信度"""
        confidence_factors = []

        # 查询理解置信度
        confidence_factors.append(query_analysis.confidence_score)

        # 数据质量
        if data_result:
            confidence_factors.append(getattr(data_result, 'confidence_level', 0.5))

        # 计算置信度
        if calculation_result:
            confidence_factors.append(calculation_result.get('confidence', 0.5))

        # 洞察质量
        if insights:
            insight_confidence = sum(getattr(insight, 'confidence_score', 0.7) for insight in insights) / len(insights)
            confidence_factors.append(insight_confidence)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _calculate_completeness(self, data_result: FetcherExecutionResult,
                                calculation_result: Optional[Dict[str, Any]],
                                insights: List[BusinessInsight]) -> float:
        """计算回答完整性"""
        completeness = 0.0

        if data_result and hasattr(data_result, 'execution_status'):
            completeness += 0.4

        if calculation_result and calculation_result.get('success'):
            completeness += 0.3

        if insights:
            completeness += 0.3

        return min(completeness, 1.0)

    def _extract_key_metrics(self, data_result: FetcherExecutionResult,
                             calculation_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """提取关键指标"""
        metrics = {}

        # 从计算结果提取
        if calculation_result and calculation_result.get('success'):
            calc_res = calculation_result.get('calculation_result')
            if calc_res and hasattr(calc_res, 'detailed_results'):
                metrics.update(calc_res.detailed_results)

        # 从数据结果提取基础指标
        if data_result and hasattr(data_result, 'processed_data'):
            processed_data = data_result.processed_data
            if isinstance(processed_data, dict):
                # 提取系统概览数据中的关键指标
                for key, value in processed_data.items():
                    if isinstance(value, dict) and '总余额' in value:
                        metrics.update({
                            '总余额': value.get('总余额'),
                            '总入金': value.get('总入金'),
                            '总出金': value.get('总出金')
                        })
                        break

        return metrics

    # ================================================================
    # 🎯 保留的必要方法 (简化版)
    # ================================================================

    def _parse_conversation_id(self, conversation_id: Optional[str]) -> Optional[int]:
        """解析对话ID"""
        if not conversation_id:
            return None
        try:
            return int(conversation_id)
        except ValueError:
            logger.warning(f"无效的对话ID: {conversation_id}")
            return None

    async def _save_user_message_if_needed(self, conversation_id_for_db: Optional[int],
                                           user_query: str, query_id: str) -> bool:
        """保存用户消息（如果需要）"""
        if not conversation_id_for_db or not self.conversation_manager:
            return False

        try:
            # 简单的重复检查
            recent_messages = getattr(self.conversation_manager, 'get_recent_messages', lambda *args: [])(
                conversation_id_for_db, 2)

            # 检查最近的用户消息是否相同
            for msg in recent_messages:
                if isinstance(msg, dict):
                    is_user = msg.get('is_user', False)
                    content = msg.get('content', '')
                    if is_user and content.strip() == user_query.strip():
                        return False  # 跳过重复保存

            # 保存用户消息
            self.conversation_manager.add_message(conversation_id_for_db, True, user_query)
            return True

        except Exception as e:
            logger.error(f"保存用户消息失败: {e}")
            return False

    async def _save_ai_response_if_needed(self, conversation_id_for_db: int,
                                          result: ProcessingResult, query_id: str):
        """保存AI响应（如果需要）"""
        if not self.conversation_manager:
            return

        try:
            # 保存AI响应
            ai_message_id = self.conversation_manager.add_message(
                conversation_id_for_db, False, result.response_text)

            # 保存可视化（如果支持）
            if result.visualizations and hasattr(self.conversation_manager, 'add_visual'):
                for i, vis in enumerate(result.visualizations):
                    if isinstance(vis, dict):
                        self.conversation_manager.add_visual(
                            message_id=ai_message_id,
                            visual_type=vis.get('type', 'chart'),
                            visual_order=i,
                            title=vis.get('title', '图表'),
                            data=vis.get('data_payload', {})
                        )
        except Exception as e:
            logger.error(f"保存AI响应失败: {e}")

    def _default_stats(self) -> Dict[str, Any]:
        """默认统计"""
        return {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time': 0.0,
            'avg_confidence_score': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def _update_stats(self, result: ProcessingResult):
        """更新统计"""
        if result.success:
            self.orchestrator_stats['successful_queries'] += 1
        else:
            self.orchestrator_stats['failed_queries'] += 1

        # 更新平均值
        total = self.orchestrator_stats['total_queries']
        if total > 0:
            current_avg_time = self.orchestrator_stats['avg_processing_time']
            self.orchestrator_stats['avg_processing_time'] = (
                    (current_avg_time * (total - 1) + result.total_processing_time) / total)

            current_avg_conf = self.orchestrator_stats['avg_confidence_score']
            self.orchestrator_stats['avg_confidence_score'] = (
                    (current_avg_conf * (total - 1) + result.confidence_score) / total)

    async def _cache_result_if_appropriate(self, result: ProcessingResult):
        """缓存结果（如果合适）"""
        if (self.config.get('enable_intelligent_caching', True) and
                result.success and result.confidence_score > 0.7):
            cache_key = f"result_{result.query_id}"
            self.result_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }

    def _create_error_result(self, error_msg: str, user_query: str,
                             query_id: str = None) -> ProcessingResult:
        """创建错误结果"""
        return ProcessingResult(
            session_id=str(uuid.uuid4()),
            query_id=query_id or f"error_{int(time.time())}",
            success=False,
            response_text=f"抱歉，处理您的查询时遇到问题：{error_msg}",
            processing_strategy=ProcessingStrategy.ERROR_HANDLING,
            processors_used=["ErrorHandler"],
            error_info={"message": error_msg, "query": user_query}
        )

    async def _handle_error(self, session_id: str, query_id: str, user_query: str,
                            error_msg: str, processing_time: float,
                            conversation_id: Optional[str]) -> ProcessingResult:
        """处理错误"""
        self.orchestrator_stats['failed_queries'] += 1

        return ProcessingResult(
            session_id=session_id,
            query_id=query_id,
            success=False,
            response_text=f"处理查询时发生错误。我们已记录此问题，请稍后重试。",
            processing_strategy=ProcessingStrategy.ERROR_HANDLING,
            processors_used=["ErrorHandler"],
            total_processing_time=processing_time,
            error_info={"message": error_msg, "query": user_query},
            conversation_id=conversation_id
        )

    # ================================================================
    # 🔧 接口方法 (保持兼容性)
    # ================================================================

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """获取编排器统计"""
        stats = self.orchestrator_stats.copy()
        total = stats.get('total_queries', 0)
        if total > 0:
            stats['success_rate'] = stats.get('successful_queries', 0) / total
            stats['failure_rate'] = stats.get('failed_queries', 0) / total
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        if not self.initialized:
            try:
                await self.initialize()
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'reason': 'Initialization failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        return {
            'status': 'healthy',
            'architecture': 'simplified',
            'version': self.config.get('version', '2.2.0-simplified'),
            'components': {
                'claude_available': self.claude_client is not None,
                'gpt_available': self.gpt_client is not None,
                'query_parser_ready': self.query_parser is not None,
                'data_fetcher_ready': self.data_fetcher is not None,
                'calculator_ready': self.statistical_calculator is not None,
                'insight_generator_ready': self.insight_generator is not None
            },
            'statistics': self.get_orchestrator_stats(),
            'timestamp': datetime.now().isoformat()
        }

    async def close(self):
        """关闭编排器"""
        logger.info("关闭简化版智能编排器...")

        if self.data_fetcher and hasattr(self.data_fetcher, 'close'):
            await self.data_fetcher.close()

        if self.db_connector and hasattr(self.db_connector, 'close'):
            await self.db_connector.close()

        self.result_cache.clear()
        self.initialized = False

        logger.info("简化版智能编排器已关闭")

    # ================================================================
    # 🔧 缺失的方法实现
    # ================================================================

    def _determine_focus_areas(self, query_analysis: QueryAnalysisResult) -> List[str]:
        """确定洞察关注领域"""
        focus_areas = []

        query_type = query_analysis.query_type.value if query_analysis.query_type else ""
        business_scenario = query_analysis.business_scenario.value if query_analysis.business_scenario else ""

        if 'financial' in business_scenario or 'data_retrieval' in query_type:
            focus_areas.append('financial_health')

        if 'trend' in query_type or 'historical' in business_scenario:
            focus_areas.append('trend_analysis')

        if 'prediction' in query_type or 'future' in business_scenario:
            focus_areas.append('future_outlook')

        if 'user' in business_scenario:
            focus_areas.append('user_behavior')

        return focus_areas if focus_areas else ['general_analysis']

    def _get_ai_summary(self, timing: Dict[str, float]) -> Dict[str, Any]:
        """获取AI协作摘要"""
        return {
            'claude_used': self.claude_client is not None,
            'gpt_used': self.gpt_client is not None,
            'total_ai_time': timing.get('parsing', 0) + timing.get('response_generation', 0),
            'architecture': 'simplified_claude_dominant'
        }

    async def _generate_visualizations(self, data_result: FetcherExecutionResult,
                                       calculation_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成可视化"""
        visualizations = []

        if not self.chart_generator:
            return visualizations

        try:
            # 从数据结果生成图表
            if data_result and hasattr(data_result, 'processed_data'):
                # 生成基础数据图表的逻辑
                pass

            # 从计算结果生成图表
            if calculation_result and calculation_result.get('success'):
                calc_res = calculation_result.get('calculation_result')
                if calc_res and hasattr(calc_res, 'detailed_results'):
                    # 生成计算结果图表的逻辑
                    pass

        except Exception as e:
            logger.error(f"可视化生成失败: {e}")

        return visualizations


# ================================================================
# 🏭 工厂函数和全局实例管理
# ================================================================

_orchestrator_instance: Optional[IntelligentQAOrchestrator] = None


def get_orchestrator(claude_client_instance: Optional[ClaudeClient] = None,
                     gpt_client_instance: Optional[OpenAIClient] = None,
                     db_connector_instance: Optional[DatabaseConnector] = None,
                     app_config_instance: Optional[AppConfig] = None) -> IntelligentQAOrchestrator:
    """获取编排器实例"""
    global _orchestrator_instance

    if _orchestrator_instance is None:
        logger.info("创建新的简化版智能编排器实例")
        _orchestrator_instance = IntelligentQAOrchestrator(
            claude_client_instance, gpt_client_instance,
            db_connector_instance, app_config_instance
        )

    return _orchestrator_instance