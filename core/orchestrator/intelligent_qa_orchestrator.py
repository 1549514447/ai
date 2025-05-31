"""
🧠 Claude驱动的智能问答编排器 - 智能增强版
核心改进：
- 智能数据提取：提取具体数值而非简单描述
- 复杂财务计算支持：复投、现金跑道、增长预测等
- 增强的Claude理解和回答生成
- 智能查询路由和API组合
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import traceback
import re

# 🎯 核心组件导入
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
from utils.calculators.statistical_calculator import (
    UnifiedCalculator, create_unified_calculator, UnifiedCalculationResult,
    CalculationType
)

# 工具类导入
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


# ============= 数据类定义 =============

@dataclass
class BusinessInsight:
    """业务洞察类"""
    title: str
    summary: str
    confidence_score: float = 0.8
    insight_type: str = "general"
    recommendations: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.supporting_data is None:
            self.supporting_data = {}


class ProcessingStrategy(Enum):
    """处理策略"""
    SIMPLE_DATA = "simple_data"
    DATA_WITH_CALC = "data_with_calc"
    COMPREHENSIVE = "comprehensive"
    QUICK_RESPONSE = "quick_response"
    ERROR_HANDLING = "error_handling"


class DataExtractionStrategy(Enum):
    """数据提取策略"""
    FINANCIAL_OVERVIEW = "financial_overview"
    EXPIRY_ANALYSIS = "expiry_analysis"
    DAILY_ANALYSIS = "daily_analysis"
    REINVESTMENT_ANALYSIS = "reinvestment_analysis"
    TREND_ANALYSIS = "trend_analysis"
    USER_ANALYSIS = "user_analysis"
    CASH_FLOW_ANALYSIS = "cash_flow_analysis"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ProcessingResult:
    """处理结果"""
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


# ============= 主编排器类 =============

class IntelligentQAOrchestrator:
    """🧠 智能问答编排器 - 智能增强版"""

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
            return

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

        # 🆕 智能提取缓存
        self.extraction_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("智能增强版 IntelligentQAOrchestrator 创建完成")

    def _initialize_component_placeholders(self):
        """组件初始化"""
        # 核心组件
        self.query_parser: Optional[SmartQueryParser] = None
        self.data_fetcher: Optional[SmartDataFetcher] = None
        self.statistical_calculator: Optional[UnifiedCalculator] = None

        # 工具组件
        self.date_utils: Optional[DateUtils] = None
        self.financial_formatter: Optional[FinancialFormatter] = None
        self.chart_generator: Optional[ChartGenerator] = None
        self.report_generator: Optional[ReportGenerator] = None
        self.conversation_manager: Optional[ConversationManager] = None

    def _load_orchestrator_config(self) -> Dict[str, Any]:
        """加载配置"""
        cfg = {
            'max_processing_time': getattr(self.app_config, 'MAX_PROCESSING_TIME', 120),
            'enable_intelligent_caching': getattr(self.app_config, 'ENABLE_INTELLIGENT_CACHING', True),
            'min_confidence_threshold': getattr(self.app_config, 'MIN_CONFIDENCE_THRESHOLD', 0.6),
            'cache_ttl_seconds': getattr(self.app_config, 'CACHE_TTL', 1800),
            'claude_timeout': getattr(self.app_config, 'CLAUDE_TIMEOUT', 60),
            'gpt_timeout': getattr(self.app_config, 'GPT_TIMEOUT', 40),
            'DEBUG': getattr(self.app_config, 'DEBUG', False),
            'version': getattr(self.app_config, 'VERSION', '3.0.0-intelligent')
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
        """智能初始化"""
        if self.initialized:
            logger.debug("Orchestrator already initialized.")
            return

        logger.info("🚀 开始初始化智能增强版编排器...")
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

            # 核心组件初始化
            self.query_parser = create_smart_query_parser(self.claude_client, self.gpt_client)

            fetcher_config = self.config.get('api_connector_config', {})
            self.data_fetcher = create_smart_data_fetcher(self.claude_client, self.gpt_client, fetcher_config)

            # 🆕 使用统一计算器
            self.statistical_calculator = create_unified_calculator(self.gpt_client, precision=6)

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
            logger.info(f"✅ 智能增强版编排器初始化完成 (耗时: {init_duration:.2f}s)")

        except Exception as e:
            self.initialized = False
            logger.error(f"❌ 智能编排器初始化失败: {str(e)}\n{traceback.format_exc()}")

    # ================================================================
    # 🎯 核心方法：智能查询处理
    # ================================================================

    async def process_intelligent_query(self, user_query: str, user_id: int = 0,
                                        conversation_id: Optional[str] = None,
                                        preferences: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """🎯 智能查询处理 - 增强版"""

        if not self.initialized:
            await self.initialize()
            if not self.initialized:
                return self._create_error_result("系统初始化失败", user_query)

        session_id = str(uuid.uuid4())
        query_id = f"q_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{hashlib.md5(user_query.encode('utf-8')).hexdigest()[:6]}"
        start_time = time.time()

        logger.info(f"🎯 QueryID: {query_id} - 开始智能处理: '{user_query[:50]}...'")
        self.orchestrator_stats['total_queries'] += 1

        # 处理对话ID
        conversation_id_for_db = self._parse_conversation_id(conversation_id)

        # 保存用户输入
        user_message_saved = await self._save_user_message_if_needed(
            conversation_id_for_db, user_query, query_id)

        try:
            # 🚀 智能快速响应检测
            quick_response = await self._try_intelligent_quick_response(user_query, query_id)
            if quick_response:
                return await self._build_quick_response_result(
                    quick_response, session_id, query_id, conversation_id, start_time, user_message_saved, conversation_id_for_db)

            timing = {}

            # 1️⃣ 智能查询理解
            start_t = time.time()
            query_analysis = await self._intelligent_query_understanding(user_query, conversation_id_for_db)
            timing['parsing'] = time.time() - start_t

            # 2️⃣ 智能数据获取
            start_t = time.time()
            data_result = await self._intelligent_data_acquisition(query_analysis)
            timing['data_fetching'] = time.time() - start_t

            # 3️⃣ 智能数据提取
            start_t = time.time()
            extracted_data = await self._intelligent_data_extraction(data_result, user_query, query_analysis)
            timing['data_extraction'] = time.time() - start_t

            # 4️⃣ 智能计算处理
            start_t = time.time()
            calculation_result = None
            if query_analysis.needs_calculation:
                calculation_result = await self._intelligent_calculation_processing(
                    query_analysis, extracted_data, user_query)
            timing['calculation'] = time.time() - start_t

            # 5️⃣ 智能回答生成
            start_t = time.time()
            response_text = await self._intelligent_response_generation(
                user_query, query_analysis, extracted_data, calculation_result)
            timing['response_generation'] = time.time() - start_t

            # 6️⃣ 智能洞察生成
            start_t = time.time()
            insights = await self._intelligent_insights_generation(
                extracted_data, calculation_result, query_analysis, user_query)
            timing['insights'] = time.time() - start_t

            # 7️⃣ 智能可视化生成
            start_t = time.time()
            visualizations = await self._intelligent_visualization_generation(
                extracted_data, calculation_result, query_analysis)
            timing['visualization'] = time.time() - start_t

            # 构建智能结果
            total_processing_time = time.time() - start_time
            confidence = self._calculate_intelligent_confidence(
                query_analysis, extracted_data, calculation_result, insights)

            result = ProcessingResult(
                session_id=session_id,
                query_id=query_id,
                success=True,
                response_text=response_text,
                insights=insights,
                key_metrics=self._extract_intelligent_metrics(extracted_data, calculation_result),
                visualizations=visualizations,
                processing_strategy=self._determine_intelligent_strategy(query_analysis),
                processors_used=self._get_intelligent_processors_used(query_analysis, calculation_result),
                ai_collaboration_summary=self._get_intelligent_ai_summary(timing),
                confidence_score=confidence,
                data_quality_score=self._calculate_data_quality_score(extracted_data),
                response_completeness=self._calculate_intelligent_completeness(
                    extracted_data, calculation_result, insights),
                total_processing_time=total_processing_time,
                ai_processing_time=timing.get('parsing', 0) + timing.get('response_generation', 0),
                data_fetching_time=timing.get('data_fetching', 0),
                processing_metadata={
                    'query_complexity': query_analysis.complexity.value,
                    'query_type': query_analysis.query_type.value,
                    'extraction_strategy': extracted_data.get('extraction_strategy', 'unknown'),
                    'step_times': timing,
                    'intelligent_architecture': True
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

            logger.info(f"✅ QueryID: {query_id} - 智能处理成功，耗时: {total_processing_time:.2f}s")
            return result

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"❌ QueryID: {query_id} - 智能处理失败: {str(e)}\n{traceback.format_exc()}")
            return await self._handle_error(session_id, query_id, user_query, str(e), error_time, conversation_id)

    # ================================================================
    # 🧠 智能处理方法
    # ================================================================

    async def _try_intelligent_quick_response(self, user_query: str, query_id: str) -> Optional[Dict[str, Any]]:
        """🚀 智能快速响应 - 增强版"""
        try:
            query_lower = user_query.lower()

            # 🧠 智能模式检测 - 更精确的模式匹配
            quick_patterns = self._detect_intelligent_quick_patterns(query_lower, user_query)

            if not quick_patterns:
                return None

            logger.info(f"⚡ 检测到智能快速模式: {quick_patterns['pattern_type']}")

            # 🚀 执行智能API调用
            start_fetch_time = time.time()
            api_result = await self._execute_intelligent_quick_api(quick_patterns)
            fetch_time = time.time() - start_fetch_time

            if not api_result or not api_result.get('success'):
                logger.warning(f"快速API调用失败，回退到完整流程")
                return None

            # 🎯 智能数据提取和格式化
            intelligent_response = await self._format_intelligent_quick_response(
                api_result, quick_patterns, user_query)

            return {
                'response': intelligent_response['text'],
                'metrics': intelligent_response.get('metrics', {}),
                'insights': intelligent_response.get('insights', []),
                'data_quality': api_result.get('validation', {}).get('confidence', 0.9),
                'fetch_time': fetch_time,
                'api_method': quick_patterns['api_method'],
                'pattern_type': quick_patterns['pattern_type']
            }

        except Exception as e:
            logger.error(f"智能快速响应处理失败: {e}")
            return None

    def _detect_intelligent_quick_patterns(self, query_lower: str, original_query: str) -> Optional[Dict[str, Any]]:
        """🧠 智能快速模式检测 - 大幅增强"""

        # 🎯 智能日期提取
        dates_info = self._extract_dates_from_query(original_query)

        patterns = [
            # 💰 财务概览查询
            {
                'pattern_type': 'financial_overview',
                'keywords': ['总资金', '总余额', '资金', '余额', '活跃会员', '用户数'],
                'exclude_keywords': ['入金', '出金', '到期', '趋势'],
                'api_method': 'get_system_data',
                'extraction_strategy': DataExtractionStrategy.FINANCIAL_OVERVIEW,
                'description': '财务概览查询'
            },

            # 📅 特定日期查询
            {
                'pattern_type': 'specific_date_query',
                'date_required': True,
                'keywords': ['入金', '出金', '注册', '数据'],
                'api_method': 'get_daily_data',
                'extraction_strategy': DataExtractionStrategy.DAILY_ANALYSIS,
                'description': '特定日期数据查询'
            },

            # ⏰ 到期产品查询
            {
                'pattern_type': 'expiry_query',
                'keywords': ['到期', '过期', '产品到期'],
                'exclude_keywords': ['复投', '计算', '预计', '趋势'],
                'api_method': 'auto_detect_expiry',  # 根据日期自动选择
                'extraction_strategy': DataExtractionStrategy.EXPIRY_ANALYSIS,
                'description': '产品到期查询'
            },

            # 👥 用户分析查询
            {
                'pattern_type': 'user_analysis',
                'keywords': ['用户', '会员', '注册', '活跃'],
                'exclude_keywords': ['到期', '入金', '出金'],
                'api_method': 'get_system_data',  # 从系统数据获取用户统计
                'extraction_strategy': DataExtractionStrategy.USER_ANALYSIS,
                'description': '用户分析查询'
            }
        ]

        # 🔍 智能匹配
        for pattern in patterns:
            if self._match_intelligent_pattern(query_lower, pattern, dates_info):
                matched_pattern = pattern.copy()
                matched_pattern.update({
                    'dates_info': dates_info,
                    'original_query': original_query,
                    'params': self._extract_pattern_params(query_lower, pattern, dates_info)
                })
                return matched_pattern

        return None

    def _extract_dates_from_query(self, query: str) -> Dict[str, Any]:
        """🧠 智能日期提取"""
        import re
        from datetime import datetime

        dates_info = {
            'has_dates': False,
            'date_type': 'none',  # single, range, relative
            'dates': [],
            'api_format_dates': []
        }

        # 🎯 多种日期格式匹配
        patterns = [
            # X月X日格式
            (r'(\d{1,2})月(\d{1,2})[日号]', 'chinese_month_day'),
            # YYYYMMDD格式
            (r'(\d{4})(\d{2})(\d{2})', 'yyyymmdd'),
            # YYYY-MM-DD格式
            (r'(\d{4})-(\d{2})-(\d{2})', 'yyyy_mm_dd'),
            # 相对日期
            (r'今[天日]|当[天日]', 'today'),
            (r'明[天日]', 'tomorrow'),
            (r'本周', 'this_week'),
            (r'上周|上个星期', 'last_week')
        ]

        current_year = datetime.now().year

        for pattern, date_type in patterns:
            matches = re.findall(pattern, query)

            if date_type == 'chinese_month_day':
                for match in matches:
                    month, day = int(match[0]), int(match[1])
                    try:
                        date_obj = datetime(current_year, month, day)
                        api_format = date_obj.strftime('%Y%m%d')
                        dates_info['dates'].append(date_obj)
                        dates_info['api_format_dates'].append(api_format)
                        dates_info['has_dates'] = True
                    except ValueError:
                        continue

            elif date_type == 'yyyymmdd':
                for match in matches:
                    try:
                        date_str = ''.join(match)
                        date_obj = datetime.strptime(date_str, '%Y%m%d')
                        dates_info['dates'].append(date_obj)
                        dates_info['api_format_dates'].append(date_str)
                        dates_info['has_dates'] = True
                    except ValueError:
                        continue

            elif date_type == 'today':
                today = datetime.now()
                dates_info['dates'].append(today)
                dates_info['api_format_dates'].append(today.strftime('%Y%m%d'))
                dates_info['has_dates'] = True

            # 可以继续添加其他日期类型的处理...

        # 🎯 判断日期类型
        if len(dates_info['dates']) == 1:
            dates_info['date_type'] = 'single'
        elif len(dates_info['dates']) == 2:
            dates_info['date_type'] = 'range'
        elif len(dates_info['dates']) > 2:
            dates_info['date_type'] = 'multiple'

        return dates_info

    def _match_intelligent_pattern(self, query_lower: str, pattern: Dict[str, Any],
                                   dates_info: Dict[str, Any]) -> bool:
        """智能模式匹配"""

        # 检查日期要求
        if pattern.get('date_required', False) and not dates_info['has_dates']:
            return False

        # 检查排除关键词
        if 'exclude_keywords' in pattern:
            if any(exc in query_lower for exc in pattern['exclude_keywords']):
                return False

        # 检查必须关键词
        if 'keywords' in pattern:
            if not any(kw in query_lower for kw in pattern['keywords']):
                return False

        return True

    def _extract_pattern_params(self, query_lower: str, pattern: Dict[str, Any],
                                dates_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取模式参数"""
        params = {}

        # 🎯 根据不同模式提取参数
        if pattern['pattern_type'] == 'specific_date_query':
            if dates_info['has_dates'] and dates_info['api_format_dates']:
                params['date'] = dates_info['api_format_dates'][0]

        elif pattern['pattern_type'] == 'expiry_query':
            if dates_info['date_type'] == 'single' and dates_info['api_format_dates']:
                params['date'] = dates_info['api_format_dates'][0]
                pattern['api_method'] = 'get_product_end_data'
            elif dates_info['date_type'] == 'range' and len(dates_info['api_format_dates']) >= 2:
                params['start_date'] = dates_info['api_format_dates'][0]
                params['end_date'] = dates_info['api_format_dates'][1]
                pattern['api_method'] = 'get_product_end_interval'
            elif '今' in query_lower or '今天' in query_lower:
                pattern['api_method'] = 'get_expiring_products_today'
            elif '本周' in query_lower:
                pattern['api_method'] = 'get_expiring_products_week'

        return params

    async def _execute_intelligent_quick_api(self, pattern_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """🚀 执行智能快速API调用"""
        try:
            api_method = pattern_info['api_method']
            params = pattern_info['params']

            if not self.data_fetcher or not self.data_fetcher.api_connector:
                logger.error("API连接器不可用")
                return None

            api_connector = self.data_fetcher.api_connector

            logger.info(f"🚀 执行智能快速API: {api_method}, 参数: {params}")

            # 🎯 智能API路由
            if api_method == 'get_daily_data':
                if 'date' in params:
                    result = await api_connector.get_daily_data(params['date'])
                else:
                    result = await api_connector.get_daily_data()

            elif api_method == 'get_system_data':
                result = await api_connector.get_system_data()

            elif api_method == 'get_expiring_products_today':
                result = await api_connector.get_expiring_products_today()

            elif api_method == 'get_expiring_products_week':
                result = await api_connector.get_expiring_products_week()

            elif api_method == 'get_product_end_data':
                if 'date' in params:
                    result = await api_connector.get_product_end_data(params['date'])
                else:
                    logger.error("get_product_end_data 需要date参数")
                    return None

            elif api_method == 'get_product_end_interval':
                if 'start_date' in params and 'end_date' in params:
                    result = await api_connector.get_product_end_interval(
                        params['start_date'], params['end_date'])
                else:
                    logger.error("get_product_end_interval 需要start_date和end_date参数")
                    return None

            else:
                logger.error(f"未知的API方法: {api_method}")
                return None

            logger.info(f"🔍 API调用结果: success={result.get('success')}")
            return result

        except Exception as e:
            logger.error(f"智能快速API调用失败: {e}")
            return None

    async def _format_intelligent_quick_response(self, api_result: Dict[str, Any],
                                                 pattern_info: Dict[str, Any],
                                                 original_query: str) -> Dict[str, Any]:
        """🎯 智能快速响应格式化"""
        try:
            data = api_result.get('data', {})
            if not data:
                return {
                    'text': f"获取到{pattern_info['description']}，但数据为空。",
                    'metrics': {}
                }

            extraction_strategy = pattern_info.get('extraction_strategy')

            # 🧠 根据提取策略智能格式化
            if extraction_strategy == DataExtractionStrategy.FINANCIAL_OVERVIEW:
                return await self._format_financial_overview_response(data, original_query)
            elif extraction_strategy == DataExtractionStrategy.DAILY_ANALYSIS:
                return await self._format_daily_analysis_response(data, original_query, pattern_info)
            elif extraction_strategy == DataExtractionStrategy.EXPIRY_ANALYSIS:
                return await self._format_expiry_analysis_response(data, original_query, pattern_info)
            elif extraction_strategy == DataExtractionStrategy.USER_ANALYSIS:
                return await self._format_user_analysis_response(data, original_query)
            else:
                return {
                    'text': f"已获取{pattern_info['description']}数据。",
                    'metrics': self._extract_basic_metrics(data)
                }

        except Exception as e:
            logger.error(f"智能响应格式化失败: {e}")
            return {
                'text': f"数据获取成功，但格式化时遇到问题：{str(e)}",
                'metrics': {}
            }

    async def _format_financial_overview_response(self, data: Dict[str, Any],
                                                  query: str) -> Dict[str, Any]:
        """💰 智能财务概览响应格式化"""
        try:
            # 🎯 智能数据提取
            total_balance = float(data.get('总余额', 0))
            total_inflow = float(data.get('总入金', 0))
            total_outflow = float(data.get('总出金', 0))
            total_investment = float(data.get('总投资金额', 0))

            user_stats = data.get('用户统计', {})
            total_users = int(user_stats.get('总用户数', 0))
            active_users = int(user_stats.get('活跃用户数', 0))

            # 🧠 根据查询内容智能选择回答重点
            query_lower = query.lower()
            response_parts = ["💰 财务概览："]

            if '活跃会员' in query_lower or '活跃用户' in query_lower:
                response_parts.append(f"🔥 活跃会员：{active_users:,}人")
                if total_users > 0:
                    activity_rate = (active_users / total_users) * 100
                    response_parts.append(f"📊 活跃率：{activity_rate:.1f}%")
                response_parts.append(f"👥 总用户：{total_users:,}人")

            elif '总资金' in query_lower or '余额' in query_lower:
                if self.financial_formatter:
                    response_parts.append(f"💵 总余额：{self.financial_formatter.format_currency(total_balance)}")
                else:
                    response_parts.append(f"💵 总余额：¥{total_balance:,.2f}")

                # 💡 智能洞察
                net_flow = total_inflow - total_outflow
                if net_flow > 0:
                    if self.financial_formatter:
                        response_parts.append(f"📈 净流入：{self.financial_formatter.format_currency(net_flow)}")
                    else:
                        response_parts.append(f"📈 净流入：¥{net_flow:,.2f}")
                    response_parts.append("💪 资金状况良好")

            else:
                # 综合展示
                if self.financial_formatter:
                    response_parts.extend([
                        f"💵 总余额：{self.financial_formatter.format_currency(total_balance)}",
                        f"👥 活跃会员：{active_users:,}人（总用户：{total_users:,}人）"
                    ])
                else:
                    response_parts.extend([
                        f"💵 总余额：¥{total_balance:,.2f}",
                        f"👥 活跃会员：{active_users:,}人（总用户：{total_users:,}人）"
                    ])

            # 🎯 智能洞察生成
            insights = []
            if total_users > 0:
                activity_rate = active_users / total_users
                if activity_rate > 0.8:
                    insights.append("用户活跃度很高，平台吸引力强")
                elif activity_rate < 0.3:
                    insights.append("用户活跃度较低，建议加强用户运营")

            if total_balance > 0 and total_inflow > 0:
                balance_ratio = total_balance / total_inflow
                if balance_ratio > 0.8:
                    insights.append("资金留存率高，用户信任度良好")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '总余额': total_balance,
                    '总入金': total_inflow,
                    '总出金': total_outflow,
                    '活跃用户数': active_users,
                    '总用户数': total_users,
                    '活跃率': (active_users / total_users) if total_users > 0 else 0
                },
                'insights': insights
            }

        except Exception as e:
            logger.error(f"财务概览格式化失败: {e}")
            return {
                'text': f"财务数据获取成功，原始数据：{str(data)[:200]}...",
                'metrics': {}
            }

    async def _format_daily_analysis_response(self, data: Dict[str, Any], query: str,
                                              pattern_info: Dict[str, Any]) -> Dict[str, Any]:
        """📅 智能每日数据响应格式化"""
        try:
            date = data.get('日期', pattern_info.get('params', {}).get('date', '未知日期'))
            inflow = float(data.get('入金', 0))
            outflow = float(data.get('出金', 0))
            registrations = int(data.get('注册人数', 0))
            purchases = int(data.get('购买产品数量', 0))
            holdings = int(data.get('持仓人数', 0))

            query_lower = query.lower()
            response_parts = [f"📅 {date} 数据分析："]

            # 🧠 智能重点突出
            if '入金' in query_lower:
                if self.financial_formatter:
                    response_parts.append(f"💰 入金：{self.financial_formatter.format_currency(inflow)}")
                else:
                    response_parts.append(f"💰 入金：¥{inflow:,.2f}")

                if inflow > 100000:  # 10万以上
                    response_parts.append("📈 入金表现优秀")
                elif inflow > 50000:  # 5万以上
                    response_parts.append("✅ 入金表现良好")

            elif '出金' in query_lower:
                if self.financial_formatter:
                    response_parts.append(f"💸 出金：{self.financial_formatter.format_currency(outflow)}")
                else:
                    response_parts.append(f"💸 出金：¥{outflow:,.2f}")

            elif '注册' in query_lower:
                response_parts.append(f"👥 新增注册：{registrations}人")
                if registrations > 100:
                    response_parts.append("🚀 注册量表现出色")

            else:
                # 完整展示
                if self.financial_formatter:
                    response_parts.extend([
                        f"💰 入金：{self.financial_formatter.format_currency(inflow)}",
                        f"💸 出金：{self.financial_formatter.format_currency(outflow)}",
                        f"👥 新增注册：{registrations}人",
                        f"🛍️ 产品购买：{purchases}笔"
                    ])
                else:
                    response_parts.extend([
                        f"💰 入金：¥{inflow:,.2f}",
                        f"💸 出金：¥{outflow:,.2f}",
                        f"👥 新增注册：{registrations}人",
                        f"🛍️ 产品购买：{purchases}笔"
                    ])

            # 🎯 智能分析
            net_flow = inflow - outflow
            insights = []

            if net_flow > 0:
                if self.financial_formatter:
                    response_parts.append(f"📈 净流入：{self.financial_formatter.format_currency(net_flow)}")
                else:
                    response_parts.append(f"📈 净流入：¥{net_flow:,.2f}")
                insights.append("资金净流入，表现积极")
            elif net_flow < 0:
                if self.financial_formatter:
                    response_parts.append(f"📉 净流出：{self.financial_formatter.format_currency(abs(net_flow))}")
                else:
                    response_parts.append(f"📉 净流出：¥{abs(net_flow):,.2f}")

            if registrations > 0 and purchases > 0:
                conversion_rate = purchases / registrations
                if conversion_rate > 0.5:
                    insights.append("注册转化率高，产品吸引力强")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '入金': inflow,
                    '出金': outflow,
                    '净流入': net_flow,
                    '注册人数': registrations,
                    '购买产品数量': purchases,
                    '持仓人数': holdings
                },
                'insights': insights
            }

        except Exception as e:
            logger.error(f"每日数据格式化失败: {e}")
            return {
                'text': f"每日数据获取成功，原始数据：{str(data)[:200]}...",
                'metrics': {}
            }

    async def _format_expiry_analysis_response(self, data: Dict[str, Any], query: str,
                                               pattern_info: Dict[str, Any]) -> Dict[str, Any]:
        """⏰ 智能到期数据响应格式化"""
        try:
            # 🧠 智能识别数据结构
            if 'interval_stats' in data:
                # 区间到期数据
                return await self._format_interval_expiry_response(data, query, pattern_info)
            else:
                # 单日到期数据
                return await self._format_single_day_expiry_response(data, query, pattern_info)

        except Exception as e:
            logger.error(f"到期数据格式化失败: {e}")
            return {
                'text': f"到期数据获取成功，原始数据：{str(data)[:200]}...",
                'metrics': {}
            }

    async def _format_interval_expiry_response(self, data: Dict[str, Any], query: str,
                                               pattern_info: Dict[str, Any]) -> Dict[str, Any]:
        """📊 区间到期数据格式化"""
        try:
            date_range = data.get('日期', '未知时间范围')
            total_count = int(data.get('到期数量', 0))
            total_amount = float(data.get('到期金额', 0))

            interval_stats = data.get('interval_stats', {})
            total_days = interval_stats.get('total_days', 1)

            response_parts = [f"📊 {date_range} 产品到期分析："]

            # 💰 核心数据展示
            if self.financial_formatter:
                response_parts.extend([
                    f"💰 总到期金额：{self.financial_formatter.format_currency(total_amount)}",
                    f"📦 总到期数量：{total_count:,}笔"
                ])
            else:
                response_parts.extend([
                    f"💰 总到期金额：¥{total_amount:,.2f}",
                    f"📦 总到期数量：{total_count:,}笔"
                ])

            # 📈 统计分析
            if total_count > 0:
                avg_amount = total_amount / total_count
                if self.financial_formatter:
                    response_parts.append(f"📊 平均金额：{self.financial_formatter.format_currency(avg_amount)}")
                else:
                    response_parts.append(f"📊 平均金额：¥{avg_amount:,.2f}")

            if total_days > 1:
                daily_avg_amount = total_amount / total_days
                daily_avg_count = total_count / total_days
                if self.financial_formatter:
                    response_parts.append(f"📅 日均到期：{self.financial_formatter.format_currency(daily_avg_amount)}（{daily_avg_count:.1f}笔）")
                else:
                    response_parts.append(f"📅 日均到期：¥{daily_avg_amount:,.2f}（{daily_avg_count:.1f}笔）")

            # 🔍 产品详情
            product_list = data.get('产品列表', [])
            if product_list:
                response_parts.append(f"\n🔍 主要产品（前3名）：")
                sorted_products = sorted(product_list, key=lambda x: float(x.get('到期金额', 0)), reverse=True)

                for i, product in enumerate(sorted_products[:3], 1):
                    name = product.get('产品名称', f'产品{i}')
                    amount = float(product.get('到期金额', 0))
                    if amount > 0:
                        if self.financial_formatter:
                            response_parts.append(f"  {i}. {name}：{self.financial_formatter.format_currency(amount)}")
                        else:
                            response_parts.append(f"  {i}. {name}：¥{amount:,.2f}")

            # 🎯 智能洞察
            insights = []
            if total_amount > 5000000:  # 500万以上
                insights.append("大额到期预警：需要充足的流动性准备")
            elif total_amount > 1000000:  # 100万以上
                insights.append("中等规模到期：建议提前准备资金")

            if total_days > 1:
                concentration = (max([float(p.get('到期金额', 0)) for p in product_list[:5]]) / total_amount) if product_list else 0
                if concentration > 0.3:
                    insights.append("到期集中度较高，存在流动性风险")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '到期金额': total_amount,
                    '到期数量': total_count,
                    '分析天数': total_days,
                    '日均金额': total_amount / total_days if total_days > 0 else 0,
                    '日均笔数': total_count / total_days if total_days > 0 else 0
                },
                'insights': insights
            }

        except Exception as e:
            logger.error(f"区间到期数据格式化失败: {e}")
            return {
                'text': f"区间到期数据处理出错：{str(e)}",
                'metrics': {}
            }

    async def _format_single_day_expiry_response(self, data: Dict[str, Any], query: str,
                                                 pattern_info: Dict[str, Any]) -> Dict[str, Any]:
        """📅 单日到期数据格式化"""
        try:
            date = data.get('日期', '未知日期')
            expiry_count = int(data.get('到期数量', 0))
            expiry_amount = float(data.get('到期金额', 0))

            response_parts = [f"⏰ {date} 产品到期情况："]

            if self.financial_formatter:
                response_parts.extend([
                    f"💰 到期金额：{self.financial_formatter.format_currency(expiry_amount)}",
                    f"📦 到期数量：{expiry_count}笔"
                ])
            else:
                response_parts.extend([
                    f"💰 到期金额：¥{expiry_amount:,.2f}",
                    f"📦 到期数量：{expiry_count}笔"
                ])

            if expiry_count > 0:
                avg_amount = expiry_amount / expiry_count
                if self.financial_formatter:
                    response_parts.append(f"📊 平均金额：{self.financial_formatter.format_currency(avg_amount)}")
                else:
                    response_parts.append(f"📊 平均金额：¥{avg_amount:,.2f}")

            # 🎯 智能评估
            insights = []
            if expiry_amount > 1000000:  # 100万以上
                insights.append("高额到期提醒：请确保流动性充足")
            elif expiry_amount == 0:
                insights.append("当日无产品到期")
            else:
                insights.append("到期金额在正常范围内")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '到期数量': expiry_count,
                    '到期金额': expiry_amount,
                    '平均金额': expiry_amount / expiry_count if expiry_count > 0 else 0
                },
                'insights': insights
            }

        except Exception as e:
            logger.error(f"单日到期数据格式化失败: {e}")
            return {
                'text': f"单日到期数据处理出错：{str(e)}",
                'metrics': {}
            }

    async def _format_user_analysis_response(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """👥 智能用户分析响应格式化"""
        try:
            user_stats = data.get('用户统计', {})
            total_users = int(user_stats.get('总用户数', 0))
            active_users = int(user_stats.get('活跃用户数', 0))

            response_parts = ["👥 用户分析："]

            # 🧠 智能重点分析
            query_lower = query.lower()

            if '活跃' in query_lower:
                response_parts.append(f"🔥 活跃用户：{active_users:,}人")
                if total_users > 0:
                    activity_rate = (active_users / total_users) * 100
                    response_parts.append(f"📊 活跃率：{activity_rate:.1f}%")

                    if activity_rate > 80:
                        response_parts.append("💪 活跃度表现优秀")
                    elif activity_rate > 60:
                        response_parts.append("✅ 活跃度表现良好")
                    else:
                        response_parts.append("⚠️ 活跃度有待提升")

            response_parts.append(f"👥 总用户数：{total_users:,}人")

            # 🎯 智能洞察
            insights = []
            if total_users > 0:
                activity_rate = active_users / total_users
                if activity_rate > 0.8:
                    insights.append("用户活跃度很高，平台粘性强")
                elif activity_rate > 0.6:
                    insights.append("用户活跃度良好，继续保持")
                elif activity_rate > 0.4:
                    insights.append("用户活跃度一般，建议加强用户运营")
                else:
                    insights.append("用户活跃度较低，需要重点关注用户留存")

                if total_users > 10000:
                    insights.append("用户规模已达到一定体量")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '总用户数': total_users,
                    '活跃用户数': active_users,
                    '活跃率': (active_users / total_users) if total_users > 0 else 0
                },
                'insights': insights
            }

        except Exception as e:
            logger.error(f"用户分析格式化失败: {e}")
            return {
                'text': f"用户数据获取成功，原始数据：{str(data)[:200]}...",
                'metrics': {}
            }

    def _extract_basic_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取基础指标"""
        metrics = {}

        # 尝试提取常见的数值字段
        common_fields = ['总余额', '入金', '出金', '到期金额', '到期数量', '注册人数', '总用户数', '活跃用户数']

        for field in common_fields:
            if field in data:
                try:
                    metrics[field] = float(data[field])
                except (ValueError, TypeError):
                    pass

        return metrics

    async def _build_quick_response_result(self, quick_response: Dict[str, Any], session_id: str,
                                           query_id: str, conversation_id: Optional[str],
                                           start_time: float, user_message_saved: bool,
                                           conversation_id_for_db: Optional[int]) -> ProcessingResult:
        """构建快速响应结果"""
        total_processing_time = time.time() - start_time

        # 🧠 转换洞察格式
        insights = []
        for insight_text in quick_response.get('insights', []):
            insights.append(BusinessInsight(
                title="快速分析洞察",
                summary=insight_text,
                confidence_score=0.9,
                insight_type="quick_analysis"
            ))

        result = ProcessingResult(
            session_id=session_id,
            query_id=query_id,
            success=True,
            response_text=quick_response['response'],
            insights=insights,
            key_metrics=quick_response.get('metrics', {}),
            processing_strategy=ProcessingStrategy.QUICK_RESPONSE,
            processors_used=["QuickResponse", "APIConnector", "IntelligentExtractor"],
            confidence_score=0.95,
            data_quality_score=quick_response.get('data_quality', 0.9),
            response_completeness=1.0,
            total_processing_time=total_processing_time,
            ai_processing_time=0.0,
            data_fetching_time=quick_response.get('fetch_time', 0.0),
            processing_metadata={
                'query_type': 'intelligent_quick_query',
                'api_used': quick_response.get('api_method', ''),
                'pattern_type': quick_response.get('pattern_type', ''),
                'quick_response': True
            },
            conversation_id=conversation_id
        )

        # 更新统计
        self._update_stats(result)

        # 保存AI响应
        if conversation_id_for_db and user_message_saved:
            await self._save_ai_response_if_needed(conversation_id_for_db, result, query_id)

        logger.info(f"⚡ QueryID: {query_id} - 智能快速处理完成，耗时: {total_processing_time:.2f}s")
        return result

    async def _intelligent_query_understanding(self, user_query: str,
                                               conversation_id_for_db: Optional[int]) -> QueryAnalysisResult:
        """🧠 智能查询理解 - 增强版"""
        logger.debug(f"🧠 智能查询理解: {user_query[:50]}...")

        # 获取对话上下文
        context = {}
        if conversation_id_for_db and self.conversation_manager:
            try:
                context = self.conversation_manager.get_context(conversation_id_for_db)
            except Exception as e:
                logger.warning(f"获取对话上下文失败: {e}")

        # 🧠 增强的Claude理解提示
        enhanced_prompt = f"""
        作为专业的金融AI系统，请深度分析这个查询并制定精确的执行计划：

        用户查询: "{user_query}"
        对话历史: {json.dumps(context or {}, ensure_ascii=False)[:300]}

        特别注意识别：
        1. 📅 时间信息：
           - 具体日期（如"6月1日"、"5月28日"）
           - 日期区间（如"6月1日至6月30日"）
           - 相对时间（如"今天"、"本周"、"上个星期"）

        2. 💰 复投/投资分析：
           - 复投比例（如"25%"、"一半"、"50%"、"百分之五十"）
           - 提现分析（如"百分之五十提现"）
           - 资金运营计算（如"公司还能运行多久"）

        3. 📊 数据需求：
           - 到期产品数据（包含利息信息）
           - 每日入金/出金数据
           - 用户/会员数据
           - 系统财务概览

        4. 🧮 计算需求：
           - 复投场景分析
           - 现金跑道计算
           - 趋势增长分析
           - 利息收益计算
           - 资金预测

        请返回详细的JSON执行计划，必须包含：
        {{
            "query_understanding": {{
                "complexity": "simple|medium|complex|expert",
                "query_type": "data_retrieval|calculation|trend_analysis|prediction|reinvestment_analysis|cash_flow_analysis",
                "business_scenario": "financial_overview|daily_operations|expiry_management|reinvestment_planning|risk_assessment",
                "user_intent": "用户具体想了解什么",
                "confidence": 0.9,
                "key_entities": {{
                    "dates": ["2024-06-01"],
                    "amounts": ["25%"],
                    "products": [],
                    "calculations": ["reinvestment"]
                }}
            }},
            "execution_plan": {{
                "api_calls": [
                    {{
                        "api_method": "get_product_end_interval",
                        "params": {{"start_date": "20240601", "end_date": "20240630"}},
                        "reason": "获取6月份产品到期数据",
                        "priority": 1
                    }}
                ],
                "needs_calculation": true,
                "calculation_type": "reinvestment_analysis|cash_runway|trend_analysis|compound_interest|financial_ratios",
                "calculation_params": {{
                    "reinvest_rate": 0.25,
                    "analysis_period": 30
                }},
                "calculation_description": "需要GPT计算的具体内容"
            }},
            "time_analysis": {{
                "has_time_requirement": true,
                "start_date": "20240601",
                "end_date": "20240630",
                "time_description": "6月1日至6月30日"
            }}
        }}

        🎯 API方法映射：
        - get_system_data(): 系统概览（余额、用户统计）
        - get_daily_data(date): 特定日期数据（入金、出金、注册）
        - get_product_end_data(date): 单日到期产品
        - get_product_end_interval(start_date, end_date): 区间到期产品
        - get_product_data(): 产品详情和持有情况
        - get_user_daily_data(date): 用户每日统计
        - get_user_data(page): 详细用户数据

        🧮 计算类型说明：
        - reinvestment_analysis: 复投分析（包含复投率、提现金额计算）
        - cash_runway: 现金跑道分析（公司能运行多久）
        - trend_analysis: 趋势分析（增长率、变化趋势）
        - compound_interest: 复利计算（利息累积）
        - financial_ratios: 财务比率分析
        - growth_prediction: 增长预测
        - withdrawal_analysis: 提现分析

        特别注意：
        - 日期格式必须是YYYYMMDD（如：20240601）
        - 复投比例转换为小数（如：25% → 0.25）
        - 复杂查询需要多个API组合
        - 计算参数要精确提取
        """

        # 调用Claude进行智能分析
        try:
            result = await asyncio.wait_for(
                self.claude_client.analyze_complex_query(enhanced_prompt, {
                    "query": user_query,
                    "context": context,
                    "current_date": datetime.now().strftime("%Y-%m-%d")
                }),
                timeout=30.0
            )

            if result.get("success"):
                analysis_text = result.get("analysis", "{}")
                analysis = self._extract_json_from_response(analysis_text)

                if analysis and self._validate_claude_response(analysis):
                    query_analysis = self._build_enhanced_analysis_result(user_query, analysis)
                    logger.info(f"🧠 智能查询理解成功: {query_analysis.query_type.value}")
                    return query_analysis

        except Exception as e:
            logger.error(f"智能查询理解失败: {e}")

        # 降级到基础解析
        logger.warning("Claude理解失败，使用智能降级解析")
        return await self._intelligent_fallback_analysis(user_query)

    def _build_enhanced_analysis_result(self, original_query: str, claude_plan: Dict[str, Any]) -> QueryAnalysisResult:
        """构建增强分析结果"""
        try:
            understanding = claude_plan["query_understanding"]
            execution = claude_plan["execution_plan"]
            time_info = claude_plan.get("time_analysis", {})

            # 🎯 增强的API调用参数处理
            api_calls = []
            for api_call in execution.get("api_calls", []):
                method = api_call.get("api_method", "get_system_data")
                params = api_call.get("params", {})

                # 🧠 智能参数处理和验证
                params = self._process_enhanced_api_params(params)

                api_calls.append({
                    "method": method,
                    "params": params,
                    "reason": api_call.get("reason", "数据获取"),
                    "priority": api_call.get("priority", 1)
                })

            # 🧠 智能时间范围处理
            time_range = None
            if time_info.get("has_time_requirement"):
                time_range = {
                    "start_date": time_info.get("start_date"),
                    "end_date": time_info.get("end_date"),
                    "description": time_info.get("time_description", "")
                }

            # 🧠 智能计算参数提取
            calculation_params = execution.get("calculation_params", {})

            # 安全地获取枚举值
            complexity = self._safe_get_enum(QueryComplexity, understanding.get("complexity", "medium"))
            query_type = self._safe_get_enum(QueryType, understanding.get("query_type", "data_retrieval"))
            business_scenario = self._safe_get_enum(BusinessScenario,
                                                    understanding.get("business_scenario", "daily_operations"))

            return QueryAnalysisResult(
                original_query=original_query,
                complexity=complexity,
                query_type=query_type,
                business_scenario=business_scenario,
                confidence_score=float(understanding.get("confidence", 0.85)),

                # 🧠 增强的执行信息
                api_calls_needed=api_calls,
                needs_calculation=execution.get("needs_calculation", False),
                calculation_type=execution.get("calculation_type") if execution.get("needs_calculation") else None,

                # 时间信息
                time_range=time_range,

                processing_metadata={
                    "user_intent": understanding.get("user_intent", ""),
                    "key_entities": understanding.get("key_entities", {}),
                    "api_count": len(api_calls),
                    "processing_method": "intelligent_claude_enhanced",
                    "calculation_description": execution.get("calculation_description", ""),
                    "calculation_params": calculation_params
                }
            )
        except Exception as e:
            logger.error(f"构建增强分析结果失败: {e}\n{traceback.format_exc()}")
            # 返回一个基本的结果
            return QueryAnalysisResult(
                original_query=original_query,
                complexity=QueryComplexity.MEDIUM,
                query_type=QueryType.DATA_RETRIEVAL,
                business_scenario=BusinessScenario.DAILY_OPERATIONS,
                confidence_score=0.5,
                api_calls_needed=[{"method": "get_system_data", "params": {}, "reason": "降级数据获取"}],
                processing_metadata={"error": str(e), "fallback": True}
            )

    def _process_enhanced_api_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """🧠 增强的API参数处理"""
        processed_params = params.copy()

        # 🎯 智能日期格式验证和转换
        date_fields = ["date", "start_date", "end_date"]
        for field in date_fields:
            if field in processed_params and processed_params[field]:
                processed_date = self._intelligent_date_conversion(processed_params[field])
                if processed_date:
                    processed_params[field] = processed_date
                else:
                    logger.warning(f"日期参数 {field}={processed_params[field]} 转换失败")

        return processed_params

    def _intelligent_date_conversion(self, date_str: str) -> Optional[str]:
        """🧠 智能日期转换"""
        if not date_str:
            return datetime.now().strftime('%Y%m%d')

        try:
            # 如果已经是YYYYMMDD格式
            if len(date_str) == 8 and date_str.isdigit():
                # 验证日期有效性
                datetime.strptime(date_str, '%Y%m%d')
                return date_str

            # 如果是YYYY-MM-DD格式
            if len(date_str) == 10 and '-' in date_str:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return dt.strftime('%Y%m%d')

            # 如果使用date_utils进行智能转换
            if self.date_utils:
                try:
                    parsed_date = self.date_utils.api_format_to_date(date_str)
                    return parsed_date.strftime('%Y%m%d')
                except:
                    pass

            # 默认返回今天
            logger.warning(f"无法解析日期格式: {date_str}, 使用今天日期")
            return datetime.now().strftime('%Y%m%d')

        except Exception as e:
            logger.error(f"智能日期转换失败: {date_str}, 错误: {e}")
            return datetime.now().strftime('%Y%m%d')

    async def _intelligent_fallback_analysis(self, query: str) -> QueryAnalysisResult:
        """🧠 智能降级分析 - 增强版"""
        logger.info(f"执行智能降级解析: {query[:50]}...")

        query_lower = query.lower()

        # 🧠 智能模式匹配
        if self._detect_reinvestment_pattern(query_lower):
            return self._build_reinvestment_analysis_result(query)
        elif self._detect_cash_runway_pattern(query_lower):
            return self._build_cash_runway_analysis_result(query)
        elif self._detect_expiry_pattern(query_lower):
            return self._build_expiry_analysis_result(query)
        elif self._detect_trend_pattern(query_lower):
            return self._build_trend_analysis_result(query)
        elif self._detect_daily_pattern(query_lower):
            return self._build_daily_analysis_result(query)
        else:
            return self._build_default_analysis_result(query)

    def _detect_reinvestment_pattern(self, query_lower: str) -> bool:
        """检测复投模式"""
        reinvestment_keywords = ['复投', '再投资', '25%', '50%', '一半', '百分之', '提现', '剩余资金']
        return any(kw in query_lower for kw in reinvestment_keywords)

    def _detect_cash_runway_pattern(self, query_lower: str) -> bool:
        """检测现金跑道模式"""
        runway_keywords = ['还能运行', '运行多久', '资金耗尽', '现金流', '没入金']
        return any(kw in query_lower for kw in runway_keywords)

    def _detect_expiry_pattern(self, query_lower: str) -> bool:
        """检测到期模式"""
        expiry_keywords = ['到期', '过期', '产品到期']
        return any(kw in query_lower for kw in expiry_keywords)

    def _detect_trend_pattern(self, query_lower: str) -> bool:
        """检测趋势模式"""
        trend_keywords = ['趋势', '增长', '变化', '平均', '历史']
        return any(kw in query_lower for kw in trend_keywords)

    def _detect_daily_pattern(self, query_lower: str) -> bool:
        """检测每日数据模式"""
        daily_keywords = ['入金', '出金', '注册', '月', '日']
        return any(kw in query_lower for kw in daily_keywords)

    def _build_reinvestment_analysis_result(self, query: str) -> QueryAnalysisResult:
        """构建复投分析结果"""
        # 🧠 智能提取复投比例
        reinvest_rate = self._extract_reinvestment_rate(query)

        # 🧠 智能提取日期范围
        dates_info = self._extract_dates_from_query(query)

        api_calls = [{"method": "get_system_data", "params": {}, "reason": "获取当前资金状况"}]

        if dates_info['has_dates'] and dates_info['date_type'] == 'range':
            api_calls.append({
                "method": "get_product_end_interval",
                "params": {
                    "start_date": dates_info['api_format_dates'][0],
                    "end_date": dates_info['api_format_dates'][1]
                },
                "reason": "获取区间到期数据"
            })
        elif dates_info['has_dates']:
            api_calls.append({
                "method": "get_product_end_data",
                "params": {"date": dates_info['api_format_dates'][0]},
                "reason": "获取到期数据"
            })

        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.COMPLEX,
            query_type=QueryType.CALCULATION,
            business_scenario=BusinessScenario.FUTURE_PROJECTION,
            confidence_score=0.8,
            api_calls_needed=api_calls,
            needs_calculation=True,
            calculation_type="reinvestment_analysis",
            processing_metadata={
                "parsing_method": "intelligent_fallback_reinvestment",
                "reinvestment_rate": reinvest_rate,
                "extracted_dates": dates_info
            }
        )

    def _extract_reinvestment_rate(self, query: str) -> float:
        """🧠 智能提取复投比例"""
        import re

        # 匹配百分比
        percent_patterns = [
            r'(\d+)%',
            r'百分之(\d+)',
            r'(\d+)成'
        ]

        for pattern in percent_patterns:
            matches = re.findall(pattern, query)
            if matches:
                try:
                    rate = float(matches[0]) / 100
                    return min(rate, 1.0)  # 确保不超过100%
                except ValueError:
                    continue

        # 匹配文字描述
        if '一半' in query or '50%' in query:
            return 0.5
        elif '四分之一' in query or '25%' in query:
            return 0.25
        elif '三分之一' in query:
            return 0.33
        elif '四分之三' in query or '75%' in query:
            return 0.75

        # 默认50%
        return 0.5

    def _build_cash_runway_analysis_result(self, query: str) -> QueryAnalysisResult:
        """构建现金跑道分析结果"""
        api_calls = [
            {"method": "get_system_data", "params": {}, "reason": "获取当前资金状况"},
            {"method": "get_daily_data", "params": {}, "reason": "获取最近支出数据"}
        ]

        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.COMPLEX,
            query_type=QueryType.PREDICTION,
            business_scenario=BusinessScenario.RISK_MANAGEMENT,
            confidence_score=0.8,
            api_calls_needed=api_calls,
            needs_calculation=True,
            calculation_type="cash_runway",
            processing_metadata={
                "parsing_method": "intelligent_fallback_cash_runway",
                "analysis_type": "financial_sustainability"
            }
        )

    def _build_expiry_analysis_result(self, query: str) -> QueryAnalysisResult:
        """构建到期分析结果"""
        dates_info = self._extract_dates_from_query(query)

        if dates_info['date_type'] == 'range' and len(dates_info['api_format_dates']) >= 2:
            api_calls = [{
                "method": "get_product_end_interval",
                "params": {
                    "start_date": dates_info['api_format_dates'][0],
                    "end_date": dates_info['api_format_dates'][1]
                },
                "reason": "获取区间到期数据"
            }]
        elif dates_info['has_dates']:
            api_calls = [{
                "method": "get_product_end_data",
                "params": {"date": dates_info['api_format_dates'][0]},
                "reason": "获取单日到期数据"
            }]
        else:
            api_calls = [{"method": "get_expiring_products_today", "params": {}, "reason": "获取今日到期数据"}]

        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.MEDIUM,
            query_type=QueryType.DATA_RETRIEVAL,
            business_scenario=BusinessScenario.DAILY_OPERATIONS,
            confidence_score=0.8,
            api_calls_needed=api_calls,
            processing_metadata={
                "parsing_method": "intelligent_fallback_expiry",
                "extracted_dates": dates_info
            }
        )

    def _build_trend_analysis_result(self, query: str) -> QueryAnalysisResult:
        """构建趋势分析结果"""
        api_calls = [
            {"method": "get_system_data", "params": {}, "reason": "获取当前状态"},
            {"method": "get_daily_data", "params": {}, "reason": "获取历史数据"}
        ]

        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.COMPLEX,
            query_type=QueryType.TREND_ANALYSIS,
            business_scenario=BusinessScenario.HISTORICAL_PERFORMANCE,
            confidence_score=0.75,
            api_calls_needed=api_calls,
            needs_calculation=True,
            calculation_type="trend_analysis",
            processing_metadata={
                "parsing_method": "intelligent_fallback_trend",
                "analysis_scope": "historical_performance"
            }
        )

    def _build_daily_analysis_result(self, query: str) -> QueryAnalysisResult:
        """构建每日分析结果"""
        dates_info = self._extract_dates_from_query(query)

        if dates_info['has_dates']:
            api_calls = [{
                "method": "get_daily_data",
                "params": {"date": dates_info['api_format_dates'][0]},
                "reason": "获取特定日期数据"
            }]
        else:
            api_calls = [{"method": "get_daily_data", "params": {}, "reason": "获取最新每日数据"}]

        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.SIMPLE,
            query_type=QueryType.DATA_RETRIEVAL,
            business_scenario=BusinessScenario.DAILY_OPERATIONS,
            confidence_score=0.8,
            api_calls_needed=api_calls,
            processing_metadata={
                "parsing_method": "intelligent_fallback_daily",
                "extracted_dates": dates_info
            }
        )

    def _build_default_analysis_result(self, query: str) -> QueryAnalysisResult:
        """构建默认分析结果"""
        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.SIMPLE,
            query_type=QueryType.DATA_RETRIEVAL,
            business_scenario=BusinessScenario.DAILY_OPERATIONS,
            confidence_score=0.6,
            api_calls_needed=[{"method": "get_system_data", "params": {}, "reason": "获取系统概览"}],
            processing_metadata={
                "parsing_method": "intelligent_fallback_default",
                "note": "使用默认数据获取策略"
            }
        )

    async def _intelligent_data_acquisition(self, query_analysis: QueryAnalysisResult) -> Optional[FetcherExecutionResult]:
        """🧠 智能数据获取"""
        logger.debug(f"📊 执行智能数据获取")

        if not self.data_fetcher:
            raise RuntimeError("SmartDataFetcher 未初始化")

        try:
            # 🧠 使用增强版智能数据获取
            if hasattr(self.data_fetcher.api_connector, 'intelligent_data_fetch_enhanced'):
                execution_result = await self.data_fetcher.api_connector.intelligent_data_fetch_enhanced(
                    query_analysis.to_dict()
                )

                # 🎯 转换增强数据为标准ExecutionResult格式
                if execution_result.get("success"):
                    # 创建模拟的ExecutionResult对象
                    mock_result = type('ExecutionResult', (), {
                        'success': True,
                        'execution_status': FetcherExecutionStatus.COMPLETED,
                        'confidence_level': FetcherDataQualityLevel.HIGH,
                        'processed_data': execution_result.get("organized_data", {}),
                        'fetched_data': execution_result.get("organized_data", {}),
                        'processing_metadata': execution_result.get("execution_summary", {})
                    })()

                    return mock_result
                else:
                    return None
            else:
                # 降级到原有逻辑
                execution_result = await self._legacy_data_fetch(query_analysis)
                return execution_result

        except Exception as e:
            logger.error(f"智能数据获取失败: {e}")
            return None

    async def _intelligent_data_extraction(self, data_result: Optional[FetcherExecutionResult],
                                           user_query: str,
                                           query_analysis: QueryAnalysisResult) -> Dict[str, Any]:
        """🧠 智能数据提取 - 核心改进"""
        logger.debug(f"🧠 执行智能数据提取")

        if not data_result:
            return {"status": "无数据", "extraction_strategy": "none"}

        # 🎯 确定提取策略
        extraction_strategy = self._determine_extraction_strategy(user_query, query_analysis)

        extracted_data = {
            "status": "数据提取成功",
            "extraction_strategy": extraction_strategy.value,
            "data_quality": getattr(data_result, 'confidence_level', 0.8),
            "raw_data_available": True
        }

        # 🧠 获取处理后的数据
        if hasattr(data_result, 'processed_data') and data_result.processed_data:
            processed = data_result.processed_data

            # 🎯 根据策略进行智能提取
            if extraction_strategy == DataExtractionStrategy.FINANCIAL_OVERVIEW:
                extracted_data.update(await self._extract_financial_overview_data(processed, user_query))
            elif extraction_strategy == DataExtractionStrategy.EXPIRY_ANALYSIS:
                extracted_data.update(await self._extract_expiry_analysis_data(processed, user_query))
            elif extraction_strategy == DataExtractionStrategy.DAILY_ANALYSIS:
                extracted_data.update(await self._extract_daily_analysis_data(processed, user_query))
            elif extraction_strategy == DataExtractionStrategy.REINVESTMENT_ANALYSIS:
                extracted_data.update(await self._extract_reinvestment_analysis_data(processed, user_query, query_analysis))
            elif extraction_strategy == DataExtractionStrategy.TREND_ANALYSIS:
                extracted_data.update(await self._extract_trend_analysis_data(processed, user_query))
            elif extraction_strategy == DataExtractionStrategy.USER_ANALYSIS:
                extracted_data.update(await self._extract_user_analysis_data(processed, user_query))
            elif extraction_strategy == DataExtractionStrategy.CASH_FLOW_ANALYSIS:
                extracted_data.update(await self._extract_cash_flow_analysis_data(processed, user_query))
            else:
                # 综合提取
                extracted_data.update(await self._extract_comprehensive_data(processed, user_query))

        # 🧠 缓存提取结果
        cache_key = self._generate_extraction_cache_key(user_query, extraction_strategy)
        self.extraction_cache[cache_key] = {
            'data': extracted_data,
            'timestamp': time.time(),
            'strategy': extraction_strategy.value
        }

        logger.info(f"🧠 智能数据提取完成: {extraction_strategy.value}")
        return extracted_data

    def _determine_extraction_strategy(self, user_query: str,
                                       query_analysis: QueryAnalysisResult) -> DataExtractionStrategy:
        """🎯 确定数据提取策略"""
        query_lower = user_query.lower()

        # 🧠 智能策略匹配
        if any(kw in query_lower for kw in ["复投", "reinvest", "25%", "一半", "50%", "提现"]):
            return DataExtractionStrategy.REINVESTMENT_ANALYSIS
        elif any(kw in query_lower for kw in ["到期", "expiry", "product_end"]):
            return DataExtractionStrategy.EXPIRY_ANALYSIS
        elif any(kw in query_lower for kw in ["入金", "出金", "daily", "每日", "月", "日"]):
            return DataExtractionStrategy.DAILY_ANALYSIS
        elif any(kw in query_lower for kw in ["趋势", "增长", "变化", "trend", "历史"]):
            return DataExtractionStrategy.TREND_ANALYSIS
        elif any(kw in query_lower for kw in ["用户", "会员", "活跃", "注册"]):
            return DataExtractionStrategy.USER_ANALYSIS
        elif any(kw in query_lower for kw in ["现金流", "运行", "多久", "资金"]):
            return DataExtractionStrategy.CASH_FLOW_ANALYSIS
        elif any(kw in query_lower for kw in ["总资金", "余额", "概览"]):
            return DataExtractionStrategy.FINANCIAL_OVERVIEW
        else:
            return DataExtractionStrategy.COMPREHENSIVE

    async def _extract_financial_overview_data(self, processed_data: Dict[str, Any],
                                               user_query: str) -> Dict[str, Any]:
        """💰 智能提取财务概览数据"""
        extracted = {"financial_overview": {}}

        for key, data in processed_data.items():
            if isinstance(data, dict):
                # 🎯 系统财务数据
                if any(field in data for field in ["总余额", "总入金", "总出金"]):
                    financial_data = {
                        "total_balance": float(data.get("总余额", 0)),
                        "total_inflow": float(data.get("总入金", 0)),
                        "total_outflow": float(data.get("总出金", 0)),
                        "total_investment": float(data.get("总投资金额", 0)),
                        "total_rewards": float(data.get("总奖励发放", 0))
                    }

                    # 🧮 计算衍生指标
                    financial_data["net_flow"] = financial_data["total_inflow"] - financial_data["total_outflow"]
                    financial_data["outflow_ratio"] = (
                        financial_data["total_outflow"] / financial_data["total_inflow"]
                        if financial_data["total_inflow"] > 0 else 0
                    )
                    financial_data["balance_utilization"] = (
                        financial_data["total_investment"] / financial_data["total_balance"]
                        if financial_data["total_balance"] > 0 else 0
                    )

                    extracted["financial_overview"].update(financial_data)

                # 🎯 用户统计数据
                if "用户统计" in data:
                    user_stats = data["用户统计"]
                    user_data = {
                        "total_users": int(user_stats.get("总用户数", 0)),
                        "active_users": int(user_stats.get("活跃用户数", 0))
                    }

                    if user_data["total_users"] > 0:
                        user_data["activity_rate"] = user_data["active_users"] / user_data["total_users"]
                        user_data["avg_balance_per_user"] = (
                            extracted["financial_overview"].get("total_balance", 0) / user_data["total_users"]
                        )

                    extracted["financial_overview"].update(user_data)

        return extracted

    async def _extract_expiry_analysis_data(self, processed_data: Dict[str, Any],
                                            user_query: str) -> Dict[str, Any]:
        """⏰ 智能提取到期分析数据"""
        extracted = {"expiry_analysis": {}}

        for key, data in processed_data.items():
            if isinstance(data, dict) and ("到期" in str(data) or "expiry" in str(key).lower()):
                expiry_data = {
                    "expiry_amount": float(data.get("到期金额", 0)),
                    "expiry_count": int(data.get("到期数量", 0)),
                    "expiry_date": data.get("日期", ""),
                    "product_details": []
                }

                # 🎯 产品详情提取
                if "产品列表" in data:
                    for product in data["产品列表"]:
                        if isinstance(product, dict):
                            product_info = {
                                "name": product.get("产品名称", ""),
                                "amount": float(product.get("到期金额", 0)),
                                "count": int(product.get("到期数量", 0)),
                                "daily_rate": float(product.get("每日利率", 0)) / 100 if product.get("每日利率") else 0,
                                "period_days": int(product.get("期限天数", 0))
                            }

                            # 🧮 计算预期利息
                            if product_info["daily_rate"] > 0 and product_info["period_days"] > 0:
                                product_info["expected_interest"] = (
                                    product_info["amount"] * product_info["daily_rate"] * product_info["period_days"]
                                )

                            expiry_data["product_details"].append(product_info)

                # 🧮 计算区间统计（如果有）
                if "interval_stats" in data:
                    interval_stats = data["interval_stats"]
                    expiry_data.update({
                        "total_days": interval_stats.get("total_days", 1),
                        "daily_average_amount": interval_stats.get("daily_average_amount", 0),
                        "daily_average_quantity": interval_stats.get("daily_average_quantity", 0)
                    })

                extracted["expiry_analysis"].update(expiry_data)

        return extracted

    async def _extract_daily_analysis_data(self, processed_data: Dict[str, Any],
                                           user_query: str) -> Dict[str, Any]:
        """📅 智能提取每日分析数据"""
        extracted = {"daily_analysis": {}}

        for key, data in processed_data.items():
            if isinstance(data, dict):
                # 🎯 每日数据字段检测
                if any(field in data for field in ["入金", "出金", "注册人数", "日期"]):
                    daily_data = {
                        "date": data.get("日期", ""),
                        "inflow": float(data.get("入金", 0)),
                        "outflow": float(data.get("出金", 0)),
                        "registrations": int(data.get("注册人数", 0)),
                        "purchases": int(data.get("购买产品数量", 0)),
                        "holdings": int(data.get("持仓人数", 0)),
                        "expired_products": int(data.get("到期产品数量", 0))
                    }

                    # 🧮 计算每日衍生指标
                    daily_data["net_flow"] = daily_data["inflow"] - daily_data["outflow"]
                    daily_data["flow_ratio"] = (
                        daily_data["inflow"] / daily_data["outflow"]
                        if daily_data["outflow"] > 0 else float('inf')
                    )
                    daily_data["conversion_rate"] = (
                        daily_data["purchases"] / daily_data["registrations"]
                        if daily_data["registrations"] > 0 else 0
                    )
                    daily_data["activity_score"] = (
                        daily_data["purchases"] + daily_data["expired_products"]
                    ) / 2

                    # 🎯 日期元数据
                    if daily_data["date"] and self.date_utils:
                        try:
                            date_obj = datetime.strptime(daily_data["date"], "%Y%m%d")
                            daily_data["date_metadata"] = {
                                "weekday": date_obj.strftime("%A"),
                                "formatted_date": date_obj.strftime("%Y-%m-%d"),
                                "is_weekend": date_obj.weekday() >= 5
                            }
                        except:
                            pass

                    extracted["daily_analysis"].update(daily_data)

        return extracted

    async def _extract_reinvestment_analysis_data(self, processed_data: Dict[str, Any],
                                                  user_query: str,
                                                  query_analysis: QueryAnalysisResult) -> Dict[str, Any]:
        """💰 智能提取复投分析数据"""
        extracted = {"reinvestment_analysis": {}}

        # 🎯 从查询分析中获取复投参数
        reinvest_rate = query_analysis.processing_metadata.get("reinvestment_rate", 0.5)

        # 🧠 提取到期数据
        expiry_data = await self._extract_expiry_analysis_data(processed_data, user_query)
        if "expiry_analysis" in expiry_data:
            expiry_info = expiry_data["expiry_analysis"]

            extracted["reinvestment_analysis"].update({
                "base_expiry_amount": expiry_info.get("expiry_amount", 0),
                "base_expiry_count": expiry_info.get("expiry_count", 0),
                "reinvestment_rate": reinvest_rate,
                "product_details": expiry_info.get("product_details", [])
            })

        # 🧠 提取当前财务状况
        financial_data = await self._extract_financial_overview_data(processed_data, user_query)
        if "financial_overview" in financial_data:
            financial_info = financial_data["financial_overview"]

            extracted["reinvestment_analysis"].update({
                "current_balance": financial_info.get("total_balance", 0),
                "current_inflow": financial_info.get("total_inflow", 0),
                "current_outflow": financial_info.get("total_outflow", 0)
            })

        # 🧮 预计算复投指标
        base_amount = extracted["reinvestment_analysis"].get("base_expiry_amount", 0)
        if base_amount > 0:
            extracted["reinvestment_analysis"].update({
                "estimated_reinvest_amount": base_amount * reinvest_rate,
                "estimated_withdrawal_amount": base_amount * (1 - reinvest_rate),
                "liquidity_impact": base_amount * (1 - reinvest_rate)
            })

        return extracted

    async def _extract_trend_analysis_data(self, processed_data: Dict[str, Any],
                                           user_query: str) -> Dict[str, Any]:
        """📈 智能提取趋势分析数据"""
        extracted = {"trend_analysis": {}}

        # 🎯 查找时间序列数据
        time_series_data = {}

        for key, data in processed_data.items():
            if isinstance(data, dict):
                # 检测每日数据序列
                if "by_date" in data or "by_type" in data:
                    time_series_data[key] = data
                elif isinstance(data, list):
                    # 处理列表格式的时间序列
                    if all(isinstance(item, dict) and "日期" in item for item in data):
                        time_series_data[key] = data

        # 🧮 提取趋势指标
        if time_series_data:
            extracted["trend_analysis"]["time_series_available"] = True
            extracted["trend_analysis"]["data_sources"] = list(time_series_data.keys())

            # 🎯 提取数值序列用于计算
            numeric_series = {}
            for source_key, series_data in time_series_data.items():
                if isinstance(series_data, list):
                    # 从列表中提取数值
                    for field in ["入金", "出金", "注册人数"]:
                        values = []
                        for item in series_data:
                            if isinstance(item, dict) and field in item:
                                try:
                                    values.append(float(item[field]))
                                except (ValueError, TypeError):
                                    continue
                        if values and len(values) >= 3:  # 至少3个数据点
                            numeric_series[f"{source_key}_{field}"] = values

            extracted["trend_analysis"]["numeric_series"] = numeric_series
        else:
            extracted["trend_analysis"]["time_series_available"] = False

        return extracted

    async def _extract_user_analysis_data(self, processed_data: Dict[str, Any],
                                          user_query: str) -> Dict[str, Any]:
        """👥 智能提取用户分析数据"""
        extracted = {"user_analysis": {}}

        for key, data in processed_data.items():
            if isinstance(data, dict):
                # 🎯 用户统计数据
                if "用户统计" in data:
                    user_stats = data["用户统计"]
                    extracted["user_analysis"].update({
                        "total_users": int(user_stats.get("总用户数", 0)),
                        "active_users": int(user_stats.get("活跃用户数", 0))
                    })

                # 🎯 VIP分布数据
                vip_data = {}
                for i in range(11):  # VIP0-VIP10
                    vip_key = f"vip{i}的人数"
                    if vip_key in data:
                        vip_data[f"vip{i}"] = int(data.get(vip_key, 0))

                if vip_data:
                    total_vip_users = sum(vip_data.values())
                    vip_distribution = {}
                    for vip_level, count in vip_data.items():
                        vip_distribution[vip_level] = {
                            "count": count,
                            "percentage": (count / total_vip_users * 100) if total_vip_users > 0 else 0
                        }

                    extracted["user_analysis"]["vip_distribution"] = vip_distribution
                    extracted["user_analysis"]["total_vip_users"] = total_vip_users

                # 🎯 用户详细列表数据
                if "用户列表" in data:
                    user_list = data["用户列表"]
                    if isinstance(user_list, list):
                        user_investments = []
                        user_rewards = []
                        user_rois = []

                        for user in user_list:
                            if isinstance(user, dict):
                                investment = float(user.get("总投入", 0))
                                reward = float(user.get("累计获得奖励金额", 0))
                                roi = float(user.get("投报比", 0))

                                if investment > 0:
                                    user_investments.append(investment)
                                if reward > 0:
                                    user_rewards.append(reward)
                                if roi > 0:
                                    user_rois.append(roi)

                        if user_investments:
                            extracted["user_analysis"]["investment_stats"] = {
                                "total_users_with_investment": len(user_investments),
                                "avg_investment": sum(user_investments) / len(user_investments),
                                "total_investment": sum(user_investments),
                                "max_investment": max(user_investments),
                                "min_investment": min(user_investments)
                            }

                        if user_rois:
                            extracted["user_analysis"]["roi_stats"] = {
                                "avg_roi": sum(user_rois) / len(user_rois),
                                "max_roi": max(user_rois),
                                "min_roi": min(user_rois)
                            }

        # 🧮 计算用户衍生指标
        if "total_users" in extracted["user_analysis"] and "active_users" in extracted["user_analysis"]:
            total = extracted["user_analysis"]["total_users"]
            active = extracted["user_analysis"]["active_users"]

            if total > 0:
                extracted["user_analysis"]["activity_rate"] = active / total
                extracted["user_analysis"]["inactive_users"] = total - active

        return extracted

    async def _extract_cash_flow_analysis_data(self, processed_data: Dict[str, Any],
                                               user_query: str) -> Dict[str, Any]:
        """💸 智能提取现金流分析数据"""
        extracted = {"cash_flow_analysis": {}}

        # 🎯 提取财务数据
        financial_data = await self._extract_financial_overview_data(processed_data, user_query)
        if "financial_overview" in financial_data:
            financial_info = financial_data["financial_overview"]

            extracted["cash_flow_analysis"].update({
                "current_balance": financial_info.get("total_balance", 0),
                "total_inflow": financial_info.get("total_inflow", 0),
                "total_outflow": financial_info.get("total_outflow", 0),
                "net_flow": financial_info.get("net_flow", 0),
                "outflow_ratio": financial_info.get("outflow_ratio", 0)
            })

        # 🎯 提取每日现金流数据
        daily_data = await self._extract_daily_analysis_data(processed_data, user_query)
        if "daily_analysis" in daily_data:
            daily_info = daily_data["daily_analysis"]

            extracted["cash_flow_analysis"]["daily_flows"] = {
                "daily_inflow": daily_info.get("inflow", 0),
                "daily_outflow": daily_info.get("outflow", 0),
                "daily_net_flow": daily_info.get("net_flow", 0)
            }

        # 🧮 现金流健康度评估
        current_balance = extracted["cash_flow_analysis"].get("current_balance", 0)
        daily_outflow = extracted["cash_flow_analysis"].get("daily_flows", {}).get("daily_outflow", 0)

        if current_balance > 0 and daily_outflow > 0:
            runway_days = current_balance / daily_outflow
            extracted["cash_flow_analysis"]["estimated_runway_days"] = runway_days

            if runway_days > 365:
                extracted["cash_flow_analysis"]["liquidity_status"] = "excellent"
            elif runway_days > 180:
                extracted["cash_flow_analysis"]["liquidity_status"] = "good"
            elif runway_days > 90:
                extracted["cash_flow_analysis"]["liquidity_status"] = "moderate"
            else:
                extracted["cash_flow_analysis"]["liquidity_status"] = "concerning"

        return extracted

    async def _extract_comprehensive_data(self, processed_data: Dict[str, Any],
                                          user_query: str) -> Dict[str, Any]:
        """🎯 综合数据提取"""
        extracted = {"comprehensive_analysis": {}}

        # 🧠 并行提取各类数据
        extraction_tasks = [
            self._extract_financial_overview_data(processed_data, user_query),
            self._extract_daily_analysis_data(processed_data, user_query),
            self._extract_user_analysis_data(processed_data, user_query)
        ]

        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # 🎯 合并提取结果
        for result in results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                extracted["comprehensive_analysis"].update(result)

        return extracted

    def _generate_extraction_cache_key(self, user_query: str,
                                       extraction_strategy: DataExtractionStrategy) -> str:
        """生成提取缓存键"""
        key_data = f"{user_query}_{extraction_strategy.value}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _intelligent_calculation_processing(self, query_analysis: QueryAnalysisResult,
                                                  extracted_data: Dict[str, Any],
                                                  user_query: str) -> Optional[Dict[str, Any]]:
        """🧮 智能计算处理 - 增强版"""
        if not query_analysis.needs_calculation:
            return None

        logger.debug(f"🧮 执行智能计算: {query_analysis.calculation_type}")

        if not self.statistical_calculator:
            logger.error("统一计算器未初始化")
            return None

        try:
            # 🧠 智能计算参数提取
            calc_params = self._extract_intelligent_calculation_params(
                query_analysis, extracted_data, user_query)

            # 🧮 执行计算
            calc_result = await self.statistical_calculator.calculate(
                calculation_type=query_analysis.calculation_type,
                data=extracted_data,
                params=calc_params
            )

            return {
                'calculation_result': calc_result,
                'calculation_type': query_analysis.calculation_type,
                'success': calc_result.success if calc_result else False,
                'confidence': calc_result.confidence if calc_result else 0.5,
                'calculation_params': calc_params
            }

        except Exception as e:
            logger.error(f"智能计算执行失败: {e}")
            return {
                'calculation_result': None,
                'calculation_type': query_analysis.calculation_type,
                'success': False,
                'error': str(e)
            }

    def _extract_intelligent_calculation_params(self, query_analysis: QueryAnalysisResult,
                                                extracted_data: Dict[str, Any],
                                                user_query: str) -> Dict[str, Any]:
        """🧠 智能计算参数提取"""
        params = {}

        # 🎯 根据计算类型提取参数
        calc_type = query_analysis.calculation_type

        if calc_type == "reinvestment_analysis":
            reinvest_data = extracted_data.get("reinvestment_analysis", {})
            params.update({
                "reinvest_rate": reinvest_data.get("reinvestment_rate", 0.5),
                "base_amount": reinvest_data.get("base_expiry_amount", 0),
                "current_balance": reinvest_data.get("current_balance", 0)
            })

        elif calc_type == "cash_runway":
            cash_flow_data = extracted_data.get("cash_flow_analysis", {})
            params.update({
                "current_balance": cash_flow_data.get("current_balance", 0),
                "daily_outflow": cash_flow_data.get("daily_flows", {}).get("daily_outflow", 0)
            })

        elif calc_type == "trend_analysis":
            trend_data = extracted_data.get("trend_analysis", {})
            params.update({
                "time_series_data": trend_data.get("numeric_series", {}),
                "analysis_period": 30
            })

        elif calc_type == "compound_interest":
            # 🧠 从到期产品中提取利率信息
            expiry_data = extracted_data.get("expiry_analysis", {})
            product_details = expiry_data.get("product_details", [])

            if product_details:
                avg_daily_rate = sum(p.get("daily_rate", 0) for p in product_details) / len(product_details)
                params.update({
                    "principal": expiry_data.get("expiry_amount", 10000),
                    "rate": avg_daily_rate * 365,  # 年化利率
                    "periods": 365,  # 一年
                    "frequency": 365  # 每日复利
                })

        # 🎯 从查询分析元数据中获取额外参数
        metadata_params = query_analysis.processing_metadata.get("calculation_params", {})
        params.update(metadata_params)

        # 🎯 时间范围参数
        if query_analysis.time_range:
            params.update({
                "start_date": query_analysis.time_range.get("start_date"),
                "end_date": query_analysis.time_range.get("end_date")
            })

        return params

    async def _intelligent_response_generation(self, user_query: str,
                                               query_analysis: QueryAnalysisResult,
                                               extracted_data: Dict[str, Any],
                                               calculation_result: Optional[Dict[str, Any]]) -> str:
        """🧠 智能回答生成 - 增强版"""
        try:
            if not self.claude_client:
                return self._generate_intelligent_fallback_response(
                    user_query, extracted_data, calculation_result)

            # 🧠 智能数据摘要生成
            intelligent_data_summary = self._generate_intelligent_data_summary(
                extracted_data, user_query, query_analysis)

            intelligent_calc_summary = self._generate_intelligent_calc_summary(calculation_result)

            # 🧠 增强的Claude回答生成提示
            enhanced_response_prompt = f"""
            作为专业的金融AI助手，请基于以下分析结果为用户生成专业、准确、易懂的回答：

            用户查询："{user_query}"

            查询分析：
            - 查询类型：{query_analysis.query_type.value}
            - 业务场景：{query_analysis.business_scenario.value}
            - 复杂度：{query_analysis.complexity.value}

            智能数据摘要：
            {json.dumps(intelligent_data_summary, ensure_ascii=False, indent=2)}

            计算结果摘要：
            {json.dumps(intelligent_calc_summary, ensure_ascii=False, indent=2)}

            回答要求：
            1. 🎯 直接回答用户的具体问题
            2. 📊 使用具体数据支持答案
            3. 💡 提供专业的分析洞察
            4. 🚀 给出可行的建议（如果适用）
            5. 💰 涉及金额时使用易读的格式（如：1,000,000 或 100万）

            特别注意：
            - 如果是复投分析，详细说明复投金额、提现金额和影响
            - 如果是现金跑道，明确说明能运行的时间和风险提示
            - 如果是趋势分析，描述变化方向和关键指标
            - 如果是到期分析，说明到期时间、金额和产品分布

            回答风格：专业但易懂，数据精准，逻辑清晰。
            """

            # 调用Claude生成回答
            result = await asyncio.wait_for(
                self.claude_client.analyze_complex_query(enhanced_response_prompt, {
                    "user_query": user_query,
                    "data_summary": intelligent_data_summary,
                    "calculation_summary": intelligent_calc_summary
                }),
                timeout=45.0
            )

            if result.get("success"):
                response_text = result.get("analysis", "")

                # 🧠 后处理：添加格式化和验证
                formatted_response = self._post_process_claude_response(
                    response_text, extracted_data, calculation_result)

                return formatted_response
            else:
                logger.warning("Claude回答生成失败，使用智能降级回答")
                return self._generate_intelligent_fallback_response(
                    user_query, extracted_data, calculation_result)

        except asyncio.TimeoutError:
            logger.error("Claude回答生成超时")
            return self._generate_intelligent_fallback_response(
                user_query, extracted_data, calculation_result)
        except Exception as e:
            logger.error(f"智能回答生成失败: {e}")
            return self._generate_intelligent_fallback_response(
                user_query, extracted_data, calculation_result)

    def _generate_intelligent_data_summary(self, extracted_data: Dict[str, Any],
                                           user_query: str,
                                           query_analysis: QueryAnalysisResult) -> Dict[str, Any]:
        """🧠 生成智能数据摘要"""
        summary = {
            "extraction_strategy": extracted_data.get("extraction_strategy", "unknown"),
            "data_quality": extracted_data.get("data_quality", 0.8),
            "core_metrics": {}
        }

        # 🎯 根据提取策略生成不同的摘要
        strategy = extracted_data.get("extraction_strategy", "")

        if strategy == "financial_overview":
            financial_data = extracted_data.get("financial_overview", {})
            summary["core_metrics"] = {
                "总余额": financial_data.get("total_balance", 0),
                "净流入": financial_data.get("net_flow", 0),
                "活跃用户": financial_data.get("active_users", 0),
                "活跃率": financial_data.get("activity_rate", 0)
            }

        elif strategy == "expiry_analysis":
            expiry_data = extracted_data.get("expiry_analysis", {})
            summary["core_metrics"] = {
                "到期金额": expiry_data.get("expiry_amount", 0),
                "到期数量": expiry_data.get("expiry_count", 0),
                "产品种类": len(expiry_data.get("product_details", []))
            }

        elif strategy == "reinvestment_analysis":
            reinvest_data = extracted_data.get("reinvestment_analysis", {})
            summary["core_metrics"] = {
                "基础到期金额": reinvest_data.get("base_expiry_amount", 0),
                "复投比例": reinvest_data.get("reinvestment_rate", 0),
                "预计复投金额": reinvest_data.get("estimated_reinvest_amount", 0),
                "预计提现金额": reinvest_data.get("estimated_withdrawal_amount", 0)
            }

        elif strategy == "daily_analysis":
            daily_data = extracted_data.get("daily_analysis", {})
            summary["core_metrics"] = {
                "入金": daily_data.get("inflow", 0),
                "出金": daily_data.get("outflow", 0),
                "净流入": daily_data.get("net_flow", 0),
                "注册人数": daily_data.get("registrations", 0)
            }

        elif strategy == "cash_flow_analysis":
            cash_data = extracted_data.get("cash_flow_analysis", {})
            summary["core_metrics"] = {
                "当前余额": cash_data.get("current_balance", 0),
                "预计运行天数": cash_data.get("estimated_runway_days", 0),
                "流动性状态": cash_data.get("liquidity_status", "unknown")
            }

        # 🧠 添加上下文相关信息
        summary["query_context"] = {
            "mentions_reinvestment": any(kw in user_query.lower() for kw in ["复投", "再投资"]),
            "mentions_timeframe": query_analysis.time_range is not None,
            "calculation_needed": query_analysis.needs_calculation
        }

        return summary

    def _generate_intelligent_calc_summary(self, calculation_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """🧠 生成智能计算摘要"""
        if not calculation_result or not calculation_result.get('success'):
            return {"status": "无计算结果"}

        calc_res = calculation_result.get('calculation_result')
        if not calc_res:
            return {"status": "计算结果为空"}

        summary = {
            "status": "计算成功",
            "calculation_type": calculation_result.get('calculation_type'),
            "confidence": calculation_result.get('confidence', 0.5),
            "primary_result": getattr(calc_res, 'primary_result', None),
            "key_results": {}
        }

        # 🎯 根据计算类型提取关键结果
        calc_type = calculation_result.get('calculation_type', '')
        detailed_results = getattr(calc_res, 'detailed_results', {})

        if calc_type == "reinvestment_analysis":
            summary["key_results"] = {
                "复投金额": detailed_results.get("estimated_reinvest_amount", 0),
                "提现金额": detailed_results.get("estimated_withdrawal_amount", 0),
                "余额影响": detailed_results.get("final_balance_impact", 0)
            }

        elif calc_type == "cash_runway":
            runway_analysis = detailed_results.get("runway_analysis", {})
            summary["key_results"] = {
                "运行天数": runway_analysis.get("runway_days", 0),
                "运行月数": runway_analysis.get("runway_months", 0),
                "风险等级": runway_analysis.get("risk_level", "unknown")
            }

        elif calc_type == "trend_analysis":
            summary["key_results"] = {
                "趋势方向": "详见趋势分析结果",
                "分析指标数": len(detailed_results) if detailed_results else 0
            }

        return summary

    def _post_process_claude_response(self, response_text: str,
                                      extracted_data: Dict[str, Any],
                                      calculation_result: Optional[Dict[str, Any]]) -> str:
        """🧠 Claude回答后处理"""
        try:
            # 🎯 数值格式化
            if self.financial_formatter:
                # 查找并格式化数值
                import re

                # 匹配货币数值模式
                money_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'

                def format_money(match):
                    try:
                        amount = float(match.group(1).replace(',', ''))
                        return self.financial_formatter.format_currency(amount)
                    except:
                        return match.group(1)

                response_text = re.sub(money_pattern, format_money, response_text)

            # 🎯 添加数据来源声明（如果需要）
            if "基于系统数据分析" not in response_text:
                response_text += "\n\n📊 *基于最新系统数据分析*"

            return response_text

        except Exception as e:
            logger.error(f"回答后处理失败: {e}")
            return response_text

    def _generate_intelligent_fallback_response(self, user_query: str,
                                                extracted_data: Dict[str, Any],
                                                calculation_result: Optional[Dict[str, Any]]) -> str:
        """🧠 智能降级回答生成"""
        response_parts = ["基于系统分析：\n"]

        # 🎯 根据提取的数据类型生成回答
        strategy = extracted_data.get("extraction_strategy", "")

        if strategy == "financial_overview":
            financial_data = extracted_data.get("financial_overview", {})
            total_balance = financial_data.get("total_balance", 0)
            active_users = financial_data.get("active_users", 0)

            if self.financial_formatter:
                response_parts.append(f"💰 当前总余额：{self.financial_formatter.format_currency(total_balance)}")
            else:
                response_parts.append(f"💰 当前总余额：¥{total_balance:,.2f}")

            response_parts.append(f"👥 活跃用户：{active_users:,}人")

        elif strategy == "expiry_analysis":
            expiry_data = extracted_data.get("expiry_analysis", {})
            expiry_amount = expiry_data.get("expiry_amount", 0)
            expiry_count = expiry_data.get("expiry_count", 0)

            if self.financial_formatter:
                response_parts.append(f"⏰ 到期金额：{self.financial_formatter.format_currency(expiry_amount)}")
            else:
                response_parts.append(f"⏰ 到期金额：¥{expiry_amount:,.2f}")

            response_parts.append(f"📦 到期数量：{expiry_count}笔")

        elif strategy == "reinvestment_analysis":
            reinvest_data = extracted_data.get("reinvestment_analysis", {})
            reinvest_amount = reinvest_data.get("estimated_reinvest_amount", 0)
            withdraw_amount = reinvest_data.get("estimated_withdrawal_amount", 0)

            if self.financial_formatter:
                response_parts.extend([
                    f"💰 预计复投金额：{self.financial_formatter.format_currency(reinvest_amount)}",
                    f"💸 预计提现金额：{self.financial_formatter.format_currency(withdraw_amount)}"
                ])
            else:
                response_parts.extend([
                    f"💰 预计复投金额：¥{reinvest_amount:,.2f}",
                    f"💸 预计提现金额：¥{withdraw_amount:,.2f}"
                ])

        # 🎯 添加计算结果
        if calculation_result and calculation_result.get('success'):
            calc_res = calculation_result.get('calculation_result')
            calc_type = calculation_result.get('calculation_type', '')

            if calc_type == "cash_runway" and calc_res:
                detailed_results = getattr(calc_res, 'detailed_results', {})
                runway_analysis = detailed_results.get("runway_analysis", {})
                runway_days = runway_analysis.get("runway_days", 0)

                if runway_days > 0:
                    response_parts.append(f"\n📊 根据当前资金状况，预计可运行 {runway_days:.0f} 天")

        if len(response_parts) == 1:  # 只有开头
            response_parts.append("抱歉，暂时无法提供详细分析。请稍后重试或联系技术支持。")

        return "\n".join(response_parts)

    async def _intelligent_insights_generation(self, extracted_data: Dict[str, Any],
                                               calculation_result: Optional[Dict[str, Any]],
                                               query_analysis: QueryAnalysisResult,
                                               user_query: str) -> List[BusinessInsight]:
        """💡 智能洞察生成 - 增强版"""
        logger.debug("💡 生成智能业务洞察")

        insights = []

        try:
            # 🧠 根据数据类型生成针对性洞察
            strategy = extracted_data.get("extraction_strategy", "")

            if strategy == "financial_overview":
                insights.extend(await self._generate_financial_insights(extracted_data))
            elif strategy == "expiry_analysis":
                insights.extend(await self._generate_expiry_insights(extracted_data))
            elif strategy == "reinvestment_analysis":
                insights.extend(await self._generate_reinvestment_insights(extracted_data, calculation_result))
            elif strategy == "cash_flow_analysis":
                insights.extend(await self._generate_cash_flow_insights(extracted_data, calculation_result))
            elif strategy == "trend_analysis":
                insights.extend(await self._generate_trend_insights(extracted_data, calculation_result))

            # 🧠 基于计算结果生成额外洞察
            if calculation_result and calculation_result.get('success'):
                insights.extend(await self._generate_calculation_insights(calculation_result))

        except Exception as e:
            logger.error(f"智能洞察生成失败: {e}")
            insights.append(BusinessInsight(
                title="系统提示",
                summary="数据分析过程中遇到问题，请稍后重试",
                confidence_score=0.3,
                insight_type="system_message"
            ))

        return insights

    async def _generate_financial_insights(self, extracted_data: Dict[str, Any]) -> List[BusinessInsight]:
        """💰 生成财务洞察"""
        insights = []
        financial_data = extracted_data.get("financial_overview", {})

        total_balance = financial_data.get("total_balance", 0)
        net_flow = financial_data.get("net_flow", 0)
        outflow_ratio = financial_data.get("outflow_ratio", 0)
        activity_rate = financial_data.get("activity_rate", 0)

        # 🎯 现金流洞察
        if net_flow > 0:
            insights.append(BusinessInsight(
                title="正向现金流",
                summary=f"当前净流入良好，资金增长稳健",
                confidence_score=0.9,
                insight_type="cash_flow_positive",
                recommendations=["继续保持良好的资金管理", "考虑扩大业务规模"],
                supporting_data={"net_flow": net_flow}
            ))
        elif net_flow < 0:
            insights.append(BusinessInsight(
                title="现金流出警示",
                summary=f"当前存在资金净流出，需要关注现金流管理",
                confidence_score=0.9,
                insight_type="cash_flow_negative",
                recommendations=["分析支出结构", "制定现金流优化策略", "考虑增加收入来源"],
                supporting_data={"net_flow": net_flow}
            ))

        # 🎯 支出比例洞察
        if outflow_ratio > 0.8:
            insights.append(BusinessInsight(
                title="高支出比例预警",
                summary=f"支出占入金比例达 {outflow_ratio:.1%}，建议优化支出结构",
                confidence_score=0.85,
                insight_type="high_outflow_warning",
                recommendations=["审核大额支出项目", "制定支出控制措施", "提升资金使用效率"],
                supporting_data={"outflow_ratio": outflow_ratio}
            ))

        # 🎯 用户活跃度洞察
        if activity_rate > 0:
            if activity_rate > 0.8:
                insights.append(BusinessInsight(
                    title="用户活跃度优秀",
                    summary=f"用户活跃率达 {activity_rate:.1%}，用户粘性很强",
                    confidence_score=0.9,
                    insight_type="high_user_engagement",
                    recommendations=["维持现有用户体验", "考虑推出更多产品"],
                    supporting_data={"activity_rate": activity_rate}
                ))
            elif activity_rate < 0.4:
                insights.append(BusinessInsight(
                    title="用户活跃度待提升",
                    summary=f"用户活跃率仅 {activity_rate:.1%}，需要加强用户运营",
                    confidence_score=0.8,
                    insight_type="low_user_engagement",
                    recommendations=["分析用户流失原因", "优化产品体验", "加强用户互动"],
                    supporting_data={"activity_rate": activity_rate}
                ))

        return insights

    async def _generate_expiry_insights(self, extracted_data: Dict[str, Any]) -> List[BusinessInsight]:
        """⏰ 生成到期洞察"""
        insights = []
        expiry_data = extracted_data.get("expiry_analysis", {})

        expiry_amount = expiry_data.get("expiry_amount", 0)
        expiry_count = expiry_data.get("expiry_count", 0)
        product_details = expiry_data.get("product_details", [])

        # 🎯 到期规模评估
        if expiry_amount > 5000000:  # 500万以上
            insights.append(BusinessInsight(
                title="大额到期预警",
                summary=f"即将到期金额达 {expiry_amount:,.0f}，需要充足的流动性准备",
                confidence_score=0.95,
                insight_type="large_expiry_warning",
                recommendations=["确保充足的现金储备", "制定流动性管理计划", "考虑分期支付方案"],
                supporting_data={"expiry_amount": expiry_amount}
            ))
        elif expiry_amount > 1000000:  # 100万以上
            insights.append(BusinessInsight(
                title="中等规模到期提醒",
                summary=f"到期金额 {expiry_amount:,.0f}，建议提前准备资金",
                confidence_score=0.8,
                insight_type="medium_expiry_notice",
                recommendations=["检查现金流状况", "预留足够的支付资金"],
                supporting_data={"expiry_amount": expiry_amount}
            ))

        # 🎯 产品集中度分析
        if len(product_details) > 0:
            max_product_amount = max([p.get("amount", 0) for p in product_details])
            concentration_ratio = max_product_amount / expiry_amount if expiry_amount > 0 else 0

            if concentration_ratio > 0.4:
                insights.append(BusinessInsight(
                    title="产品集中度较高",
                    summary=f"单一产品占到期金额 {concentration_ratio:.1%}，存在集中度风险",
                    confidence_score=0.8,
                    insight_type="product_concentration_risk",
                    recommendations=["分散产品结构", "降低单一产品依赖", "优化产品组合"],
                    supporting_data={"concentration_ratio": concentration_ratio}
                ))

        return insights

    async def _generate_reinvestment_insights(self, extracted_data: Dict[str, Any],
                                              calculation_result: Optional[Dict[str, Any]]) -> List[BusinessInsight]:
        """💰 生成复投洞察"""
        insights = []
        reinvest_data = extracted_data.get("reinvestment_analysis", {})

        reinvest_rate = reinvest_data.get("reinvestment_rate", 0)
        estimated_withdraw = reinvest_data.get("estimated_withdrawal_amount", 0)
        current_balance = reinvest_data.get("current_balance", 0)

        # 🎯 复投策略评估
        if reinvest_rate > 0.7:
            insights.append(BusinessInsight(
                title="高复投率策略",
                summary=f"复投率 {reinvest_rate:.1%} 有利于资金增长，但需注意流动性",
                confidence_score=0.8,
                insight_type="high_reinvestment_strategy",
                recommendations=["保持适度现金储备", "监控流动性状况", "设置流动性预警"],
                supporting_data={"reinvest_rate": reinvest_rate}
            ))
        elif reinvest_rate < 0.3:
            insights.append(BusinessInsight(
                title="保守复投策略",
                summary=f"复投率 {reinvest_rate:.1%} 较为保守，流动性充足但增长有限",
                confidence_score=0.8,
                insight_type="conservative_reinvestment",
                recommendations=["评估提高复投比例的可能性", "优化资金利用效率"],
                supporting_data={"reinvest_rate": reinvest_rate}
            ))

        # 🎯 流动性影响评估
        if estimated_withdraw > 0 and current_balance > 0:
            withdraw_impact = estimated_withdraw / current_balance
            if withdraw_impact > 0.3:
                insights.append(BusinessInsight(
                    title="显著流动性影响",
                    summary=f"预计提现将影响 {withdraw_impact:.1%} 的资金，需要谨慎管理",
                    confidence_score=0.85,
                    insight_type="liquidity_impact_significant",
                    recommendations=["制定资金调配计划", "确保核心业务资金充足"],
                    supporting_data={"withdraw_impact": withdraw_impact}
                ))

        return insights

    async def _generate_cash_flow_insights(self, extracted_data: Dict[str, Any],
                                           calculation_result: Optional[Dict[str, Any]]) -> List[BusinessInsight]:
        """💸 生成现金流洞察"""
        insights = []
        cash_data = extracted_data.get("cash_flow_analysis", {})

        runway_days = cash_data.get("estimated_runway_days", 0)
        liquidity_status = cash_data.get("liquidity_status", "unknown")

        # 🎯 现金跑道评估
        if runway_days > 0:
            if runway_days < 30:
                insights.append(BusinessInsight(
                    title="现金跑道紧急预警",
                    summary=f"按当前支出水平，资金仅能维持 {runway_days:.0f} 天",
                    confidence_score=0.95,
                    insight_type="critical_cash_runway",
                    recommendations=["立即制定紧急资金计划", "削减非必要支出", "寻找快速融资渠道"],
                    supporting_data={"runway_days": runway_days}
                ))
            elif runway_days < 90:
                insights.append(BusinessInsight(
                    title="现金跑道预警",
                    summary=f"资金可维持约 {runway_days:.0f} 天，建议提前准备",
                    confidence_score=0.9,
                    insight_type="cash_runway_warning",
                    recommendations=["制定资金筹措计划", "优化现金流管理", "考虑延期支付安排"],
                    supporting_data={"runway_days": runway_days}
                ))
            elif runway_days > 365:
                insights.append(BusinessInsight(
                    title="充足的资金储备",
                    summary=f"资金可维持超过一年，流动性状况优秀",
                    confidence_score=0.9,
                    insight_type="excellent_liquidity",
                    recommendations=["考虑投资增值机会", "优化资金配置", "制定长期发展规划"],
                    supporting_data={"runway_days": runway_days}
                ))

        return insights

    async def _generate_trend_insights(self, extracted_data: Dict[str, Any],
                                       calculation_result: Optional[Dict[str, Any]]) -> List[BusinessInsight]:
        """📈 生成趋势洞察"""
        insights = []

        # 🎯 基于计算结果的趋势洞察
        if calculation_result and calculation_result.get('success'):
            calc_res = calculation_result.get('calculation_result')
            if calc_res and hasattr(calc_res, 'detailed_results'):
                detailed_results = getattr(calc_res, 'detailed_results', {})

                for metric_name, trend_data in detailed_results.items():
                    if isinstance(trend_data, dict) and 'trend_direction' in trend_data:
                        direction = trend_data.get('trend_direction', 'stable')
                        confidence = trend_data.get('confidence', 0.5)

                        if direction == "上升" and confidence > 0.7:
                            insights.append(BusinessInsight(
                                title=f"{metric_name}呈上升趋势",
                                summary=f"{metric_name}显示明显的上升趋势，发展态势良好",
                                confidence_score=confidence,
                                insight_type="positive_trend",
                                recommendations=["继续保持当前策略", "适度扩大规模"],
                                supporting_data=trend_data
                            ))
                        elif direction == "下降" and confidence > 0.7:
                            insights.append(BusinessInsight(
                                title=f"{metric_name}呈下降趋势",
                                summary=f"{metric_name}显示下降趋势，需要关注并采取措施",
                                confidence_score=confidence,
                                insight_type="negative_trend",
                                recommendations=["分析下降原因", "制定改进措施", "加强监控"],
                                supporting_data=trend_data
                            ))

        return insights

    async def _generate_calculation_insights(self, calculation_result: Dict[str, Any]) -> List[BusinessInsight]:
        """🧮 生成计算洞察"""
        insights = []

        calc_type = calculation_result.get('calculation_type', '')
        calc_res = calculation_result.get('calculation_result')

        if not calc_res:
            return insights

        if calc_type == "reinvestment_analysis":
            detailed_results = getattr(calc_res, 'detailed_results', {})
            balance_impact = detailed_results.get("final_balance_impact", 0)

            if balance_impact > 0:
                insights.append(BusinessInsight(
                    title="复投策略效果积极",
                    summary=f"当前复投策略将产生积极的资金增长效果",
                    confidence_score=calc_res.confidence,
                    insight_type="positive_reinvestment_impact",
                    recommendations=["保持当前复投比例", "监控市场变化"],
                    supporting_data=detailed_results
                ))

        elif calc_type == "cash_runway":
            detailed_results = getattr(calc_res, 'detailed_results', {})
            runway_analysis = detailed_results.get("runway_analysis", {})
            risk_level = runway_analysis.get("risk_level", "unknown")

            if risk_level == "critical":
                insights.append(BusinessInsight(
                    title="现金流风险临界",
                    summary="根据现金跑道分析，当前现金流状况需要立即关注",
                    confidence_score=calc_res.confidence,
                    insight_type="critical_cash_flow_risk",
                    recommendations=["立即制定应急预案", "寻求外部资金支持"],
                    supporting_data=runway_analysis
                ))

        return insights

    async def _intelligent_visualization_generation(self, extracted_data: Dict[str, Any],
                                                    calculation_result: Optional[Dict[str, Any]],
                                                    query_analysis: QueryAnalysisResult) -> List[Dict[str, Any]]:
        """🎨 智能可视化生成 - 增强版"""
        visualizations = []

        if not self.chart_generator:
            return visualizations

        try:
            strategy = extracted_data.get("extraction_strategy", "")

            # 🎯 根据数据类型生成智能图表
            if strategy == "financial_overview":
                visualizations.extend(await self._generate_financial_visualizations(extracted_data))
            elif strategy == "expiry_analysis":
                visualizations.extend(await self._generate_expiry_visualizations(extracted_data))
            elif strategy == "daily_analysis":
                visualizations.extend(await self._generate_daily_visualizations(extracted_data))
            elif strategy == "trend_analysis":
                visualizations.extend(await self._generate_trend_visualizations(extracted_data, calculation_result))
            elif strategy == "reinvestment_analysis":
                visualizations.extend(await self._generate_reinvestment_visualizations(extracted_data, calculation_result))

        except Exception as e:
            logger.error(f"智能可视化生成失败: {e}")

        return visualizations

    async def _generate_financial_visualizations(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """💰 生成财务可视化"""
        visualizations = []
        financial_data = extracted_data.get("financial_overview", {})

        # 🎯 资金概览饼图
        total_balance = financial_data.get("total_balance", 0)
        total_inflow = financial_data.get("total_inflow", 0)
        total_outflow = financial_data.get("total_outflow", 0)

        if total_inflow > 0 and total_outflow > 0:
            chart_data = {
                'labels': ['总余额', '历史入金', '历史出金'],
                'values': [total_balance, total_inflow, total_outflow],
                'colors': ['#4CAF50', '#2196F3', '#FF9800']
            }

            visualizations.append({
                'type': 'pie',
                'title': '资金概览分布',
                'data_payload': chart_data,
                'description': '当前资金状况的整体分布'
            })

        # 🎯 用户活跃度条形图
        total_users = financial_data.get("total_users", 0)
        active_users = financial_data.get("active_users", 0)

        if total_users > 0:
            inactive_users = total_users - active_users
            user_chart_data = {
                'labels': ['活跃用户', '非活跃用户'],
                'values': [active_users, inactive_users],
                'colors': ['#4CAF50', '#E0E0E0']
            }

            visualizations.append({
                'type': 'bar',
                'title': '用户活跃度分布',
                'data_payload': user_chart_data,
                'description': '用户活跃状况分析'
            })

        return visualizations

    async def _generate_expiry_visualizations(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """⏰ 生成到期可视化"""
        visualizations = []
        expiry_data = extracted_data.get("expiry_analysis", {})
        product_details = expiry_data.get("product_details", [])

        # 🎯 产品到期金额分布
        if len(product_details) > 0:
            # 按金额排序并取前5名
            sorted_products = sorted(product_details,
                                     key=lambda x: x.get("amount", 0), reverse=True)[:5]

            if sorted_products:
                chart_data = {
                    'labels': [p.get("name", f"产品{i+1}") for i, p in enumerate(sorted_products)],
                    'values': [p.get("amount", 0) for p in sorted_products]
                }

                visualizations.append({
                    'type': 'bar',
                    'title': '主要产品到期金额分布',
                    'data_payload': chart_data,
                    'description': '按到期金额排序的主要产品分布'
                })

        return visualizations

    async def _generate_reinvestment_visualizations(self, extracted_data: Dict[str, Any],
                                                    calculation_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """💰 生成复投可视化"""
        visualizations = []
        reinvest_data = extracted_data.get("reinvestment_analysis", {})

        # 🎯 复投分配饼图
        reinvest_amount = reinvest_data.get("estimated_reinvest_amount", 0)
        withdraw_amount = reinvest_data.get("estimated_withdrawal_amount", 0)

        if reinvest_amount > 0 or withdraw_amount > 0:
            chart_data = {
                'labels': ['复投金额', '提现金额'],
                'values': [reinvest_amount, withdraw_amount],
                'colors': ['#4CAF50', '#FF9800']
            }

            visualizations.append({
                'type': 'pie',
                'title': '复投资金分配',
                'data_payload': chart_data,
                'description': '复投策略下的资金分配情况'
            })

        return visualizations

    async def _generate_daily_visualizations(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """📅 生成每日数据可视化"""
        visualizations = []
        daily_data = extracted_data.get("daily_analysis", {})

        # 🎯 当日资金流向
        inflow = daily_data.get("inflow", 0)
        outflow = daily_data.get("outflow", 0)

        if inflow > 0 or outflow > 0:
            chart_data = {
                'labels': ['入金', '出金'],
                'values': [inflow, outflow],
                'colors': ['#4CAF50', '#F44336']
            }

            visualizations.append({
                'type': 'bar',
                'title': '当日资金流向',
                'data_payload': chart_data,
                'description': '当日入金和出金对比'
            })

        return visualizations

    async def _generate_trend_visualizations(self, extracted_data: Dict[str, Any],
                                             calculation_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """📈 生成趋势可视化"""
        visualizations = []

        # 🎯 基于计算结果生成趋势图
        if calculation_result and calculation_result.get('success'):
            calc_res = calculation_result.get('calculation_result')
            if calc_res and hasattr(calc_res, 'detailed_results'):
                detailed_results = getattr(calc_res, 'detailed_results', {})

                # 查找有趋势数据的指标
                for metric_name, trend_data in detailed_results.items():
                    if isinstance(trend_data, dict) and 'data_points' in trend_data:
                        data_points = trend_data.get('data_points', 0)
                        if data_points > 3:  # 有足够数据点
                            # 生成趋势线图（这里简化处理）
                            visualizations.append({
                                'type': 'line',
                                'title': f'{metric_name}趋势分析',
                                'data_payload': {'message': f'{metric_name}趋势分析图表'},
                                'description': f'{metric_name}的历史趋势变化'
                            })

        return visualizations

    # ================================================================
    # 🛠️ 辅助方法和工具函数
    # ================================================================

    def _safe_get_enum(self, enum_class, value: str):
        """安全地获取枚举值"""
        try:
            return enum_class(value)
        except ValueError:
            logger.warning(f"无效的枚举值 {value} for {enum_class.__name__}, 使用默认值")
            return list(enum_class)[0]

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """从Claude响应中提取JSON"""
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            try:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if brace_match:
                    return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"无法从响应中提取有效JSON: {response_text[:300]}")
        return None

    def _validate_claude_response(self, analysis: Dict[str, Any]) -> bool:
        """验证Claude响应的完整性"""
        try:
            required_top_fields = ["query_understanding", "execution_plan"]
            for field in required_top_fields:
                if field not in analysis:
                    return False

            understanding = analysis["query_understanding"]
            required_understanding_fields = ["complexity", "query_type", "confidence"]
            for field in required_understanding_fields:
                if field not in understanding:
                    return False

            execution = analysis["execution_plan"]
            required_execution_fields = ["api_calls", "needs_calculation"]
            for field in required_execution_fields:
                if field not in execution:
                    return False

            if not isinstance(execution["api_calls"], list):
                return False

            return True
        except Exception:
            return False

    async def _legacy_data_fetch(self, query_analysis: QueryAnalysisResult) -> Optional[FetcherExecutionResult]:
        """降级数据获取逻辑"""
        try:
            api_calls = query_analysis.api_calls_needed

            if not api_calls:
                api_calls = [{"method": "get_system_data", "params": {}, "reason": "默认数据获取"}]

            @dataclass
            class SimpleAcquisitionPlan:
                plan_id: str = f"plan_{int(time.time())}"
                api_call_plans: List = field(default_factory=list)
                parallel_groups: List = field(default_factory=list)
                total_estimated_time: float = 30.0
                data_requirements: List = field(default_factory=list)

            acquisition_plan = SimpleAcquisitionPlan()

            for api_call in api_calls:
                call_plan = type('CallPlan', (), {
                    'call_method': api_call.get('method', 'get_system_data'),
                    'parameters': api_call.get('params', {}),
                    'reason': api_call.get('reason', '数据获取'),
                    'priority': type('Priority', (), {'value': 'normal'})(),
                    'retry_strategy': 'default'
                })()

                acquisition_plan.api_call_plans.append(call_plan)

            execution_result = await self.data_fetcher.execute_data_acquisition_plan(acquisition_plan)
            return execution_result

        except Exception as e:
            logger.error(f"降级数据获取失败: {e}")
            return None

    def _determine_intelligent_strategy(self, query_analysis: QueryAnalysisResult) -> ProcessingStrategy:
        """确定智能处理策略"""
        if query_analysis.needs_calculation:
            return ProcessingStrategy.DATA_WITH_CALC
        elif query_analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            return ProcessingStrategy.COMPREHENSIVE
        else:
            return ProcessingStrategy.SIMPLE_DATA

    def _get_intelligent_processors_used(self, query_analysis: QueryAnalysisResult,
                                         calculation_result: Optional[Dict[str, Any]]) -> List[str]:
        """获取使用的智能处理器"""
        processors = ["IntelligentClaude", "SmartDataFetcher", "IntelligentExtractor"]

        if query_analysis.needs_calculation and calculation_result:
            processors.append("UnifiedCalculator")

        return processors

    def _calculate_intelligent_confidence(self, query_analysis: QueryAnalysisResult,
                                          extracted_data: Dict[str, Any],
                                          calculation_result: Optional[Dict[str, Any]],
                                          insights: List[BusinessInsight]) -> float:
        """计算智能置信度"""
        confidence_factors = []

        # 查询理解置信度
        confidence_factors.append(query_analysis.confidence_score)

        # 数据提取质量
        data_quality = extracted_data.get("data_quality", 0.5)
        confidence_factors.append(data_quality)

        # 计算置信度
        if calculation_result and calculation_result.get('success'):
            calc_confidence = calculation_result.get('confidence', 0.5)
            confidence_factors.append(calc_confidence)

        # 洞察质量
        if insights:
            insight_confidence = sum(insight.confidence_score for insight in insights) / len(insights)
            confidence_factors.append(insight_confidence)

        # 策略匹配度
        strategy = extracted_data.get("extraction_strategy", "")
        if strategy != "comprehensive":  # 精确策略匹配
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _calculate_data_quality_score(self, extracted_data: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        base_quality = extracted_data.get("data_quality", 0.8)

        # 根据提取策略调整质量分数
        strategy = extracted_data.get("extraction_strategy", "")
        if strategy in ["financial_overview", "expiry_analysis", "reinvestment_analysis"]:
            # 这些策略通常有更准确的数据提取
            return min(base_quality + 0.1, 1.0)

        return base_quality

    def _calculate_intelligent_completeness(self, extracted_data: Dict[str, Any],
                                            calculation_result: Optional[Dict[str, Any]],
                                            insights: List[BusinessInsight]) -> float:
        """计算智能完整性"""
        completeness = 0.0

        # 数据提取完整性
        if extracted_data.get("status") == "数据提取成功":
            completeness += 0.4

        # 计算完整性
        if calculation_result and calculation_result.get('success'):
            completeness += 0.3

        # 洞察完整性
        if insights:
            completeness += 0.3

        return min(completeness, 1.0)

    def _extract_intelligent_metrics(self, extracted_data: Dict[str, Any],
                                     calculation_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """提取智能关键指标"""
        metrics = {}

        # 🎯 从提取数据中获取核心指标
        strategy = extracted_data.get("extraction_strategy", "")

        if strategy == "financial_overview":
            financial_data = extracted_data.get("financial_overview", {})
            metrics.update({
                '总余额': financial_data.get("total_balance", 0),
                '净流入': financial_data.get("net_flow", 0),
                '活跃用户数': financial_data.get("active_users", 0)
            })

        elif strategy == "expiry_analysis":
            expiry_data = extracted_data.get("expiry_analysis", {})
            metrics.update({
                '到期金额': expiry_data.get("expiry_amount", 0),
                '到期数量': expiry_data.get("expiry_count", 0)
            })

        elif strategy == "reinvestment_analysis":
            reinvest_data = extracted_data.get("reinvestment_analysis", {})
            metrics.update({
                '复投金额': reinvest_data.get("estimated_reinvest_amount", 0),
                '提现金额': reinvest_data.get("estimated_withdrawal_amount", 0)
            })

        # 🎯 从计算结果中获取关键指标
        if calculation_result and calculation_result.get('success'):
            calc_res = calculation_result.get('calculation_result')
            if calc_res:
                if hasattr(calc_res, 'primary_result'):
                    metrics['计算主结果'] = calc_res.primary_result
                if hasattr(calc_res, 'detailed_results'):
                    detailed = calc_res.detailed_results
                    if isinstance(detailed, dict):
                        metrics.update(detailed)

        return metrics

    def _get_intelligent_ai_summary(self, timing: Dict[str, float]) -> Dict[str, Any]:
        """获取智能AI协作摘要"""
        return {
            'claude_used': self.claude_client is not None,
            'gpt_used': self.gpt_client is not None,
            'total_ai_time': timing.get('parsing', 0) + timing.get('response_generation', 0),
            'data_extraction_time': timing.get('data_extraction', 0),
            'intelligence_level': 'enhanced',
            'extraction_strategy_applied': True,
            'architecture': 'intelligent_claude_dominant'
        }

    # ================================================================
    # 🎯 保留的必要方法 (继承自原版本)
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
            recent_messages = getattr(self.conversation_manager, 'get_recent_messages', lambda *args: [])(
                conversation_id_for_db, 2)

            for msg in recent_messages:
                if isinstance(msg, dict):
                    is_user = msg.get('is_user', False)
                    content = msg.get('content', '')
                    if is_user and content.strip() == user_query.strip():
                        return False

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
            ai_message_id = self.conversation_manager.add_message(
                conversation_id_for_db, False, result.response_text)

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
            'quick_response_queries': 0,
            'complex_calculation_queries': 0,
            'avg_processing_time': 0.0,
            'avg_confidence_score': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'intelligent_extractions': 0
        }

    def _update_stats(self, result: ProcessingResult):
        """更新统计"""
        if result.success:
            self.orchestrator_stats['successful_queries'] += 1
        else:
            self.orchestrator_stats['failed_queries'] += 1

        # 特殊统计
        if result.processing_strategy == ProcessingStrategy.QUICK_RESPONSE:
            self.orchestrator_stats['quick_response_queries'] += 1
        elif result.processing_strategy == ProcessingStrategy.DATA_WITH_CALC:
            self.orchestrator_stats['complex_calculation_queries'] += 1

        if 'extraction_strategy' in result.processing_metadata:
            self.orchestrator_stats['intelligent_extractions'] += 1

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
            stats['quick_response_rate'] = stats.get('quick_response_queries', 0) / total
            stats['intelligent_extraction_rate'] = stats.get('intelligent_extractions', 0) / total
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
            'architecture': 'intelligent_enhanced',
            'version': self.config.get('version', '3.0.0-intelligent'),
            'components': {
                'claude_available': self.claude_client is not None,
                'gpt_available': self.gpt_client is not None,
                'query_parser_ready': self.query_parser is not None,
                'data_fetcher_ready': self.data_fetcher is not None,
                'calculator_ready': self.statistical_calculator is not None,
                'date_utils_ready': self.date_utils is not None,
                'formatter_ready': self.financial_formatter is not None
            },
            'intelligent_features': {
                'smart_extraction': True,
                'intelligent_calculation': True,
                'enhanced_insights': True,
                'adaptive_visualization': True
            },
            'statistics': self.get_orchestrator_stats(),
            'timestamp': datetime.now().isoformat()
        }

    async def close(self):
        """关闭编排器"""
        logger.info("关闭智能增强版编排器...")

        if self.data_fetcher and hasattr(self.data_fetcher, 'close'):
            await self.data_fetcher.close()

        if self.db_connector and hasattr(self.db_connector, 'close'):
            await self.db_connector.close()

        self.result_cache.clear()
        self.extraction_cache.clear()
        self.initialized = False

        logger.info("智能增强版编排器已关闭")


# ================================================================
# 🏭 工厂函数和全局实例管理
# ================================================================

_orchestrator_instance: Optional[IntelligentQAOrchestrator] = None


def get_orchestrator(claude_client_instance: Optional[ClaudeClient] = None,
                     gpt_client_instance: Optional[OpenAIClient] = None,
                     db_connector_instance: Optional[DatabaseConnector] = None,
                     app_config_instance: Optional[AppConfig] = None) -> IntelligentQAOrchestrator:
    """获取智能编排器实例"""
    global _orchestrator_instance

    if _orchestrator_instance is None:
        logger.info("创建新的智能增强版编排器实例")
        _orchestrator_instance = IntelligentQAOrchestrator(
            claude_client_instance, gpt_client_instance,
            db_connector_instance, app_config_instance
        )

    return _orchestrator_instance


# ================================================================
# 🎯 使用示例和测试
# ================================================================

async def test_intelligent_orchestrator():
    """测试智能编排器"""
    try:
        orchestrator = get_orchestrator()
        await orchestrator.initialize()

        test_queries = [
            "今天总资金多少",  # 快速响应
            "6月1日的有多少产品到期，总资金多少",  # 智能组合
            "6月1日至6月30日产品到期金额是多少，如果使用25%复投，7月1日剩余资金有多少",  # 复杂计算
            "假设现在没入金的情况公司还能运行多久"  # 现金跑道分析
        ]

        for query in test_queries:
            print(f"\n🧠 测试查询: {query}")
            result = await orchestrator.process_intelligent_query(query)
            print(f"✅ 成功: {result.success}")
            print(f"📊 策略: {result.processing_strategy.value}")
            print(f"🎯 置信度: {result.confidence_score:.2f}")
            print(f"💬 回答: {result.response_text[:200]}...")

        await orchestrator.close()

    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_intelligent_orchestrator())