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
from typing import Dict, Any, List, Optional, Tuple, Union
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


# ============= 简化的数据类定义 =============

@dataclass
class BusinessInsight:
    """简化的业务洞察类 - 替代已删除的复杂版本"""
    title: str
    summary: str
    confidence_score: float = 0.8
    insight_type: str = "general"
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class ProcessingStrategy(Enum):
    """简化的处理策略"""
    SIMPLE_DATA = "simple_data"  # 简单数据获取
    DATA_WITH_CALC = "data_with_calc"  # 数据获取 + 计算
    COMPREHENSIVE = "comprehensive"  # 全面分析
    ERROR_HANDLING = "error_handling"  # 错误处理


@dataclass
class ProcessingResult:
    """处理结果 - 简化版"""
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
        """🎯 大幅简化的组件初始化"""
        # 🎯 核心三剑客 - 只保留必要的
        self.query_parser: Optional[SmartQueryParser] = None
        self.data_fetcher: Optional[SmartDataFetcher] = None
        self.statistical_calculator: Optional[UnifiedCalculator] = None

        # 工具组件
        self.date_utils: Optional[DateUtils] = None
        self.financial_formatter: Optional[FinancialFormatter] = None
        self.chart_generator: Optional[ChartGenerator] = None
        self.report_generator: Optional[ReportGenerator] = None
        self.conversation_manager: Optional[ConversationManager] = None

        # ❌ 删除的冗余组件 - 已不存在的文件
        # self.insight_generator = None
        # self.financial_data_analyzer = None
        # self.data_requirements_analyzer = None
        # self.current_data_processor = None
        # self.historical_analysis_processor = None
        # self.prediction_processor = None

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

        logger.info(f"🎯 QueryID: {query_id} - 开始处理: '{user_query[:50]}...'")
        self.orchestrator_stats['total_queries'] += 1

        # 处理对话ID
        conversation_id_for_db = self._parse_conversation_id(conversation_id)

        # 保存用户输入
        user_message_saved = await self._save_user_message_if_needed(
            conversation_id_for_db, user_query, query_id)

        try:
            # 🚀 新增：快速查询检测和处理
            quick_response = await self._try_quick_response(user_query, query_id)
            if quick_response:
                logger.info(f"⚡ QueryID: {query_id} - 使用快速响应路径")

                # 构建快速结果
                total_processing_time = time.time() - start_time
                result = ProcessingResult(
                    session_id=session_id,
                    query_id=query_id,
                    success=True,
                    response_text=quick_response['response'],
                    key_metrics=quick_response.get('metrics', {}),
                    processing_strategy=ProcessingStrategy.SIMPLE_DATA,
                    processors_used=["QuickResponse", "APIConnector"],
                    confidence_score=0.95,  # 简单查询高置信度
                    data_quality_score=quick_response.get('data_quality', 0.9),
                    response_completeness=1.0,  # 直接数据查询完整度高
                    total_processing_time=total_processing_time,
                    ai_processing_time=0.0,  # 快速路径无AI处理
                    data_fetching_time=quick_response.get('fetch_time', 0.0),
                    processing_metadata={
                        'query_type': 'quick_data_query',
                        'api_used': quick_response.get('api_method', ''),
                        'quick_response': True
                    },
                    conversation_id=conversation_id
                )

                # 更新统计
                self._update_stats(result)

                # 保存AI响应
                if conversation_id_for_db and user_message_saved:
                    await self._save_ai_response_if_needed(conversation_id_for_db, result, query_id)

                logger.info(f"⚡ QueryID: {query_id} - 快速处理完成，耗时: {total_processing_time:.2f}s")
                return result

            # 🎯 原有的完整处理流程
            timing = {}

            # 1️⃣ Claude 理解查询
            start_t = time.time()
            query_analysis = await self._claude_understand_query(user_query, conversation_id_for_db)
            timing['parsing'] = time.time() - start_t

            # 2️⃣ 获取数据
            start_t = time.time()
            data_result = await self._execute_simplified_data_fetch(query_analysis)
            timing['data_fetching'] = time.time() - start_t

            # 3️⃣ 计算处理 (如果需要)
            start_t = time.time()
            calculation_result = None
            if query_analysis.needs_calculation:
                calculation_result = await self._execute_unified_calculation(query_analysis, data_result)
            timing['calculation'] = time.time() - start_t

            # 4️⃣ Claude 生成最终回答
            start_t = time.time()
            response_text = await self._claude_generate_final_response(
                user_query, query_analysis, data_result, calculation_result)
            timing['response_generation'] = time.time() - start_t

            # 5️⃣ 简化的洞察生成
            start_t = time.time()
            insights = await self._generate_simple_insights(data_result, calculation_result, query_analysis)
            timing['insights'] = time.time() - start_t

            # 6️⃣ 简化的可视化生成
            start_t = time.time()
            visualizations = await self._generate_simple_visualizations(data_result, calculation_result)
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
                data_quality_score=getattr(data_result, 'confidence_level', 0.8) if data_result else 0.5,
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

    async def _try_quick_response(self, user_query: str, query_id: str) -> Optional[Dict[str, Any]]:
        """🚀 尝试快速响应 - 检测简单数据查询"""
        try:
            query_lower = user_query.lower()

            # 🔍 快速查询模式检测
            quick_patterns = self._detect_quick_query_patterns(query_lower)

            if not quick_patterns:
                return None

            logger.info(f"⚡ 检测到快速查询模式: {quick_patterns}")

            # 🚀 执行快速数据获取
            start_fetch_time = time.time()
            api_result = await self._execute_quick_api_call(quick_patterns)
            fetch_time = time.time() - start_fetch_time

            if not api_result or not api_result.get('success'):
                logger.warning(f"快速API调用失败，回退到完整流程")
                return None

            # 🎯 格式化快速响应
            formatted_response = self._format_quick_response(api_result, quick_patterns, user_query)

            return {
                'response': formatted_response['text'],
                'metrics': formatted_response.get('metrics', {}),
                'data_quality': api_result.get('validation', {}).get('confidence', 0.9),
                'fetch_time': fetch_time,
                'api_method': quick_patterns['api_method']
            }

        except Exception as e:
            logger.error(f"快速响应处理失败: {e}")
            return None

    def _detect_quick_query_patterns(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """🔍 检测快速查询模式 - 增强版"""

        # 🎯 更精确的模式匹配
        patterns = [
            # 特定日期的每日数据查询
            {
                'name': 'specific_date_daily',
                'regex': r'(\d{1,2})月(\d{1,2})[日号]',
                'keywords': ['入金', '出金', '注册', '数据', '购买'],
                'api_method': 'get_daily_data',
                'description': '特定日期每日数据'
            },

            # YYYYMMDD格式日期查询
            {
                'name': 'date_format_daily',
                'regex': r'(\d{4})(\d{2})(\d{2})',
                'keywords': ['数据', '入金', '出金'],
                'api_method': 'get_daily_data',
                'description': 'YYYYMMDD格式日期查询'
            },

            # 特定日期到期查询
            {
                'name': 'specific_date_expiry',
                'regex': r'(\d{1,2})月(\d{1,2})[日号]',
                'keywords': ['到期', '过期', '产品到期'],
                'api_method': 'get_product_end_data',
                'description': '特定日期到期查询'
            },

            # 今日查询
            {
                'name': 'today_query',
                'keywords': ['今天', '今日', '当天'],
                'data_types': ['入金', '出金', '注册', '数据', '到期'],
                'api_method': 'auto',  # 根据数据类型自动选择
                'description': '今日数据查询'
            },

            # 系统概览查询
            {
                'name': 'system_simple',
                'keywords': ['活跃会员', '总用户', '用户数', '总余额', '当前资金'],
                'data_types': [],
                'api_method': 'get_system_data',
                'description': '系统概览查询'
            },

            # 🆕 区间到期查询（简单版）
            {
                'name': 'simple_interval_expiry',
                'regex': r'(\d{1,2})月(\d{1,2})[日号].*?(\d{1,2})月(\d{1,2})[日号]',
                'keywords': ['到期', '产品到期'],
                'exclude_keywords': ['复投', '计算', '预计', '分析'],  # 排除复杂计算
                'api_method': 'get_product_end_interval',
                'description': '简单区间到期查询'
            }
        ]

        # 🔍 逐个匹配模式
        for pattern in patterns:
            if self._match_pattern(query_lower, pattern):
                return self._extract_pattern_info(query_lower, pattern)

        return None

    def _match_pattern(self, query_lower: str, pattern: Dict[str, Any]) -> bool:
        """匹配单个模式"""
        import re

        # 检查排除关键词
        if 'exclude_keywords' in pattern:
            if any(exc in query_lower for exc in pattern['exclude_keywords']):
                return False

        # 正则匹配
        if 'regex' in pattern:
            if not re.search(pattern['regex'], query_lower):
                return False

        # 关键词匹配
        if 'keywords' in pattern:
            if not any(kw in query_lower for kw in pattern['keywords']):
                return False

        # 数据类型匹配
        if 'data_types' in pattern and pattern['data_types']:
            if not any(dt in query_lower for dt in pattern['data_types']):
                return False

        return True

    def _extract_pattern_info(self, query_lower: str, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """提取模式信息和参数"""
        import re

        result = {
            'pattern': pattern['name'],
            'description': pattern['description'],
            'params': {}
        }

        # 🎯 根据模式类型确定API方法和参数
        if pattern['name'] == 'specific_date_daily':
            date_match = re.search(r'(\d{1,2})月(\d{1,2})[日号]', query_lower)
            if date_match:
                month, day = int(date_match.group(1)), int(date_match.group(2))
                date_str = f"2024{month:02d}{day:02d}"  # 假设2024年
                result['api_method'] = 'get_daily_data'
                result['params'] = {'date': date_str}

        elif pattern['name'] == 'date_format_daily':
            date_match = re.search(r'(\d{8})', query_lower)
            if date_match:
                result['api_method'] = 'get_daily_data'
                result['params'] = {'date': date_match.group(1)}

        elif pattern['name'] == 'specific_date_expiry':
            date_match = re.search(r'(\d{1,2})月(\d{1,2})[日号]', query_lower)
            if date_match:
                month, day = int(date_match.group(1)), int(date_match.group(2))
                date_str = f"2024{month:02d}{day:02d}"
                result['api_method'] = 'get_product_end_data'
                result['params'] = {'date': date_str}


        elif pattern['name'] == 'today_query':

            # 根据数据类型选择API

            if any(kw in query_lower for kw in ['到期', '过期']):

                result['api_method'] = 'get_expiring_products_today'

                result['params'] = {}

            else:

                result['api_method'] = 'get_daily_data'

                result['params'] = {'date': datetime.now().strftime('%Y%m%d')}

        elif pattern['name'] == 'system_simple':
            result['api_method'] = 'get_system_data'

        elif pattern['name'] == 'simple_interval_expiry':
            dates = re.findall(r'(\d{1,2})月(\d{1,2})[日号]', query_lower)
            if len(dates) >= 2:
                start_month, start_day = int(dates[0][0]), int(dates[0][1])
                end_month, end_day = int(dates[1][0]), int(dates[1][1])
                start_date = f"2024{start_month:02d}{start_day:02d}"
                end_date = f"2024{end_month:02d}{end_day:02d}"
                result['api_method'] = 'get_product_end_interval'
                result['params'] = {'start_date': start_date, 'end_date': end_date}

        return result

    def _parse_date_from_query(self, query_lower: str) -> Optional[str]:
        """🔍 从查询中解析日期"""
        import re

        # 匹配 "X月X日" 或 "X月X号" 格式
        date_pattern = r'(\d{1,2})月(\d{1,2})[日号]'
        match = re.search(date_pattern, query_lower)

        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            current_year = datetime.now().year

            try:
                # 构造日期
                target_date = datetime(current_year, month, day)
                return target_date.strftime('%Y%m%d')
            except ValueError:
                logger.warning(f"无效日期: {month}月{day}日")
                return None

        # 匹配 "YYYYMMDD" 格式
        date_pattern2 = r'(\d{8})'
        match2 = re.search(date_pattern2, query_lower)
        if match2:
            date_str = match2.group(1)
            try:
                # 验证日期格式
                datetime.strptime(date_str, '%Y%m%d')
                return date_str
            except ValueError:
                return None

        return None

    async def _execute_quick_api_call(self, pattern_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """🚀 执行快速API调用 - 修复版"""
        try:
            api_method = pattern_info['api_method']
            params = pattern_info['params']

            if not self.data_fetcher or not self.data_fetcher.api_connector:
                logger.error("API连接器不可用")
                return None

            # 🎯 根据方法名调用对应的API
            api_connector = self.data_fetcher.api_connector

            logger.info(f"🚀 执行快速API调用: {api_method}, 参数: {params}")

            if api_method == 'get_daily_data':
                if 'date' in params:
                    result = await api_connector.get_daily_data(params['date'])
                else:
                    result = await api_connector.get_daily_data()

            elif api_method == 'get_system_data':
                result = await api_connector.get_system_data()

            elif api_method == 'get_expiring_products_today':
                result = await api_connector.get_expiring_products_today()

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

            # 🔍 调试：打印API返回结果
            logger.info(
                f"🔍 API调用结果: success={result.get('success')}, data_keys={list(result.get('data', {}).keys()) if result.get('data') else 'None'}")

            # 🎯 即使验证失败，只要API调用成功就继续处理
            if result.get('success'):
                return result
            else:
                logger.warning(f"API调用失败: {result.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"快速API调用失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _format_quick_response(self, api_result: Dict[str, Any],
                               pattern_info: Dict[str, Any],
                               original_query: str) -> Dict[str, Any]:
        """🎯 格式化快速响应 - 修复版"""
        try:
            # 🔍 调试：打印输入数据
            logger.info(f"🎯 格式化快速响应: pattern={pattern_info['pattern']}")
            logger.info(f"🔍 API结果keys: {list(api_result.keys())}")

            data = api_result.get('data', {})

            if not data:
                logger.warning("API返回的data为空")
                return {
                    'text': f"获取到{pattern_info['description']}，但数据为空。",
                    'metrics': {}
                }

            # 🔍 调试：打印data内容
            logger.info(f"🔍 数据内容: {data}")

            pattern = pattern_info['pattern']

            if pattern == 'specific_date_daily' or pattern == 'date_format_daily':
                return self._format_daily_data_response(data, original_query)

            elif pattern == 'system_simple':
                return self._format_system_overview_response(data, original_query)

            elif pattern == 'specific_date_expiry':
                return self._format_expiry_response(data, original_query)

            elif pattern == 'simple_interval_expiry':
                return self._format_interval_expiry_response(data, original_query)

            else:
                return {
                    'text': f"已获取{pattern_info['description']}数据，但格式化器未实现。数据：{str(data)[:200]}",
                    'metrics': {}
                }

        except Exception as e:
            logger.error(f"响应格式化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'text': f"数据获取成功，但格式化时遇到问题：{str(e)}",
                'metrics': {}
            }

    def _format_daily_data_response(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """📅 格式化每日数据响应 - 修复版"""
        try:
            logger.info(f"📅 格式化每日数据: {data}")

            if not data:
                return {'text': "未找到当日数据。", 'metrics': {}}

            # 🔍 从API返回的数据中提取字段
            date = data.get('日期', data.get('date', '未知日期'))
            inflow = float(data.get('入金', data.get('inflow', 0)))
            outflow = float(data.get('出金', data.get('outflow', 0)))
            registrations = int(data.get('注册人数', data.get('registrations', 0)))
            purchases = int(data.get('购买产品数量', data.get('purchases', 0)))
            holdings = int(data.get('持仓人数', data.get('holdings', 0)))
            expired_products = int(data.get('到期产品数量', data.get('expired_products', 0)))

            # 🎯 根据查询内容重点突出相关数据
            query_lower = query.lower()

            response_parts = [f"📅 {date} 数据概览："]

            # 🎯 智能响应：根据查询关键词决定显示内容
            if '入金' in query_lower:
                response_parts.append(f"💰 入金：¥{inflow:,.2f}")
                if inflow > 0:
                    response_parts.append("📈 入金情况良好")

            elif '出金' in query_lower:
                response_parts.append(f"💸 出金：¥{outflow:,.2f}")

            elif '注册' in query_lower:
                response_parts.append(f"👥 新增注册：{registrations}人")

            elif '购买' in query_lower or '产品' in query_lower:
                response_parts.append(f"🛍️ 产品购买：{purchases}笔")

            else:
                # 完整数据展示
                response_parts.extend([
                    f"💰 入金：¥{inflow:,.2f}",
                    f"💸 出金：¥{outflow:,.2f}",
                    f"👥 新增注册：{registrations}人",
                    f"🛍️ 产品购买：{purchases}笔"
                ])

                if holdings > 0:
                    response_parts.append(f"📊 持仓人数：{holdings}人")
                if expired_products > 0:
                    response_parts.append(f"⏰ 到期产品：{expired_products}笔")

            # 添加简单分析
            net_flow = inflow - outflow
            if net_flow > 0:
                response_parts.append(f"📈 净流入：¥{net_flow:,.2f}")
            elif net_flow < 0:
                response_parts.append(f"📉 净流出：¥{abs(net_flow):,.2f}")
            else:
                response_parts.append("⚖️ 资金流平衡")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '入金': inflow,
                    '出金': outflow,
                    '净流入': net_flow,
                    '注册人数': registrations,
                    '购买产品数量': purchases,
                    '持仓人数': holdings
                }
            }

        except Exception as e:
            logger.error(f"每日数据格式化失败: {e}")
            return {
                'text': f"数据获取成功，原始数据：{str(data)}",
                'metrics': {}
            }

    def _format_interval_expiry_response(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """📊 格式化区间到期数据响应"""
        try:
            if not data:
                return {'text': "未找到区间到期数据。", 'metrics': {}}

            date_range = data.get('日期', '未知时间范围')
            total_count = int(data.get('到期数量', 0))
            total_amount = float(data.get('到期金额', 0))

            response_parts = [f"📊 {date_range} 产品到期汇总："]
            response_parts.append(f"📦 总到期数量：{total_count}笔")
            response_parts.append(f"💰 总到期金额：¥{total_amount:,.2f}")

            if total_count > 0:
                avg_amount = total_amount / total_count
                response_parts.append(f"📊 平均金额：¥{avg_amount:,.2f}")

            # 如果有产品列表，显示前几个
            product_list = data.get('产品列表', [])
            if product_list:
                response_parts.append(f"\n🔍 涉及产品：")
                for i, product in enumerate(product_list[:3]):  # 只显示前3个
                    name = product.get('产品名称', f'产品{i + 1}')
                    amount = float(product.get('到期金额', 0))
                    if amount > 0:
                        response_parts.append(f"  • {name}：¥{amount:,.2f}")

            return {
                'text': '\n'.join(response_parts),
                'metrics': {
                    '到期数量': total_count,
                    '到期金额': total_amount
                }
            }

        except Exception as e:
            logger.error(f"区间到期数据格式化失败: {e}")
            return {
                'text': f"数据获取成功，原始数据：{str(data)}",
                'metrics': {}
            }

    def _format_system_overview_response(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """🏦 格式化系统概览响应"""
        if not data:
            return {'text': "未找到系统数据。", 'metrics': {}}

        total_balance = float(data.get('总余额', 0))
        total_inflow = float(data.get('总入金', 0))
        total_outflow = float(data.get('总出金', 0))

        user_stats = data.get('用户统计', {})
        total_users = int(user_stats.get('总用户数', 0))
        active_users = int(user_stats.get('活跃用户数', 0))

        response_parts = ["🏦 系统概览："]

        query_lower = query.lower()

        if '余额' in query_lower or '资金' in query_lower:
            response_parts.append(
                f"💰 总余额：{self.financial_formatter.format_currency(total_balance) if self.financial_formatter else f'{total_balance:.2f}'}")

        if '入金' in query_lower:
            response_parts.append(
                f"📈 总入金：{self.financial_formatter.format_currency(total_inflow) if self.financial_formatter else f'{total_inflow:.2f}'}")

        if '出金' in query_lower:
            response_parts.append(
                f"📉 总出金：{self.financial_formatter.format_currency(total_outflow) if self.financial_formatter else f'{total_outflow:.2f}'}")

        if '用户' in query_lower or '活跃' in query_lower:
            response_parts.append(f"👥 总用户：{total_users}人")
            response_parts.append(f"🔥 活跃用户：{active_users}人")
            if total_users > 0:
                activity_rate = (active_users / total_users) * 100
                response_parts.append(f"📊 活跃率：{activity_rate:.1f}%")

        # 如果没有特定关键词，显示核心数据
        if len(response_parts) == 1:
            response_parts.extend([
                f"💰 总余额：{self.financial_formatter.format_currency(total_balance) if self.financial_formatter else f'{total_balance:.2f}'}",
                f"👥 总用户：{total_users}人",
                f"🔥 活跃用户：{active_users}人"
            ])

        return {
            'text': '\n'.join(response_parts),
            'metrics': {
                '总余额': total_balance,
                '总入金': total_inflow,
                '总出金': total_outflow,
                '总用户数': total_users,
                '活跃用户数': active_users
            }
        }

    def _format_expiry_response(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """⏰ 格式化到期数据响应"""
        if not data:
            return {'text': "未找到到期数据。", 'metrics': {}}

        date = data.get('日期', '今日')
        expiry_count = int(data.get('到期数量', 0))
        expiry_amount = float(data.get('到期金额', 0))

        response_parts = [f"⏰ {date} 产品到期情况："]
        response_parts.append(f"📦 到期数量：{expiry_count}笔")
        response_parts.append(
            f"💰 到期金额：{self.financial_formatter.format_currency(expiry_amount) if self.financial_formatter else f'{expiry_amount:.2f}'}")

        if expiry_count > 0:
            avg_amount = expiry_amount / expiry_count
            response_parts.append(
                f"📊 平均金额：{self.financial_formatter.format_currency(avg_amount) if self.financial_formatter else f'{avg_amount:.2f}'}")

        return {
            'text': '\n'.join(response_parts),
            'metrics': {
                '到期数量': expiry_count,
                '到期金额': expiry_amount
            }
        }
    async def _claude_generate_final_response(self, user_query: str,
                                              query_analysis: QueryAnalysisResult,
                                              data_result: Optional[FetcherExecutionResult],
                                              calculation_result: Optional[Dict[str, Any]]) -> str:
        """🎯 Claude生成最终回答"""
        try:
            if not self.claude_client:
                return self._generate_fallback_response(user_query, data_result, calculation_result, [])

            # 准备数据摘要
            data_summary = self._summarize_data_for_claude(data_result, user_query, query_analysis)
            calc_summary = self._summarize_calculation_for_claude(calculation_result)

            # Claude生成回答的提示
            response_prompt = f"""
            基于以下数据分析，请为用户查询生成专业、准确的回答：

            用户查询："{user_query}"

            数据摘要：{json.dumps(data_summary, ensure_ascii=False, indent=2)}

            计算结果：{json.dumps(calc_summary, ensure_ascii=False, indent=2)}

            请生成：
            1. 直接回答用户问题
            2. 提供具体数据支持
            3. 给出简明的分析结论
            4. 如有必要，提供建议

            回答要专业、准确、易懂。
            """

            # 调用Claude
            result = await self.claude_client.analyze_complex_query(response_prompt, {
                "user_query": user_query,
                "data_summary": data_summary,
                "calculation_summary": calc_summary
            })

            if result.get("success"):
                return result.get("analysis", "抱歉，无法生成详细回答")
            else:
                return self._generate_fallback_response(user_query, data_result, calculation_result, [])

        except Exception as e:
            logger.error(f"Claude生成回答失败: {e}")
            return self._generate_fallback_response(user_query, data_result, calculation_result, [])

    async def _legacy_data_fetch(self, query_analysis: QueryAnalysisResult) -> Optional[FetcherExecutionResult]:
        """降级数据获取逻辑"""
        try:
            # 🎯 构建简化的数据获取计划
            api_calls = query_analysis.api_calls_needed

            if not api_calls:
                # 默认获取系统数据
                api_calls = [{"method": "get_system_data", "params": {}, "reason": "默认数据获取"}]

            # 🎯 创建模拟的acquisition_plan对象
            @dataclass
            class SimpleAcquisitionPlan:
                plan_id: str = f"plan_{int(time.time())}"
                api_call_plans: List = field(default_factory=list)
                parallel_groups: List = field(default_factory=list)
                total_estimated_time: float = 30.0
                data_requirements: List = field(default_factory=list)

            # 转换api_calls格式以匹配SmartDataFetcher期望的格式
            acquisition_plan = SimpleAcquisitionPlan()

            for api_call in api_calls:
                # 创建模拟的call_plan对象
                call_plan = type('CallPlan', (), {
                    'call_method': api_call.get('method', 'get_system_data'),
                    'parameters': api_call.get('params', {}),
                    'reason': api_call.get('reason', '数据获取'),
                    'priority': type('Priority', (), {'value': 'normal'})(),
                    'retry_strategy': 'default'
                })()

                acquisition_plan.api_call_plans.append(call_plan)

            # 调用SmartDataFetcher的主要方法
            execution_result = await self.data_fetcher.execute_data_acquisition_plan(acquisition_plan)
            return execution_result

        except Exception as e:
            logger.error(f"降级数据获取失败: {e}")
            return None

    def _extract_user_insights(self, user_data: Dict[str, Any], time_scope: str) -> Dict[str, Any]:
        """提取用户洞察"""
        return {"用户数据": user_data.get("summary", "用户数据处理中")}

    def _extract_product_insights(self, product_data: Dict[str, Any], time_scope: str) -> Dict[str, Any]:
        """提取产品洞察"""
        return {"产品数据": product_data.get("summary", "产品数据处理中")}

    def _extract_expiry_insights(self, expiry_data: Dict[str, Any], time_scope: str) -> Dict[str, Any]:
        """提取到期洞察"""
        return {"到期数据": expiry_data.get("summary", "到期数据处理中")}

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

    async def _execute_simplified_data_fetch(self, query_analysis: QueryAnalysisResult) -> Optional[
        FetcherExecutionResult]:
        """2️⃣ 增强版数据获取"""
        logger.debug(f"📊 执行增强版数据获取")

        if not self.data_fetcher:
            raise RuntimeError("SmartDataFetcher 未初始化")

        try:
            # 🎯 使用增强版智能数据获取
            if hasattr(self.data_fetcher.api_connector, 'intelligent_data_fetch_enhanced'):
                execution_result = await self.data_fetcher.api_connector.intelligent_data_fetch_enhanced(
                    query_analysis.to_dict()
                )
            else:
                # 降级到原有逻辑
                execution_result = await self._legacy_data_fetch(query_analysis)

            return execution_result

        except Exception as e:
            logger.error(f"增强版数据获取失败: {e}")
            return None

    async def _execute_unified_calculation(self, query_analysis: QueryAnalysisResult,
                                           data_result: Optional[FetcherExecutionResult]) -> Optional[Dict[str, Any]]:
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

    async def _generate_simple_insights(self, data_result: Optional[FetcherExecutionResult],
                                        calculation_result: Optional[Dict[str, Any]],
                                        query_analysis: QueryAnalysisResult) -> List[BusinessInsight]:
        """4️⃣ 生成简化的业务洞察"""
        logger.debug("💡 生成简化业务洞察")

        insights = []

        try:
            # 基于数据结果生成基础洞察
            if data_result and hasattr(data_result, 'processed_data'):
                system_data = data_result.processed_data.get('system_data', {})
                if system_data:
                    total_balance = system_data.get('total_balance', 0)
                    total_inflow = system_data.get('total_inflow', 0)
                    total_outflow = system_data.get('total_outflow', 0)

                    if total_balance > 0:
                        net_flow = total_inflow - total_outflow
                        outflow_ratio = total_outflow / total_inflow if total_inflow > 0 else 0

                        # 现金流洞察
                        if net_flow > 0:
                            insights.append(BusinessInsight(
                                title="正向现金流",
                                summary=f"当前净流入为 {self.financial_formatter.format_currency(net_flow) if self.financial_formatter else f'{net_flow:.2f}'}，资金状况良好",
                                confidence_score=0.9,
                                insight_type="cash_flow",
                                recommendations=["继续保持良好的资金管理"]
                            ))
                        elif net_flow < 0:
                            insights.append(BusinessInsight(
                                title="负向现金流",
                                summary=f"当前净流出为 {self.financial_formatter.format_currency(-net_flow) if self.financial_formatter else f'{-net_flow:.2f}'}，需要关注资金状况",
                                confidence_score=0.9,
                                insight_type="cash_flow",
                                recommendations=["考虑优化支出结构", "增加资金来源"]
                            ))

                        # 支出比例洞察
                        if outflow_ratio > 0.8:
                            insights.append(BusinessInsight(
                                title="高支出比例警告",
                                summary=f"支出占入金比例为 {outflow_ratio:.1%}，建议控制支出",
                                confidence_score=0.85,
                                insight_type="risk_warning",
                                recommendations=["审核支出项目", "制定支出控制策略"]
                            ))

            # 基于计算结果生成洞察
            if calculation_result and calculation_result.get('success'):
                calc_res = calculation_result.get('calculation_result')
                if calc_res and hasattr(calc_res, 'detailed_results'):
                    calc_type = calculation_result.get('calculation_type', '')

                    if 'trend' in calc_type.lower():
                        insights.append(BusinessInsight(
                            title="趋势分析结果",
                            summary="基于历史数据的趋势分析已完成",
                            confidence_score=calc_res.confidence,
                            insight_type="trend_analysis",
                            recommendations=["根据趋势调整业务策略"]
                        ))
                    elif 'roi' in calc_type.lower() or 'financial' in calc_type.lower():
                        insights.append(BusinessInsight(
                            title="财务指标分析",
                            summary="关键财务比率已计算完成",
                            confidence_score=calc_res.confidence,
                            insight_type="financial_analysis",
                            recommendations=["关注ROI指标变化", "优化投资结构"]
                        ))

        except Exception as e:
            logger.error(f"洞察生成失败: {e}")
            insights.append(BusinessInsight(
                title="系统提示",
                summary="数据分析过程中遇到问题，请稍后重试",
                confidence_score=0.3,
                insight_type="system_message"
            ))

        return insights

    def _summarize_data_for_claude(self, data_result: Optional[FetcherExecutionResult],
                                   user_query: str = "", query_analysis: Optional[QueryAnalysisResult] = None) -> Dict[
        str, Any]:
        """🎯 简化版数据总结 - 专注于核心数据"""
        if not data_result:
            return {"status": "无数据"}

        summary = {
            "status": "数据获取成功",
            "data_quality": getattr(data_result, 'confidence_level', 0.8),
            "core_data": {}
        }

        # 简化数据提取 - 只关注核心数据
        if hasattr(data_result, 'processed_data') and data_result.processed_data:
            processed = data_result.processed_data

            # 提取系统数据
            if 'system_data' in processed or any('system' in str(k).lower() for k in processed.keys()):
                summary["core_data"]["系统概览"] = "已获取系统概览数据"

            # 提取日常数据
            if 'daily_data' in processed or any('daily' in str(k).lower() for k in processed.keys()):
                summary["core_data"]["每日数据"] = "已获取每日数据"

            # 提取产品数据
            if 'product_data' in processed or any('product' in str(k).lower() for k in processed.keys()):
                summary["core_data"]["产品数据"] = "已获取产品数据"

            # 提取到期数据
            if 'expiry_data' in processed or any(
                    'expiry' in str(k).lower() or 'end' in str(k).lower() for k in processed.keys()):
                summary["core_data"]["到期数据"] = "已获取到期数据"

        return summary



    async def _generate_simple_visualizations(self, data_result: Optional[FetcherExecutionResult],
                                              calculation_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """6️⃣ 生成简化的可视化"""
        visualizations = []

        if not self.chart_generator:
            return visualizations

        try:
            # 基于数据结果生成简单图表
            if data_result and hasattr(data_result, 'processed_data'):
                processed_data = data_result.processed_data

                # 资金流向图表
                if 'system_data' in processed_data:
                    system_data = processed_data['system_data']
                    inflow = system_data.get('total_inflow', 0)
                    outflow = system_data.get('total_outflow', 0)
                    balance = system_data.get('total_balance', 0)

                    if inflow > 0 or outflow > 0:
                        chart_data = {
                            'labels': ['入金', '出金', '余额'],
                            'values': [inflow, outflow, balance]
                        }

                        visualizations.append({
                            'type': 'bar',
                            'title': '资金概览',
                            'data_payload': chart_data,
                            'description': '当前资金流向和余额状况'
                        })

                # 日度趋势图表
                if 'daily_data' in processed_data:
                    daily_data = processed_data['daily_data']
                    if isinstance(daily_data, list) and len(daily_data) > 1:
                        dates = [d.get('date', '') for d in daily_data]
                        inflows = [d.get('daily_inflow', 0) for d in daily_data]

                        chart_data = {
                            'labels': dates[-7:],  # 最近7天
                            'values': inflows[-7:]
                        }

                        visualizations.append({
                            'type': 'line',
                            'title': '入金趋势',
                            'data_payload': chart_data,
                            'description': '最近7天的入金变化趋势'
                        })

            # 基于计算结果生成图表
            if calculation_result and calculation_result.get('success'):
                calc_res = calculation_result.get('calculation_result')
                if calc_res and hasattr(calc_res, 'detailed_results'):
                    # 这里可以根据不同的计算类型生成对应的图表
                    visualizations.append({
                        'type': 'info',
                        'title': '计算结果图表',
                        'data_payload': {'message': '计算结果可视化功能开发中'},
                        'description': '计算结果的图形化展示'
                    })

        except Exception as e:
            logger.error(f"可视化生成失败: {e}")

        return visualizations

    # ================================================================
    # 🛠️ 辅助方法
    # ================================================================

    def _prepare_calculation_data(self, data_result: Optional[FetcherExecutionResult]) -> Dict[str, Any]:
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

    def _generate_fallback_response(self, user_query: str, data_result: Optional[FetcherExecutionResult],
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
                response_parts.append(f"{i}. {insight.title}: {insight.summary}")

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
                              data_result: Optional[FetcherExecutionResult],
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
            insight_confidence = sum(insight.confidence_score for insight in insights) / len(insights)
            confidence_factors.append(insight_confidence)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _calculate_completeness(self, data_result: Optional[FetcherExecutionResult],
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

    def _extract_key_metrics(self, data_result: Optional[FetcherExecutionResult],
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

    def _get_ai_summary(self, timing: Dict[str, float]) -> Dict[str, Any]:
        """获取AI协作摘要"""
        return {
            'claude_used': self.claude_client is not None,
            'gpt_used': self.gpt_client is not None,
            'total_ai_time': timing.get('parsing', 0) + timing.get('response_generation', 0),
            'architecture': 'simplified_claude_dominant'
        }

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
                'calculator_ready': self.statistical_calculator is not None
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