# core/orchestrator/intelligent_qa_orchestrator.py
"""
🚀 AI驱动的智能问答编排器
整个金融AI分析系统的核心大脑，负责协调所有组件协同工作
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

# 核心组件导入 (与您之前提供的一致)
from core.analyzers.query_parser import (
    SmartQueryParser, create_smart_query_parser, QueryAnalysisResult,
    QueryComplexity as QueryParserComplexity,
    QueryType as QueryParserQueryType,
    BusinessScenario as QueryParserBusinessScenario,
    ExecutionStep as QueryParserExecutionStep
)
from core.analyzers.data_requirements_analyzer import (
    DataRequirementsAnalyzer, create_data_requirements_analyzer, DataAcquisitionPlan
)
from core.analyzers.financial_data_analyzer import (
    FinancialDataAnalyzer, create_financial_data_analyzer, AnalysisResult as FinancialAnalysisResultType,
    TrendAnalysis as FinancialTrendAnalysis, AnomalyDetection as FinancialAnomalyDetection
)
from core.analyzers.insight_generator import (
    InsightGenerator, create_insight_generator, BusinessInsight
)
from core.data_orchestration.smart_data_fetcher import (
    SmartDataFetcher, create_smart_data_fetcher, ExecutionResult as FetcherExecutionResult,
    DataQualityLevel as FetcherDataQualityLevel
)
from data.processors.current_data_processor import CurrentDataProcessor, CurrentDataResponse
from data.processors.historical_analysis_processor import HistoricalAnalysisProcessor, HistoricalAnalysisResponse
from data.processors.prediction_processor import PredictionProcessor, PredictionResponse

# 工具类导入 (与您之前提供的一致)
from utils.helpers.date_utils import DateUtils, create_date_utils
from utils.formatters.financial_formatter import FinancialFormatter, create_financial_formatter
from utils.formatters.chart_generator import ChartGenerator as UtilsChartGenerator, \
    create_chart_generator as create_utils_chart_generator, ChartType as VisChartType
from utils.formatters.report_generator import ReportGenerator, create_report_generator, Report as ReportObject, \
    ReportSection
from data.models.conversation import ConversationManager, create_conversation_manager, Message as ConversationMessage, \
    Visual as ConversationVisual
from data.connectors.database_connector import DatabaseConnector, create_database_connector
from data.connectors.api_connector import APIConnector

# AI 客户端导入 (与您之前提供的一致)
from core.models.claude_client import ClaudeClient, CustomJSONEncoder
from core.models.openai_client import OpenAIClient

# 应用配置导入 (与您之前提供的一致)
from config import Config as AppConfig

logger = logging.getLogger(__name__)


class OrchestratorProcessingStrategy(Enum):
    DIRECT_RESPONSE = "direct_response"
    SINGLE_PROCESSOR = "single_processor"
    MULTI_PROCESSOR = "multi_processor"
    FULL_PIPELINE = "full_pipeline"
    ERROR_HANDLING = "error_handling"


@dataclass
class ProcessingResult:
    # (与您之前提供的定义一致)
    session_id: str
    query_id: str
    success: bool
    response_text: str
    insights: List[BusinessInsight] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    processing_strategy: OrchestratorProcessingStrategy = OrchestratorProcessingStrategy.SINGLE_PROCESSOR
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
            # (更新依赖并标记重初始化的逻辑)
            needs_reinit = False
            if claude_client_instance and self.claude_client != claude_client_instance: self.claude_client = claude_client_instance; needs_reinit = True
            if gpt_client_instance and self.gpt_client != gpt_client_instance: self.gpt_client = gpt_client_instance; needs_reinit = True
            if db_connector_instance and self.db_connector != db_connector_instance: self.db_connector = db_connector_instance; needs_reinit = True
            if app_config_instance and self.app_config != app_config_instance:
                self.app_config = app_config_instance
                self.config = self._load_orchestrator_config()
                needs_reinit = True
            if needs_reinit:
                self.initialized = False
                logger.info(
                    "Orchestrator dependencies updated, will re-initialize components on next use or explicit call to initialize().")
            return

        self.claude_client = claude_client_instance
        self.gpt_client = gpt_client_instance
        self.db_connector = db_connector_instance
        self.app_config = app_config_instance if app_config_instance is not None else AppConfig()
        self.config = self._load_orchestrator_config()

        self.initialized = False
        self._initialize_component_placeholders()
        self.active_sessions: Dict[str, Any] = {}
        self.orchestrator_stats = self._default_stats()
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = self.config.get('cache_ttl_seconds', 1800)
        logger.info("IntelligentQAOrchestrator instance created. Call async initialize() to set up components.")

    def _initialize_component_placeholders(self):
        # (与之前一致)
        self.query_parser: Optional[SmartQueryParser] = None
        self.data_requirements_analyzer: Optional[DataRequirementsAnalyzer] = None
        self.financial_data_analyzer: Optional[FinancialDataAnalyzer] = None
        self.insight_generator: Optional[InsightGenerator] = None
        self.data_fetcher: Optional[SmartDataFetcher] = None
        self.current_data_processor: Optional[CurrentDataProcessor] = None
        self.historical_analysis_processor: Optional[HistoricalAnalysisProcessor] = None
        self.prediction_processor: Optional[PredictionProcessor] = None
        self.date_utils: Optional[DateUtils] = None
        self.financial_formatter: Optional[FinancialFormatter] = None
        self.chart_generator: Optional[UtilsChartGenerator] = None
        self.report_generator: Optional[ReportGenerator] = None
        self.conversation_manager: Optional[ConversationManager] = None

    def _default_stats(self) -> Dict[str, Any]:
        # (与之前一致)
        return {
            'total_queries': 0, 'successful_queries': 0, 'failed_queries': 0,
            'avg_processing_time': 0.0, 'avg_confidence_score': 0.0,
            'ai_collaboration_count': 0, 'cache_hits': 0, 'cache_misses': 0,
            'processor_usage': {}, 'error_types': {}
        }

    def _load_orchestrator_config(self) -> Dict[str, Any]:
        # (与之前一致)
        cfg = {
            'max_processing_time': getattr(self.app_config, 'MAX_PROCESSING_TIME', 120),
            'enable_parallel_processing': getattr(self.app_config, 'ENABLE_PARALLEL_PROCESSING', True),
            'enable_intelligent_caching': getattr(self.app_config, 'ENABLE_INTELLIGENT_CACHING', True),
            'enable_ai_collaboration': getattr(self.app_config, 'ENABLE_AI_COLLABORATION', True),
            'min_confidence_threshold': getattr(self.app_config, 'MIN_CONFIDENCE_THRESHOLD', 0.6),
            'min_confidence_threshold_parsing': 0.7,
            'enable_result_validation': getattr(self.app_config, 'ENABLE_RESULT_VALIDATION', True),
            'enable_quality_monitoring': getattr(self.app_config, 'ENABLE_QUALITY_MONITORING', True),
            'cache_ttl_seconds': getattr(self.app_config, 'CACHE_TTL', 1800),
            'max_cache_size': getattr(self.app_config, 'MAX_CACHE_SIZE', 100),
            'max_concurrent_queries': getattr(self.app_config, 'MAX_CONCURRENT_QUERIES', 50),
            'enable_smart_routing': getattr(self.app_config, 'ENABLE_SMART_ROUTING', True),
            'claude_timeout': getattr(self.app_config, 'CLAUDE_TIMEOUT', 60),
            'gpt_timeout': getattr(self.app_config, 'GPT_TIMEOUT', 40),
            'max_ai_retries': getattr(self.app_config, 'MAX_AI_RETRIES', 2),
            'enable_graceful_degradation': getattr(self.app_config, 'ENABLE_GRACEFUL_DEGRADATION', True),
            'fallback_response_enabled': getattr(self.app_config, 'FALLBACK_RESPONSE_ENABLED', True),
            'DEBUG': getattr(self.app_config, 'DEBUG', False),
            'version': getattr(self.app_config, 'VERSION', '2.1.9-fuller-methods-no-main-fixed-prompts')  # 更新版本
        }
        if hasattr(self.app_config, 'CLAUDE_API_KEY'): cfg['CLAUDE_API_KEY'] = self.app_config.CLAUDE_API_KEY
        if hasattr(self.app_config, 'OPENAI_API_KEY'): cfg['OPENAI_API_KEY'] = self.app_config.OPENAI_API_KEY
        if hasattr(self.app_config, 'DATABASE_CONFIG'): cfg['DATABASE_CONFIG'] = self.app_config.DATABASE_CONFIG
        api_connector_cfg = {}
        if hasattr(self.app_config, 'FINANCE_API_BASE_URL'): api_connector_cfg[
            'base_url'] = self.app_config.FINANCE_API_BASE_URL
        if hasattr(self.app_config, 'FINANCE_API_KEY'): api_connector_cfg['api_key'] = self.app_config.FINANCE_API_KEY
        if api_connector_cfg: cfg['api_connector_config'] = api_connector_cfg
        return cfg

    async def initialize(self):
        # (与之前一致，确保所有组件被正确创建)
        if self.initialized:
            logger.debug("Orchestrator already initialized.")
            return

        logger.info("🚀 开始初始化智能编排器组件...")
        start_init_time = time.time()
        try:
            if not self.claude_client and self.config.get('CLAUDE_API_KEY'):
                self.claude_client = ClaudeClient(api_key=self.config['CLAUDE_API_KEY'])
            if not self.gpt_client and self.config.get('OPENAI_API_KEY'):
                self.gpt_client = OpenAIClient(api_key=self.config['OPENAI_API_KEY'])

            if not self.claude_client: logger.warning("ClaudeClient未配置。")
            if not self.gpt_client: logger.warning("OpenAIClient未配置。")

            if not self.db_connector and self.config.get('DATABASE_CONFIG'):
                db_cfg_details = self.config['DATABASE_CONFIG']
                if db_cfg_details.get('user') and db_cfg_details.get('password') and db_cfg_details.get(
                        'host') and db_cfg_details.get('database'):
                    self.db_connector = create_database_connector(db_cfg_details)
                    logger.info(f"DatabaseConnector for host '{db_cfg_details.get('host')}' implicitly initialized.")
                else:
                    logger.error("DATABASE_CONFIG不完整。")
            elif not self.db_connector:
                logger.warning("DatabaseConnector未配置。对话历史等功能将使用内存模式或受限。")

            self.query_parser = create_smart_query_parser(self.claude_client, self.gpt_client)
            self.data_requirements_analyzer = create_data_requirements_analyzer(self.claude_client, self.gpt_client)
            fetcher_config = self.config.get('api_connector_config', {})
            self.data_fetcher = create_smart_data_fetcher(self.claude_client, self.gpt_client, fetcher_config)
            self.financial_data_analyzer = create_financial_data_analyzer(self.claude_client, self.gpt_client)
            self.insight_generator = create_insight_generator(self.claude_client, self.gpt_client)
            self.current_data_processor = CurrentDataProcessor(self.claude_client, self.gpt_client)
            self.historical_analysis_processor = HistoricalAnalysisProcessor(self.claude_client, self.gpt_client)
            self.prediction_processor = PredictionProcessor(self.claude_client, self.gpt_client)
            self.date_utils = create_date_utils(self.claude_client)
            self.financial_formatter = create_financial_formatter()
            self.chart_generator = create_utils_chart_generator()
            self.report_generator = create_report_generator()

            if self.db_connector:
                self.conversation_manager = create_conversation_manager(self.db_connector)
                logger.info("ConversationManager initialized with DB backend.")
            else:
                logger.warning("DatabaseConnector not available. ConversationManager using non-persistent in-memory.")
                self.conversation_manager = ConversationManager(database_connector=None)

            self._inject_dependencies()
            self.initialized = True
            init_duration = time.time() - start_init_time
            logger.info(f"✅ 智能编排器组件初始化完成 (耗时: {init_duration:.2f}s)。")
        except Exception as e:
            self.initialized = False
            logger.error(f"❌ 智能编排器初始化过程中发生严重错误: {str(e)}\n{traceback.format_exc()}")

    def _inject_dependencies(self):
        # (与之前一致)
        logger.debug("Injecting dependencies into components...")
        if not self.data_fetcher:
            logger.error("SmartDataFetcher is not initialized, cannot inject APIConnector.")
            return

        api_connector_instance = getattr(self.data_fetcher, 'api_connector', None)
        if not api_connector_instance and self.data_fetcher:
            logger.error("APIConnector is missing from SmartDataFetcher! This is critical.")
            api_conn_config_details = self.config.get('api_connector_config', {})
            if api_conn_config_details and api_conn_config_details.get('base_url'):
                from data.connectors.api_connector import create_enhanced_api_connector
                self.data_fetcher.api_connector = create_enhanced_api_connector(api_conn_config_details,
                                                                                self.claude_client, self.gpt_client)
                api_connector_instance = self.data_fetcher.api_connector
                logger.info("Dynamically re-initialized APIConnector for SmartDataFetcher during DI.")
            else:
                logger.error("Cannot dynamically re-init APIConnector for SmartDataFetcher due to missing config.")

        components_to_inject_into = [
            self.current_data_processor, self.historical_analysis_processor, self.prediction_processor,
            self.financial_data_analyzer, self.insight_generator,
            self.query_parser, self.data_requirements_analyzer, self.data_fetcher
        ]

        for component in components_to_inject_into:
            if not component: continue
            if api_connector_instance and isinstance(component, (
            CurrentDataProcessor, HistoricalAnalysisProcessor, PredictionProcessor, FinancialDataAnalyzer)):
                if not hasattr(component, 'api_connector') or component.api_connector is None:
                    component.api_connector = api_connector_instance

            if self.financial_data_analyzer and isinstance(component, (
            HistoricalAnalysisProcessor, PredictionProcessor, InsightGenerator)):
                if not hasattr(component, 'financial_data_analyzer') or component.financial_data_analyzer is None:
                    component.financial_data_analyzer = self.financial_data_analyzer

            financial_calculator_instance = getattr(self.financial_data_analyzer, 'financial_calculator',
                                                    None) if self.financial_data_analyzer else None
            if financial_calculator_instance and isinstance(component,
                                                            (HistoricalAnalysisProcessor, PredictionProcessor)):
                if not hasattr(component, 'financial_calculator') or component.financial_calculator is None:
                    component.financial_calculator = financial_calculator_instance

            if self.date_utils and (not hasattr(component, 'date_utils') or component.date_utils is None):
                component.date_utils = self.date_utils
        logger.debug("Dependency injection check complete.")

    # ========================================================================
    # ============= 核心流程方法 process_intelligent_query (保持不变) ============
    # ========================================================================
    async def process_intelligent_query(self, user_query: str, user_id: int = 0,
                                        conversation_id: Optional[str] = None,
                                        preferences: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        if not self.initialized:
            logger.critical("CRITICAL: Orchestrator not initialized. Attempting to initialize now.")
            await self.initialize()
            if not self.initialized:
                return self._handle_processing_error_sync(  # 使用已定义的同步错误处理器
                    str(uuid.uuid4()), f"init_fail_{int(time.time())}", user_query,
                    "系统核心组件初始化失败，请联系管理员。", 0.0, conversation_id
                )

        session_id = str(uuid.uuid4())
        query_id = f"q_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{hashlib.md5(user_query.encode('utf-8')).hexdigest()[:6]}"
        start_time = time.time()

        conversation_id_for_db: Optional[int] = None
        if conversation_id:
            try:
                conversation_id_for_db = int(conversation_id)
            except ValueError:
                logger.warning(f"QueryID: {query_id} - Invalid Conversation ID '{conversation_id}'.")
                conversation_id = None

        logger.info(
            f"🧠 QueryID: {query_id} - UserID: {user_id} - ConvStrID: {conversation_id} (DBIntID: {conversation_id_for_db}) - Processing: '{user_query[:100]}...'")
        self.orchestrator_stats['total_queries'] += 1

        if conversation_id_for_db and self.conversation_manager:
            try:
                user_msg_id = self.conversation_manager.add_message(conversation_id_for_db, True, user_query)
                logger.debug(
                    f"QueryID: {query_id} - User query (MsgID: {user_msg_id}) logged to ConvID {conversation_id_for_db}.")
            except Exception as conv_err:
                logger.error(
                    f"QueryID: {query_id} - Failed to log user query for ConvID {conversation_id_for_db}: {conv_err}")

        parsing_time, data_fetching_time, business_processing_time = 0.0, 0.0, 0.0
        insight_processing_time, response_formatting_time, visualization_processing_time = 0.0, 0.0, 0.0
        query_analysis: Optional[QueryAnalysisResult] = None
        data_acquisition_result: Optional[FetcherExecutionResult] = None
        business_result_data: Dict[str, Any] = {}
        insights_list: List[BusinessInsight] = []
        visualizations_data: List[Dict[str, Any]] = []
        response_text: str = ""
        routing_strategy: OrchestratorProcessingStrategy = OrchestratorProcessingStrategy.ERROR_HANDLING
        data_quality: float = 0.1

        try:
            if self.config.get('enable_intelligent_caching', True):
                current_context_for_cache = {}
                if conversation_id_for_db and self.conversation_manager:
                    current_context_for_cache = self.conversation_manager.get_context(conversation_id_for_db)
                cache_key_full_query = f"full_res_{self._generate_query_cache_key(user_query, current_context_for_cache)}"
                cached_full_result = self._get_cached_result(cache_key_full_query)
                if cached_full_result and isinstance(cached_full_result, ProcessingResult):
                    logger.info(f"QueryID: {query_id} - Full result cache HIT.")
                    self.orchestrator_stats['cache_hits'] += 1
                    cached_full_result.total_processing_time = time.time() - start_time
                    self._update_orchestrator_stats(cached_full_result)
                    if conversation_id_for_db and self.conversation_manager:
                        self._log_ai_response_to_conversation(conversation_id_for_db, cached_full_result, query_id)
                    return cached_full_result
            self.orchestrator_stats['cache_misses'] += 1

            parsing_start_time = time.time()
            query_analysis = await self._intelligent_query_parsing(user_query, conversation_id_for_db)
            parsing_time = time.time() - parsing_start_time
            logger.info(
                f"QueryID: {query_id} - 1. Query Parsing ({parsing_time:.3f}s): Complexity='{query_analysis.complexity.value}', Type='{query_analysis.query_type.value}'")

            routing_strategy = await self._determine_processing_strategy(query_analysis, preferences)
            logger.info(f"QueryID: {query_id} - 2. Processing Strategy: {routing_strategy.value}")

            data_acquisition_start_time = time.time()
            data_acquisition_result = await self._orchestrate_data_acquisition(query_analysis, routing_strategy)
            data_fetching_time = time.time() - data_acquisition_start_time
            data_quality = getattr(data_acquisition_result, 'confidence_level', 0.3)
            logger.info(
                f"QueryID: {query_id} - 3. Data Acquisition ({data_fetching_time:.3f}s): Quality={data_quality:.2f}, Status='{getattr(data_acquisition_result, 'execution_status', 'unknown').value if hasattr(getattr(data_acquisition_result, 'execution_status', None), 'value') else 'unknown'}'")

            if not data_acquisition_result or \
                    (not getattr(data_acquisition_result, 'fetched_data', None) and not getattr(data_acquisition_result,
                                                                                                'processed_data',
                                                                                                None)) or \
                    (hasattr(data_acquisition_result,
                             'execution_status') and data_acquisition_result.execution_status.value not in ['completed',
                                                                                                            'partial_success']):
                err_msg = f"Critical data acquisition failed. Status: {getattr(data_acquisition_result, 'execution_status', {}).get('value', 'unknown')}. Error: {getattr(data_acquisition_result, 'error_info', 'N/A')}"
                logger.error(f"QueryID: {query_id} - {err_msg}")
                raise ValueError(err_msg)

            business_processing_start_time = time.time()
            business_result_data = await self._orchestrate_business_processing(query_analysis, data_acquisition_result,
                                                                               routing_strategy)
            business_processing_time = time.time() - business_processing_start_time
            logger.info(
                f"QueryID: {query_id} - 4. Business Processing ({business_processing_time:.3f}s): Processor='{business_result_data.get('processor_used', 'N/A')}'")

            insight_generation_start_time = time.time()
            insights_list = await self._orchestrate_insight_generation(business_result_data, query_analysis,
                                                                       data_acquisition_result)
            insight_processing_time = time.time() - insight_generation_start_time
            logger.info(
                f"QueryID: {query_id} - 5. Insight Generation ({insight_processing_time:.3f}s): Found {len(insights_list)} insights.")

            response_formatting_start_time = time.time()
            response_text = await self._generate_intelligent_response(user_query, query_analysis, business_result_data,
                                                                      insights_list)
            response_formatting_time = time.time() - response_formatting_start_time
            logger.info(f"QueryID: {query_id} - 6. Response Text Generation ({response_formatting_time:.3f}s).")

            visualization_start_time = time.time()
            visualizations_data = await self._generate_visualizations(business_result_data, query_analysis,
                                                                      insights_list)
            visualization_processing_time = time.time() - visualization_start_time
            logger.info(
                f"QueryID: {query_id} - 7. Visualization Data Generation ({visualization_processing_time:.3f}s): Found {len(visualizations_data)} visuals.")

            total_ai_time = parsing_time
            biz_proc_metadata = business_result_data.get('metadata', {})
            total_ai_time += biz_proc_metadata.get('ai_processing_time', 0.0) if isinstance(biz_proc_metadata,
                                                                                            dict) else 0.0
            insight_gen_metadata = business_result_data.get('insight_generation_metadata',
                                                            {}) if business_result_data else {}
            total_ai_time += insight_gen_metadata.get('ai_time', insight_processing_time) if isinstance(
                insight_gen_metadata, dict) else insight_processing_time
            total_ai_time += response_formatting_time

            total_processing_time = time.time() - start_time
            overall_confidence = self._calculate_overall_confidence(query_analysis, business_result_data, insights_list,
                                                                    data_quality)

            final_result = ProcessingResult(
                session_id=session_id, query_id=query_id, success=True,
                response_text=response_text,
                insights=insights_list,
                key_metrics=self._extract_key_metrics(business_result_data),
                visualizations=visualizations_data,
                processing_strategy=routing_strategy,
                processors_used=self._get_processors_used(business_result_data, routing_strategy),
                ai_collaboration_summary=self._get_ai_collaboration_summary(query_analysis, business_result_data,
                                                                            insights_list),
                confidence_score=overall_confidence,
                data_quality_score=data_quality,
                response_completeness=self._calculate_response_completeness(business_result_data, insights_list),
                total_processing_time=total_processing_time,
                ai_processing_time=total_ai_time,
                data_fetching_time=data_fetching_time,
                processing_metadata={
                    'query_complexity': query_analysis.complexity.value,
                    'query_type': query_analysis.query_type.value,
                    'business_scenario': query_analysis.business_scenario.value if query_analysis.business_scenario else "N/A",
                    'data_sources_used': getattr(data_acquisition_result, 'data_sources_used', []),
                    'step_times': {
                        'parsing': round(parsing_time, 3), 'data_fetching': round(data_fetching_time, 3),
                        'business_processing': round(business_processing_time, 3),
                        'insight_generation': round(insight_processing_time, 3),
                        'response_formatting': round(response_formatting_time, 3),
                        'visualization': round(visualization_processing_time, 3)
                    },
                    'ai_models_invoked': self._get_ai_models_invoked_summary(query_analysis, business_result_data,
                                                                             insights_list)
                },
                conversation_id=conversation_id,
                query_analysis_snapshot=query_analysis.to_dict() if hasattr(query_analysis,
                                                                            'to_dict') else query_analysis.__dict__ if query_analysis else None
            )

            self._update_orchestrator_stats(final_result)
            if self.config.get('enable_intelligent_caching', True) and final_result.success:
                self._cache_result(cache_key_full_query, final_result)

            if conversation_id_for_db and self.conversation_manager:
                self._log_ai_response_to_conversation(conversation_id_for_db, final_result, query_id)

            logger.info(
                f"QueryID: {query_id} - ✅ 智能查询处理成功。总耗时: {total_processing_time:.3f}s, 置信度: {final_result.confidence_score:.2f}")
            return final_result

        except Exception as e:
            total_processing_time_on_error = time.time() - start_time
            logger.error(f"❌ QueryID: {query_id} - 智能查询处理主流程失败: {str(e)}\n{traceback.format_exc()}")
            return await self._handle_processing_error(
                session_id, query_id, user_query, str(e), total_processing_time_on_error, conversation_id
            )

    # ========================================================================
    # ============= 以下是补全的辅助方法 (串联核心逻辑) ========================
    # ========================================================================

        # 在 IntelligentQAOrchestrator 类中添加这些方法：

    def _generate_basic_response(self, business_result_data: Dict[str, Any],
                                     insights_list: List[BusinessInsight]) -> str:
            """
            在AI响应生成失败时，提供一个基于模板的、更简单的中文文本响应。
            """
            logger.warning("Generating basic (non-AI) response due to previous errors or lack of AI client.")
            response_parts = ["根据系统分析：\n"]

            key_metrics = self._extract_key_metrics(business_result_data)  # 假设此方法已实现
            if key_metrics:
                response_parts.append("关键指标：")
                # 确保 FinancialFormatter 已初始化并可用
                # for name, value in list(key_metrics.items())[:5]: # 最多显示5个
                #     response_parts.append(f"  - {name}: {value}") # value 应已由 _extract_key_metrics 格式化
                # 简化版，不依赖 _extract_key_metrics 完全正确：
                metric_items = []
                temp_metrics_to_show = {}
                primary_res_content = business_result_data.get('primary_result', {})
                if isinstance(primary_res_content, dict):
                    if 'key_metrics' in primary_res_content: temp_metrics_to_show.update(
                        primary_res_content['key_metrics'])
                    if 'metrics' in primary_res_content: temp_metrics_to_show.update(primary_res_content['metrics'])
                    if 'main_prediction' in primary_res_content: temp_metrics_to_show.update(
                        primary_res_content['main_prediction'])

                for name, value in list(temp_metrics_to_show.items())[:5]:
                    val_str = str(value)
                    if self.financial_formatter and isinstance(value, (int, float)):  # 尝试格式化
                        try:
                            if "rate" in name.lower() or "ratio" in name.lower() or "%" in val_str:
                                val_str = self.financial_formatter.format_percentage(
                                    float(val_str.replace('%', '')) / 100 if "%" in val_str else float(val_str))
                            else:
                                val_str = self.financial_formatter.format_currency(float(value))
                        except:
                            pass  # 保持原样
                    metric_items.append(f"  - {name}: {val_str}")
                if metric_items:
                    response_parts.extend(metric_items)
                else:
                    response_parts.append("  暂无关键指标可直接展示。")
            else:
                primary_res = business_result_data.get('primary_result')
                main_ans_candidate = "分析已执行，但无特定指标输出。"
                if isinstance(primary_res, dict):
                    main_ans_candidate = primary_res.get('main_answer', primary_res.get('summary', main_ans_candidate))
                elif primary_res:
                    main_ans_candidate = getattr(primary_res, 'main_answer',
                                                 getattr(primary_res, 'summary', main_ans_candidate))
                response_parts.append(f"主要信息: {str(main_ans_candidate)[:300]}")

            if insights_list:
                response_parts.append("\n重要洞察：")
                for i, insight in enumerate(insights_list[:3], 1):  # 最多3条
                    title = getattr(insight, 'title', f"洞察{i}")
                    summary = getattr(insight, 'summary', "请查看详细分析。")
                    response_parts.append(f"  {i}. **{title}**: {summary}")
            else:
                response_parts.append("\n暂无特别的业务洞察。")

            confidence = business_result_data.get('confidence_score', 0.5)
            data_quality_src = business_result_data.get('data_quality_score',
                                                        getattr(business_result_data.get('data_acquisition_result', {}),
                                                                'confidence_level', 0.5)
                                                        )
            response_parts.append(f"\n本次分析的整体置信度约为 {confidence:.0%}，数据质量评分为 {data_quality_src:.0%}")
            if confidence < 0.7 or data_quality_src < 0.7:
                response_parts.append("请注意，由于数据质量或分析复杂性，结果可能存在一定不确定性。")

            return "\n".join(response_parts)

    def _summarize_business_result_for_ai(self, business_result_data: Dict[str, Any]) -> Dict[str, Any]:
            """为AI准备业务结果的摘要，以便AI能更好地整合信息生成自然语言回复。"""
            summary = {}
            if not business_result_data or not isinstance(business_result_data, dict):
                logger.warning("_summarize_business_result_for_ai: business_result_data is empty or invalid.")
                return {"摘要": "业务处理阶段未返回有效结果。"}

            summary['processing_type'] = business_result_data.get('processing_type')
            summary['confidence_score'] = round(business_result_data.get('confidence_score', 0.0), 2)

            primary_res_payload = business_result_data.get('primary_result')  # This is object or dict of objects

            # 辅助函数，将处理器返回的对象转换为字典以便摘要
            def _convert_to_dict_for_summary(obj: Any) -> Optional[Dict[str, Any]]:
                if hasattr(obj, 'to_dict') and callable(obj.to_dict): return obj.to_dict()
                if hasattr(obj, '__dict__'): return obj.__dict__  # For dataclasses
                if isinstance(obj, dict): return obj
                logger.warning(f"Cannot convert object of type {type(obj)} to dict for summary.")
                return None

            if isinstance(primary_res_payload, dict) and 'multi_results' in primary_res_payload:
                summary['multi_analysis_highlights'] = {}
                for key, res_obj_or_dict in primary_res_payload['multi_results'].items():
                    res_dict = _convert_to_dict_for_summary(res_obj_or_dict)
                    if res_dict and 'error' not in res_dict:
                        single_summary = {}
                        # 从 CurrentDataResponse, HistoricalAnalysisResponse, PredictionResponse 中提取关键信息
                        if 'main_answer' in res_dict: single_summary['answer_snippet'] = str(res_dict['main_answer'])[
                                                                                         :100] + "..."

                        metrics_src = res_dict.get('key_metrics') or res_dict.get('metrics') or res_dict.get(
                            'main_prediction')
                        if isinstance(metrics_src, dict): single_summary['top_metrics_sample'] = dict(
                            list(metrics_src.items())[:2])  # 最多2个示例指标

                        if 'trend_summary' in res_dict and res_dict['trend_summary']: single_summary['trend_snapshot'] = \
                        res_dict['trend_summary']
                        if 'key_findings' in res_dict and isinstance(res_dict['key_findings'], list) and res_dict[
                            'key_findings']:
                            single_summary['top_finding'] = str(res_dict['key_findings'][0])[:150] + "..."

                        if single_summary:  # 只添加有内容的摘要
                            summary['multi_analysis_highlights'][key] = single_summary
            else:  # 单个处理器的结果 (对象或其字典)
                res_dict = _convert_to_dict_for_summary(primary_res_payload)
                if res_dict:
                    if 'main_answer' in res_dict: summary['main_answer_snippet'] = str(res_dict['main_answer'])[
                                                                                   :200] + "..."
                    metrics_src = res_dict.get('key_metrics') or res_dict.get('metrics') or res_dict.get(
                        'main_prediction')
                    if isinstance(metrics_src, dict): summary['main_metrics_sample'] = dict(
                        list(metrics_src.items())[:3])  # 最多3个示例指标
                    if 'trend_summary' in res_dict and res_dict['trend_summary']: summary['trend_snapshot'] = res_dict[
                        'trend_summary']
                    if 'key_findings' in res_dict and isinstance(res_dict['key_findings'], list) and res_dict[
                        'key_findings']:
                        summary['top_finding'] = str(res_dict['key_findings'][0])[:150] + "..."
                elif primary_res_payload:  # 如果转换字典失败，但对象存在
                    summary['raw_primary_result_type'] = str(type(primary_res_payload))

            if not summary.get('main_answer_snippet') and not summary.get('multi_analysis_highlights'):
                summary['general_summary'] = "系统已完成多项分析，具体细节请参考指标和洞察部分。"

            return summary

    async def _generate_fallback_response(self, user_query: str, error: str) -> str:
            """在处理失败时生成一个用户友好的中文降级响应。"""
            logger.info(f"为查询生成降级响应: '{user_query[:50]}...'，错误: {error[:100]}")

            # 尝试使用AI（如果一个模型可用且错误不是AI服务本身的问题）生成更智能的错误提示
            ai_client_for_fallback = None
            error_lower = error.lower()

            # 优先选择与导致错误的模型不同的模型，或者如果错误与AI无关，则优先Claude
            if "claude" not in error_lower and self.claude_client:
                ai_client_for_fallback = self.claude_client
            elif ("gpt" not in error_lower and "openai" not in error_lower) and self.gpt_client:
                ai_client_for_fallback = self.gpt_client
            elif self.claude_client:  # 如果两个都没在错误信息中，默认尝试Claude
                ai_client_for_fallback = self.claude_client
            elif self.gpt_client:  # 其次尝试GPT
                ai_client_for_fallback = self.gpt_client

            if ai_client_for_fallback and self.config.get('fallback_response_enabled', True):
                try:
                    # 中文提示词
                    fallback_prompt = f"""
                    用户之前的查询是：“{user_query}”
                    系统在尝试处理这个查询时遇到了一个内部错误，错误信息摘要如下（此摘要仅供您参考，不要直接展示给用户）：
                    “{error[:150]}...”

                    请你扮演一个乐于助人且专业的AI客服，用简体中文给用户生成一个简洁、友好的回复。
                    您的回复应该包含以下要点：
                    1. 对用户表示歉意，说明他们的请求未能成功处理。
                    2. 无需向用户复述具体的错误信息摘要，除非是非常明确的用户输入参数问题（但这种情况很少）。
                    3. 建议用户可以尝试的操作，例如：
                        - 稍后重试。
                        - 尝试用不同的方式表述他们的问题，或者简化查询。
                        - （如果适用）检查他们的输入是否符合预期格式。
                    4. 如果错误看起来是临时的系统性问题，可以暗示技术团队可能已知晓并在处理。
                    请确保回复语气专业、安抚用户，并且不要提供虚假承诺。回复长度控制在1-2句话。
                    """
                    ai_response_data = None
                    if isinstance(ai_client_for_fallback, ClaudeClient) and hasattr(ai_client_for_fallback,
                                                                                    'generate_text'):
                        ai_response_data = await ai_client_for_fallback.generate_text(fallback_prompt, max_tokens=200)
                    elif isinstance(ai_client_for_fallback, OpenAIClient) and hasattr(ai_client_for_fallback,
                                                                                      'generate_completion'):
                        ai_response_data = await ai_client_for_fallback.generate_completion(fallback_prompt,
                                                                                            max_tokens=200)

                    if ai_response_data and ai_response_data.get('success', False):
                        content = ai_response_data.get('text', ai_response_data.get('response',
                                                                                    ai_response_data.get('completion')))
                        if content and isinstance(content, str) and content.strip():
                            logger.info("AI成功生成了降级响应。")
                            return content.strip()
                except Exception as ai_fallback_err:
                    logger.error(f"使用AI生成降级响应时也发生错误: {ai_fallback_err}")

            # 如果AI生成降级响应失败，或没有AI客户端，则使用基于规则的简单中文响应
            query_lower = user_query.lower()
            if any(kw in query_lower for kw in ["余额", "balance"]):
                return "抱歉，系统当前无法查询余额信息。我们的工程师正在紧急处理，请您稍后重试或联系我们的支持团队。"
            elif any(kw in query_lower for kw in ["趋势", "trend", "历史", "history"]):
                return "抱歉，历史数据分析功能暂时遇到一些技术问题。我们正在努力修复中，请您稍后再试。"
            elif any(kw in query_lower for kw in ["预测", "predict", "预计", "forecast"]):
                return "抱歉，预测服务当前暂时不可用。请您稍等片刻再尝试您的预测请求。"
            return f"处理您的查询 '{user_query[:30]}...' 时遇到一个问题。我们的技术团队已经注意到此情况，并将尽快解决。给您带来不便，我们深表歉意，请稍后重试。"
    def _calculate_overall_confidence(self, query_analysis: Optional[QueryAnalysisResult],
                                      business_result_data: Dict[str, Any],
                                      insights_list: List[BusinessInsight],
                                      data_quality_score: float) -> float:
        """
        计算整个查询处理流程的整体置信度。
        综合考虑查询解析、数据质量、业务处理和洞察的置信度。
        """
        factors: List[float] = []

        # 1. 查询解析置信度
        if query_analysis and hasattr(query_analysis, 'confidence_score'):
            factors.append(float(query_analysis.confidence_score))
        else:
            factors.append(0.5)  # 如果没有查询分析结果，给一个较低的默认值

        # 2. 数据质量评分 (由 SmartDataFetcher 提供)
        factors.append(float(data_quality_score))

        # 3. 业务处理置信度
        # business_result_data 包含了 'confidence_score' 键
        if business_result_data and isinstance(business_result_data, dict):
            factors.append(float(business_result_data.get('confidence_score', 0.7)))  # 默认0.7如果业务处理层未提供
        else:
            factors.append(0.5)

        # 4. 洞察质量 (基于洞察列表和单个洞察的置信度)
        if insights_list:
            insights_confidences = [
                float(getattr(i, 'confidence_score', 0.7)) for i in insights_list if hasattr(i, 'confidence_score')
            ]
            if insights_confidences:
                factors.append(sum(insights_confidences) / len(insights_confidences))
            else:
                factors.append(0.6)  # 有洞察但没有具体置信度
        else:
            factors.append(0.5)  # 没有洞察

        # 定义各因素的权重，总和为1
        # 权重可以根据实际情况调整，例如，数据质量可能比洞察置信度更重要
        # 顺序：解析、数据质量、业务处理、洞察
        weights = [0.20, 0.30, 0.30, 0.20]

        if len(factors) == len(weights):
            weighted_sum = sum(factor * weight for factor, weight in zip(factors, weights))
            final_confidence = round(weighted_sum, 3)
        elif factors:  # 如果权重和因素数量不匹配（不应发生），则取简单平均
            logger.warning(
                f"Confidence factors count ({len(factors)}) does not match weights count ({len(weights)}). Using simple average.")
            final_confidence = round(sum(factors) / len(factors), 3)
        else:  # 不应发生，但作为保护
            final_confidence = 0.3

        logger.debug(f"Calculated overall confidence: {final_confidence}, based on factors: {factors}")
        return final_confidence

    def _extract_key_metrics(self, business_result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从业务处理结果中提取关键指标，并使用 FinancialFormatter 进行格式化。
        返回一个字典，键是指标名称，值是格式化后的指标值（字符串）。
        """
        raw_metrics: Dict[str, Any] = {}  # 存储原始提取的指标
        if not business_result_data or not isinstance(business_result_data, dict):
            logger.debug("_extract_key_metrics: business_result_data is empty or invalid.")
            return {}

        primary_res_content = business_result_data.get('primary_result')
        logger.debug(f"_extract_key_metrics: primary_res_content type is {type(primary_res_content)}")

        def _extract_from_single_source(source_dict: Any, prefix: str = ""):
            """辅助函数，从单个结果对象或字典中提取指标。"""
            if not source_dict: return

            data_to_search: Optional[Dict[str, Any]] = None
            if hasattr(source_dict, 'to_dict') and callable(source_dict.to_dict):  # 处理dataclass实例
                data_to_search = source_dict.to_dict()
            elif hasattr(source_dict, '__dict__'):  # 处理其他普通对象实例
                data_to_search = source_dict.__dict__
            elif isinstance(source_dict, dict):  # 如果已经是字典
                data_to_search = source_dict

            if not data_to_search:
                logger.debug(
                    f"_extract_from_single_source: source_dict (prefix: {prefix}) is not dict-convertible or is None.")
                return

            # 查找常见的指标容器键
            # 您的处理器响应对象（CurrentDataResponse, HistoricalAnalysisResponse, PredictionResponse）
            # 中包含 'key_metrics', 'metrics', 'main_prediction' 等字段
            metric_container_keys = ['key_metrics', 'metrics', 'main_prediction', 'predictions_summary',
                                     'data']  # 'data' for CurrentDataResponse

            found_metrics_in_container = False
            for container_key in metric_container_keys:
                if container_key in data_to_search and isinstance(data_to_search[container_key], dict):
                    for k, v in data_to_search[container_key].items():
                        metric_name_to_store = f"{prefix}{k}".strip('_')
                        if metric_name_to_store in raw_metrics and raw_metrics[metric_name_to_store] != v:
                            logger.debug(
                                f"Metric conflict for '{metric_name_to_store}'. Current: {raw_metrics[metric_name_to_store]}, New: {v}. Making key unique.")
                            metric_name_to_store = f"{prefix}{container_key}_{k}".strip('_')  # 增加唯一性
                        raw_metrics[metric_name_to_store] = v
                    found_metrics_in_container = True
                    # 如果一个容器找到了指标，可能不需要再检查此对象的其他容器键了，
                    # 但这取决于您的数据结构设计。如果多个容器都可能包含独立指标，则移除break。
                    # break

            # 如果没有在标准容器中找到，并且data_to_search本身就是顶层指标 (例如CurrentDataResponse的data字段)
            if not found_metrics_in_container and data_to_search and container_key == 'data' and 'key_metrics' not in data_to_search and 'metrics' not in data_to_search:
                for k, v in data_to_search.items():  # 假设顶层键值对是指标
                    # 避免提取非指标的元数据等
                    if isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and len(v) < 5 and all(
                            isinstance(i, (str, int, float)) for i in v)):  # 简单类型或短列表
                        metric_name_to_store = f"{prefix}{k}".strip('_')
                        if metric_name_to_store in raw_metrics and raw_metrics[metric_name_to_store] != v:
                            metric_name_to_store = f"{prefix}top_level_{k}".strip('_')
                        raw_metrics[metric_name_to_store] = v

        if isinstance(primary_res_content, dict):
            if 'multi_results' in primary_res_content and isinstance(primary_res_content['multi_results'], dict):
                logger.debug("_extract_key_metrics: Processing multi_results.")
                for proc_name, proc_res_obj_or_dict in primary_res_content['multi_results'].items():
                    if proc_res_obj_or_dict and not (
                            isinstance(proc_res_obj_or_dict, dict) and 'error' in proc_res_obj_or_dict):
                        _extract_from_single_source(proc_res_obj_or_dict, prefix=f"{proc_name}_")
            else:  # Single processor result (可能已经是字典，也可能是对象)
                logger.debug("_extract_key_metrics: Processing single primary_result (dict or unknown obj).")
                _extract_from_single_source(primary_res_content)
        elif primary_res_content:  # It's a single response object (not a dict from multi_results)
            logger.debug(
                f"_extract_key_metrics: Processing single primary_result (object type: {type(primary_res_content)}).")
            _extract_from_single_source(primary_res_content, prefix="main_")

        # 应用 FinancialFormatter 进行格式化
        if self.financial_formatter:
            formatted_metrics: Dict[str, Any] = {}
            logger.debug(f"_extract_key_metrics: Applying financial formatting to {len(raw_metrics)} raw metrics.")
            for k, v_original in raw_metrics.items():
                v_formatted = v_original  # 默认使用原始值
                try:
                    # 尝试将值转换为可格式化的数值
                    numeric_value_for_formatting: Optional[Union[int, float]] = None
                    if isinstance(v_original, (int, float)):
                        numeric_value_for_formatting = v_original
                    elif isinstance(v_original, str):
                        val_str_cleaned = v_original.strip().replace('%', '').replace('¥', '').replace(',', '').replace(
                            '￥', '')
                        if val_str_cleaned.replace('.', '', 1).replace('-', '', 1).isdigit():
                            numeric_value_for_formatting = float(val_str_cleaned)

                    if numeric_value_for_formatting is not None:
                        key_lower = k.lower()
                        original_str_lower = str(v_original).lower()  # 用于检查原始字符串中的 '%' 等

                        if "rate" in key_lower or "ratio" in key_lower or "growth" in key_lower or "change" in key_lower or "%" in original_str_lower:
                            # 如果原始字符串含 '%', 假设它是百分点 (例如 '5%')，否则是小数 (例如 0.05)
                            value_to_format_as_percentage = numeric_value_for_formatting / 100.0 if "%" in original_str_lower else numeric_value_for_formatting
                            v_formatted = self.financial_formatter.format_percentage(value_to_format_as_percentage)
                        elif any(curr_kw in key_lower for curr_kw in
                                 ["balance", "amount", "value", "inflow", "outflow", "资金", "金额", "capital", "fund",
                                  "asset", "revenue", "profit", "cost", "余额", "入金", "出金"]):
                            v_formatted = self.financial_formatter.format_currency(numeric_value_for_formatting)
                        else:  # 其他数值的默认格式化
                            v_formatted = f"{numeric_value_for_formatting:,.2f}" if isinstance(
                                numeric_value_for_formatting, float) else f"{numeric_value_for_formatting:,}"
                    # else v_formatted 保持 v_original (非数值类型)
                    formatted_metrics[k] = v_formatted
                except Exception as fmt_e:
                    logger.warning(
                        f"Financial formatting failed for metric '{k}' (value: '{v_original}'): {fmt_e}. Using original value.")
                    formatted_metrics[k] = v_original  # 出错时回退到原始值
            logger.debug(f"_extract_key_metrics: Returning {len(formatted_metrics)} formatted metrics.")
            return formatted_metrics

        logger.debug(
            f"_extract_key_metrics: Returning {len(raw_metrics)} unformatted metrics (FinancialFormatter not available or no numeric values).")
        return raw_metrics  # 如果没有 formatter，返回原始提取的指标

    def _get_processors_used(self, business_result_data: Dict[str, Any], strategy: OrchestratorProcessingStrategy) -> \
    List[str]:
        """
        从业务处理结果中提取实际使用的处理器名称列表。
        """
        if not business_result_data or not isinstance(business_result_data, dict):
            logger.warning("_get_processors_used: business_result_data is empty or invalid.")
            return []

        # 'processor_used' 字段应由 _orchestrate_business_processing 方法在返回时设置
        # 它通常是一个包含处理器类名的字符串，用逗号分隔，或者是类名列表
        processor_info = business_result_data.get('processor_used')  # 这个键是在 _orchestrate_business_processing 中设置的

        if isinstance(processor_info, str) and processor_info:
            # 分割逗号分隔的字符串，并去除两端空格
            return [name.strip() for name in processor_info.split(',') if name.strip()]
        elif isinstance(processor_info, list):
            # 确保列表中的每个元素都是字符串
            return [str(name).strip() for name in processor_info if str(name).strip()]

        # 如果 'processor_used' 字段缺失或格式不正确，尝试根据策略推断 (这只是一个后备方案)
        logger.warning(
            f"'_get_processors_used' could not determine processors from 'processor_used' field (value: {processor_info}). Falling back to strategy-based inference.")
        if strategy == OrchestratorProcessingStrategy.DIRECT_RESPONSE:
            return [
                self.current_data_processor.__class__.__name__ if self.current_data_processor else "CurrentDataProcessor"]
        if strategy == OrchestratorProcessingStrategy.SINGLE_PROCESSOR:
            # 对于单处理器，如果没有明确记录，很难精确知道是哪个。
            # _orchestrate_business_processing 应该设置 processor_used。
            # 这里返回一个通用占位符或尝试从 primary_result 的类型推断。
            primary_res = business_result_data.get('primary_result')
            if isinstance(primary_res, CurrentDataResponse): return [CurrentDataProcessor.__class__.__name__]
            if isinstance(primary_res, HistoricalAnalysisResponse): return [
                HistoricalAnalysisProcessor.__class__.__name__]
            if isinstance(primary_res, PredictionResponse): return [PredictionProcessor.__class__.__name__]
            return ["UnknownSingleProcessor"]
        if strategy == OrchestratorProcessingStrategy.MULTI_PROCESSOR:
            # 提取 multi_results 中的键名作为处理器标识符
            multi_res = business_result_data.get('primary_result', {}).get('multi_results', {})
            if isinstance(multi_res, dict):
                return list(multi_res.keys())  # 例如 ['current_data_analysis', 'historical_analysis']
            return ["MultipleProcessorsInferred"]
        if strategy == OrchestratorProcessingStrategy.FULL_PIPELINE:
            multi_res = business_result_data.get('primary_result', {}).get('multi_results', {})
            processors = []
            if isinstance(multi_res, dict): processors.extend(list(multi_res.keys()))
            if 'financial_deep_dive' in processors:  # 通常 financial_deep_dive 是 FinancialDataAnalyzer
                # Remove and add proper class name if available
                if self.financial_data_analyzer:
                    processors = [p for p in processors if p != 'financial_deep_dive']
                    processors.append(self.financial_data_analyzer.__class__.__name__)
            return list(set(processors))  # 去重

        return ["UnknownStrategyOrNoProcessors"]

    def _get_ai_collaboration_summary(self, query_analysis: Optional[QueryAnalysisResult],
                                      business_result_data: Dict[str, Any],
                                      insights_list: List[BusinessInsight]) -> Dict[str, Any]:
        """
        总结AI模型（如Claude, GPT）在整个处理流程中的使用情况。
        更准确的统计需要各组件在其返回的元数据中报告AI使用情况。
        """
        claude_invoked_count = 0
        gpt_invoked_count = 0
        other_ai_invoked_count = 0  # 为未来扩展

        # 1. 查询解析阶段的AI使用
        if query_analysis and hasattr(query_analysis, 'ai_collaboration_plan') and \
                isinstance(query_analysis.ai_collaboration_plan, dict):
            plan = query_analysis.ai_collaboration_plan
            if plan.get('primary_ai', '').lower() == 'claude' or plan.get('secondary_ai', '').lower() == 'claude' or \
                    (isinstance(plan.get('claude_tasks'), list) and len(plan['claude_tasks']) > 0):
                claude_invoked_count += 1
            if plan.get('primary_ai', '').lower() == 'gpt' or plan.get('secondary_ai', '').lower() == 'gpt' or \
                    (isinstance(plan.get('gpt_tasks'), list) and len(plan['gpt_tasks']) > 0):
                gpt_invoked_count += 1
        elif query_analysis:  # 如果没有详细计划，但解析置信度高，也可能用了AI
            if query_analysis.confidence_score > 0.7 and (self.claude_client or self.gpt_client):
                # 无法区分是哪个，假设是主AI (e.g., Claude)
                if self.claude_client:
                    claude_invoked_count += 1
                elif self.gpt_client:
                    gpt_invoked_count += 1

        # 2. 业务处理阶段的AI使用 (需要处理器在其返回的元数据中报告)
        # 例如， business_result_data['metadata']['ai_models_used_in_processing'] = ['claude', 'gpt']
        biz_proc_meta = business_result_data.get('metadata', {})
        if isinstance(biz_proc_meta, dict):
            models_in_biz = biz_proc_meta.get('ai_models_used_in_processing', [])  # 期望是列表
            if 'claude' in models_in_biz: claude_invoked_count += 1
            if 'gpt' in models_in_biz: gpt_invoked_count += 1
            # 也可以从 primary_result.multi_results 的每个结果的metadata中累加

        # 3. 洞察生成阶段的AI使用
        # InsightGenerator 通常强依赖AI
        if insights_list:  # 如果生成了洞察
            insight_gen_meta = business_result_data.get('insight_generation_metadata', {})
            if isinstance(insight_gen_meta, dict) and insight_gen_meta.get('ai_model_used'):
                model_str = str(insight_gen_meta['ai_model_used']).lower()
                if 'claude' in model_str:
                    claude_invoked_count += 1
                elif 'gpt' in model_str:
                    gpt_invoked_count += 1
            elif self.claude_client:  # 默认InsightGenerator使用Claude
                claude_invoked_count += 1
            elif self.gpt_client:
                gpt_invoked_count += 1

        # 4. 最终响应文本生成阶段的AI使用
        # _generate_intelligent_response 会选择一个客户端
        if self.claude_client or self.gpt_client:  # 如果有任何AI客户端用于响应生成
            if self.claude_client and not self.gpt_client:
                claude_invoked_count += 1  # 假设优先Claude
            elif self.gpt_client and not self.claude_client:
                gpt_invoked_count += 1
            elif self.claude_client and self.gpt_client:
                claude_invoked_count += 1  # 假设优先Claude

        claude_actually_used = claude_invoked_count > 0
        gpt_actually_used = gpt_invoked_count > 0

        collaboration_level = "none"
        if claude_actually_used and gpt_actually_used:
            collaboration_level = "dual_ai_collaboration"
        elif claude_actually_used or gpt_actually_used:
            collaboration_level = "single_ai_assist"

        return {
            'claude_used_in_process': claude_actually_used,
            'gpt_used_in_process': gpt_actually_used,
            'claude_invocation_count_estimate': claude_invoked_count,  # 估算调用次数
            'gpt_invocation_count_estimate': gpt_invoked_count,
            'collaboration_level': collaboration_level,
            'ai_enhanced_parsing': getattr(query_analysis, 'confidence_score', 0.0) > 0.75 and \
                                   getattr(query_analysis, 'processing_metadata', {}).get(
                                       'parser_status') != 'fallback_rule_based' if query_analysis else False,
            'ai_generated_insights': bool(insights_list),
        }
    async def _intelligent_query_parsing(self, user_query: str,
                                         conversation_id_for_db: Optional[int] = None) -> QueryAnalysisResult:
        """
        AI驱动的智能查询解析，使用 SmartQueryParser。
        获取对话上下文，调用查询解析器，并处理缓存。
        """
        method_start_time = time.time()
        logger.debug(
            f"Orchestrator: Calling QueryParser for '{user_query[:50]}...' with ConvID_DB: {conversation_id_for_db}")
        if not self.query_parser:
            logger.error("QueryParser not initialized in Orchestrator. Cannot parse query.")
            return await self._fallback_query_parsing(user_query)

        context_for_parser: Dict[str, Any] = {}
        if conversation_id_for_db and self.conversation_manager:
            try:
                context_for_parser = self.conversation_manager.get_context(
                    conversation_id_for_db)  # Implemented in ConversationManager
                logger.debug(
                    f"Context for QueryParser (ConvID {conversation_id_for_db}): Found {len(context_for_parser.get('recent_history', []))} history messages.")
            except Exception as e_ctx:
                logger.error(
                    f"Failed to get context for ConvID {conversation_id_for_db} from ConversationManager: {e_ctx}")

        cache_key = self._generate_query_cache_key(user_query, context_for_parser)  # Implemented
        cached_result = self._get_cached_result(cache_key)  # Implemented
        if cached_result and isinstance(cached_result, QueryAnalysisResult):
            self.orchestrator_stats['cache_hits'] += 1
            logger.info(f"QueryParser cache hit for: '{user_query[:50]}...'")
            return cached_result

        self.orchestrator_stats['cache_misses'] += 1
        logger.info(f"QueryParser cache miss. Executing AI parsing for: '{user_query[:50]}...'")

        try:
            query_analysis_result: QueryAnalysisResult = await self.query_parser.parse_complex_query(user_query,
                                                                                                     context_for_parser)
            if query_analysis_result and query_analysis_result.confidence_score > self.config.get(
                    'min_confidence_threshold_parsing_for_cache', 0.5):
                self._cache_result(cache_key, query_analysis_result)  # Implemented
            elif not query_analysis_result:
                logger.warning(f"QueryParser returned None for query: '{user_query[:50]}...'. Using fallback.")
                query_analysis_result = await self._fallback_query_parsing(user_query)  # Implemented
        except Exception as e_parse:
            logger.error(f"Error during query parsing by SmartQueryParser: {e_parse}\n{traceback.format_exc()}")
            query_analysis_result = await self._fallback_query_parsing(user_query)

        logger.debug(
            f"Query parsing took {time.time() - method_start_time:.3f}s. Complexity: {query_analysis_result.complexity.value if query_analysis_result else 'N/A'}")
        return query_analysis_result

    def _generate_query_cache_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        为查询和可选的上下文生成一个一致的、可用于缓存的哈希键。
        使用 CustomJSONEncoder 处理 context 中可能存在的复杂对象（如 datetime）。
        """
        try:
            # 对上下文进行规范化排序，以确保相同内容的上下文生成相同的键
            context_str = json.dumps(context or {}, sort_keys=True, cls=CustomJSONEncoder, ensure_ascii=False)
        except TypeError as te:
            logger.warning(
                f"Failed to serialize context for cache key due to TypeError: {te}. Using empty context for key.")
            context_str = "{}"  # Fallback to empty context string if serialization fails

        cache_data = f"{query}_{context_str}"
        return hashlib.md5(cache_data.encode('utf-8')).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Union[QueryAnalysisResult, ProcessingResult]]:
        """
        从缓存中获取结果。如果找到且未过期，则返回缓存数据。
        否则，移除过期条目并返回 None。
        """
        if not self.config.get('enable_intelligent_caching', False):
            return None

        if cache_key in self.result_cache:
            cache_entry = self.result_cache[cache_key]
            if (time.time() - cache_entry['timestamp']) < self.cache_ttl:
                logger.debug(f"Cache HIT for key: {cache_key[:10]}...")
                self.orchestrator_stats['cache_hits'] = self.orchestrator_stats.get('cache_hits', 0) + 1
                return cache_entry['data']
            else:
                logger.debug(f"Cache EXPIRED for key: {cache_key[:10]}... Removing.")
                del self.result_cache[cache_key]

        logger.debug(f"Cache MISS for key: {cache_key[:10]}...")
        self.orchestrator_stats['cache_misses'] = self.orchestrator_stats.get('cache_misses', 0) + 1
        return None

    def _cache_result(self, cache_key: str, result: Union[QueryAnalysisResult, ProcessingResult]):
        """
        将结果（QueryAnalysisResult 或 ProcessingResult）存入缓存。
        管理缓存大小，如果超出限制则移除最旧的条目。
        """
        if not self.config.get('enable_intelligent_caching', False):
            return

        max_cache_size = self.config.get('max_cache_size', 100)
        if len(self.result_cache) >= max_cache_size:
            try:
                # 简单的LRU近似：移除时间戳最早的条目
                oldest_key = min(self.result_cache, key=lambda k: self.result_cache[k]['timestamp'])
                del self.result_cache[oldest_key]
                logger.debug(f"Cache max size ({max_cache_size}) reached. Removed oldest entry: {oldest_key[:10]}...")
            except ValueError:  # 缓存可能在并发情况下变空
                pass
            except Exception as e:
                logger.error(f"Error during cache eviction: {e}")

        self.result_cache[cache_key] = {
            'data': result,  # 可以是 QueryAnalysisResult 或 ProcessingResult
            'timestamp': time.time()
        }
        logger.debug(f"Cached result for key: {cache_key[:10]}... Cache size: {len(self.result_cache)}")

    def _update_orchestrator_stats(self, result: ProcessingResult):
        """更新编排器的性能和使用统计。"""
        if not isinstance(result, ProcessingResult):
            logger.warning(f"Attempted to update stats with invalid result type: {type(result)}")
            return

        # total_queries 在 process_intelligent_query 开始时已增加
        total_queries = self.orchestrator_stats.get('total_queries', 0)
        if total_queries == 0:
            logger.warning(
                "Total queries is 0 in stats, cannot calculate averages. This might happen if called before total_queries is incremented.")
            # Potentially increment here if it's guaranteed this method is called once per query
            # self.orchestrator_stats['total_queries'] = 1
            # total_queries = 1
            return  # Avoid division by zero if total_queries is still 0

        if result.success:
            self.orchestrator_stats['successful_queries'] = self.orchestrator_stats.get('successful_queries', 0) + 1
        else:  # failed_queries 已经在 _handle_processing_error 或 _handle_processing_error_sync 中增加
            pass
            # self.orchestrator_stats['failed_queries'] = self.orchestrator_stats.get('failed_queries', 0) + 1
            # if result.error_info and isinstance(result.error_info, dict):
            #     error_type = result.error_info.get('error_type', 'unknown_in_stats_update')
            #     self.orchestrator_stats['error_types'][error_type] = \
            #         self.orchestrator_stats['error_types'].get(error_type, 0) + 1

        # 安全地计算平均值
        current_avg_time = float(self.orchestrator_stats.get('avg_processing_time', 0.0))
        self.orchestrator_stats['avg_processing_time'] = \
            (current_avg_time * (total_queries - 1) + float(result.total_processing_time)) / total_queries

        current_avg_confidence = float(self.orchestrator_stats.get('avg_confidence_score', 0.0))
        self.orchestrator_stats['avg_confidence_score'] = \
            (current_avg_confidence * (total_queries - 1) + float(result.confidence_score)) / total_queries

        for processor_name in result.processors_used:  # processors_used is List[str]
            if isinstance(processor_name, str):
                self.orchestrator_stats['processor_usage'][processor_name] = \
                    self.orchestrator_stats['processor_usage'].get(processor_name, 0) + 1

        # AI协作统计
        ai_collab_summary = result.ai_collaboration_summary
        if isinstance(ai_collab_summary, dict) and \
                ai_collab_summary.get('claude_used_in_process') and \
                ai_collab_summary.get('gpt_used_in_process'):
            self.orchestrator_stats['ai_collaboration_count'] = self.orchestrator_stats.get('ai_collaboration_count',
                                                                                            0) + 1

        logger.debug(f"Orchestrator stats updated for QueryID: {result.query_id}. Total queries: {total_queries}")

    def _log_ai_response_to_conversation(self, conversation_id_for_db: int,
                                         final_result: ProcessingResult, query_id: str):
        """辅助方法：将AI响应和可视化内容记录到对话历史中。"""
        if not self.conversation_manager:
            logger.warning(
                f"QueryID: {query_id} - ConversationManager not available. Skipping logging AI response to ConvID {conversation_id_for_db}.")
            return
        try:
            # 确保 conversation_id_for_db 是整数 (ConversationManager 的方法期望 int)
            if not isinstance(conversation_id_for_db, int):
                logger.error(
                    f"QueryID: {query_id} - Invalid conversation_id_for_db type: {type(conversation_id_for_db)}. Must be int for DB operations with ConversationManager. Skipping log.")
                return

            # 添加AI响应消息
            # add_message 返回新消息的 ID
            ai_message_id = self.conversation_manager.add_message(
                conversation_id=conversation_id_for_db,
                is_user=False,  # AI的响应
                content=final_result.response_text,
                ai_model_used=str(final_result.processing_metadata.get('ai_models_invoked', [])),  # 从元数据获取
                ai_strategy=final_result.processing_strategy.value,  # 使用枚举值
                processing_time=final_result.total_processing_time,
                confidence_score=final_result.confidence_score
            )
            logger.debug(
                f"QueryID: {query_id} - AI response (MessageID in DB: {ai_message_id}) logged to ConvID {conversation_id_for_db}.")

            # 添加可视化内容到 message_visuals 表
            if final_result.visualizations and ai_message_id > 0:  # 确保有消息ID
                for vis_idx, vis_data_dict in enumerate(final_result.visualizations):
                    if not isinstance(vis_data_dict, dict):
                        logger.warning(
                            f"QueryID: {query_id} - Invalid visual data for MsgID {ai_message_id}: {vis_data_dict}. Skipping.")
                        continue
                    try:
                        self.conversation_manager.add_visual(
                            message_id=ai_message_id,
                            visual_type=vis_data_dict.get('type', 'chart'),  # 'type' 在 visualization dict 中
                            visual_order=vis_idx,  # 简单的顺序
                            title=vis_data_dict.get('title', '生成的图表'),
                            # 'data_payload' 或 'data' 键可能包含图表的实际数据/配置
                            data=vis_data_dict.get('data_payload', vis_data_dict.get('data', {}))
                        )
                    except Exception as e_vis_add:
                        logger.error(
                            f"QueryID: {query_id} - Failed to add visual to MsgID {ai_message_id} for ConvID {conversation_id_for_db}: {e_vis_add}")
                logger.debug(
                    f"QueryID: {query_id} - {len(final_result.visualizations)} visuals logged to MessageID {ai_message_id} for ConvID {conversation_id_for_db}.")
            elif final_result.visualizations and ai_message_id <= 0:
                logger.warning(
                    f"QueryID: {query_id} - Got visuals but AI message ID ({ai_message_id}) is invalid. Visuals not logged for ConvID {conversation_id_for_db}.")


        except Exception as conv_err:
            # 捕获 ConversationManager 可能抛出的任何异常
            logger.error(
                f"QueryID: {query_id} - Error logging AI response/visuals to ConvID {conversation_id_for_db}: {conv_err}\n{traceback.format_exc()}")
    def _handle_processing_error_sync(self, session_id: str, query_id: str,
                                      user_query: str, error_msg: str,
                                      total_time_on_error: float,
                                      conversation_id: Optional[str] = None) -> ProcessingResult:
        """
        同步处理流程中的关键错误，生成错误结果对象。
        用于初始化失败等无法进入完整异步流程的场景。
        """
        # 注意：此方法是同步的，不应调用任何异步操作或依赖AI生成降级响应。

        current_stats = getattr(self, 'orchestrator_stats', self._default_stats())
        current_config = getattr(self, 'config', self._load_orchestrator_config())

        current_stats['failed_queries'] = current_stats.get('failed_queries', 0) + 1
        error_type = self._classify_error(error_msg)  # _classify_error 必须是同步的
        current_stats['error_types'][error_type] = \
            current_stats['error_types'].get(error_type, 0) + 1

        logger.error(f"QueryID: {query_id} - SYNC Handling CRITICAL processing error: {error_msg}, Type: {error_type}.")

        simple_fallback_text = f"抱歉，系统在处理您的查询 '{user_query[:30]}...' 时遇到一个关键错误。"

        debug_mode = current_config.get("DEBUG", False)

        if debug_mode:
            simple_fallback_text += f" 错误详情: {error_msg}"
        else:
            simple_fallback_text += " 技术团队已被通知，请稍后重试或联系支持。"

        error_processing_strategy = OrchestratorProcessingStrategy.ERROR_HANDLING

        error_result = ProcessingResult(
            session_id=session_id,
            query_id=query_id,
            success=False,
            response_text=simple_fallback_text,
            insights=[],
            key_metrics={},
            visualizations=[],
            error_info={
                'error_type': error_type,
                'message': error_msg,
                'original_query': user_query,
                'trace': traceback.format_exc() if debug_mode else "Error trace hidden in sync error handler."
            },
            processing_strategy=error_processing_strategy,
            processors_used=['SyncErrorHandling'],
            total_processing_time=total_time_on_error,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )

        # 对于初始化阶段的错误，完整的统计更新可能不适用或不准确
        # 此处仅更新了失败计数和错误类型。
        # 如果需要，可以有条件地调用 _update_orchestrator_stats，但要确保 total_queries 已正确计数。
        # self._update_orchestrator_stats(error_result) # 暂时注释以避免潜在问题

        return error_result
    async def _determine_processing_strategy(self, query_analysis: QueryAnalysisResult, preferences: Optional[
        Dict[str, Any]] = None) -> OrchestratorProcessingStrategy:
        """智能确定处理策略，基于规则和AI辅助。"""
        method_start_time = time.time()
        logger.debug(
            f"Determining processing strategy for query type: {query_analysis.query_type.value}, complexity: {query_analysis.complexity.value}")

        if preferences and 'processing_strategy' in preferences:
            try:
                preferred_strategy = OrchestratorProcessingStrategy(preferences['processing_strategy'])
                logger.info(f"Using user preferred strategy: {preferred_strategy.value}")
                return preferred_strategy
            except ValueError:
                logger.warning(
                    f"Invalid processing_strategy in preferences: {preferences['processing_strategy']}. Defaulting.")

        complexity_from_parser: QueryParserComplexity = query_analysis.complexity
        strategy_map = {
            QueryParserComplexity.SIMPLE: OrchestratorProcessingStrategy.DIRECT_RESPONSE,
            QueryParserComplexity.MEDIUM: OrchestratorProcessingStrategy.SINGLE_PROCESSOR,
            QueryParserComplexity.COMPLEX: OrchestratorProcessingStrategy.MULTI_PROCESSOR,
            QueryParserComplexity.EXPERT: OrchestratorProcessingStrategy.FULL_PIPELINE,
        }
        determined_strategy = strategy_map.get(complexity_from_parser, OrchestratorProcessingStrategy.SINGLE_PROCESSOR)

        # AI辅助决策 (中文提示词)
        if self.config.get('enable_smart_routing', True) and self.claude_client:
            try:
                ai_prompt_for_strategy = f"""
                作为智能金融分析系统的决策核心，请为以下用户查询的分析结果选择最合适的处理策略。
                用户查询的分析概要:
                - 复杂度评估: {query_analysis.complexity.value}
                - 查询意图类型: {query_analysis.query_type.value}
                - 涉及业务场景: {query_analysis.business_scenario.value if query_analysis.business_scenario else '未明确'}
                - 系统解析置信度: {query_analysis.confidence_score:.2f}
                - 预计执行步骤数: {len(query_analysis.execution_plan)}

                可选的处理策略包括:
                - "direct_response": 用于简单信息获取或无需复杂计算的查询。
                - "single_processor": 用于需要单一类型核心处理器（如仅当前数据、仅历史分析、或仅预测）的标准查询。
                - "multi_processor": 用于涉及多种分析维度，可能需要多个处理器协作的复杂查询。
                - "full_pipeline": 用于需要系统进行最全面、最深度分析的专家级查询，可能调动所有相关处理和分析模块。

                当前基于规则的初步建议策略是: {determined_strategy.value}
                请结合以上信息，特别是查询的复杂度和解析置信度，给出您的最终策略选择。
                如果解析置信度较低（例如低于0.7），请倾向于选择更简单、更稳健的策略。
                请仅返回策略的英文名小写字符串，例如："single_processor"。
                """
                # 假设ClaudeClient的analyze_complex_query方法第一个参数是prompt，第二个是上下文（可为空）
                ai_suggestion_result = await self.claude_client.analyze_complex_query(ai_prompt_for_strategy, {})
                if ai_suggestion_result.get('success'):
                    suggested_strategy_str = ai_suggestion_result.get('response', ai_suggestion_result.get('analysis',
                                                                                                           '')).strip().lower()
                    try:
                        ai_determined_strategy = OrchestratorProcessingStrategy(suggested_strategy_str)
                        logger.info(
                            f"AI suggested strategy: {ai_determined_strategy.value} (Rule-based was: {determined_strategy.value})")
                        determined_strategy = ai_determined_strategy  # AI 建议覆盖规则
                    except ValueError:
                        logger.warning(f"AI返回了无效的策略字符串: '{suggested_strategy_str}'")
                else:
                    logger.warning(f"AI策略决策失败: {ai_suggestion_result.get('error', 'Unknown AI error')}")
            except Exception as e_ai_strat:
                logger.error(f"AI策略决策过程中发生异常: {e_ai_strat}")

        if query_analysis.confidence_score < self.config.get('min_confidence_threshold_parsing', 0.7):
            logger.warning(
                f"解析置信度低 ({query_analysis.confidence_score:.2f}), 最终策略从 {determined_strategy.value} 降级。")
            if determined_strategy == OrchestratorProcessingStrategy.FULL_PIPELINE:
                determined_strategy = OrchestratorProcessingStrategy.MULTI_PROCESSOR
            elif determined_strategy == OrchestratorProcessingStrategy.MULTI_PROCESSOR:
                determined_strategy = OrchestratorProcessingStrategy.SINGLE_PROCESSOR
            elif determined_strategy == OrchestratorProcessingStrategy.SINGLE_PROCESSOR:
                determined_strategy = OrchestratorProcessingStrategy.DIRECT_RESPONSE

        logger.debug(
            f"Strategy determination took {time.time() - method_start_time:.3f}s. Final strategy: {determined_strategy.value}")
        return determined_strategy

    async def _orchestrate_data_acquisition(self, query_analysis: QueryAnalysisResult,
                                            strategy: OrchestratorProcessingStrategy) -> FetcherExecutionResult:
        """编排数据获取流程，调用 DataRequirementsAnalyzer 和 SmartDataFetcher。"""
        method_start_time = time.time()
        logger.info(f"DataAcquisition: Analyzing requirements for query type '{query_analysis.query_type.value}'")
        if not self.data_requirements_analyzer: raise SystemError("DataRequirementsAnalyzer not initialized.")
        if not self.data_fetcher: raise SystemError("SmartDataFetcher not initialized.")

        data_plan: DataAcquisitionPlan = await self.data_requirements_analyzer.analyze_data_requirements(query_analysis)

        if not data_plan or not data_plan.api_call_plans:
            logger.warning(
                "DataRequirementsAnalyzer did not produce API call plans. This might be ok for very simple queries or indicate an issue.")
            # 即使没有计划，也调用fetcher，它应该能处理空计划
            # 或者如果绝对需要数据，尝试获取基础数据
            if strategy != OrchestratorProcessingStrategy.DIRECT_RESPONSE or query_analysis.query_type != QueryParserQueryType.GENERAL_KNOWLEDGE:
                logger.info(
                    "No API calls in plan, attempting fallback to fetch basic system data for non-direct/general queries.")
                # (Fallback logic from previous full version can be inserted here if needed)
                # For now, let SmartDataFetcher handle an empty plan if that's its design.
                # If not, we must ensure data_plan always has something or handle this case explicitly.
                # Assume SmartDataFetcher returns an ExecutionResult indicating no data if plan is empty.
                pass  # Allow execution of potentially empty plan

        logger.info(
            f"DataAcquisition: Executing data plan '{data_plan.plan_id if data_plan else 'N/A'}' with {len(data_plan.api_call_plans) if data_plan else 0} API calls.")
        fetch_result: FetcherExecutionResult = await self.data_fetcher.execute_data_acquisition_plan(
            data_plan)  # SmartDataFetcher handles empty plan

        logger.debug(
            f"Data acquisition took {time.time() - method_start_time:.3f}s. Fetch status: {fetch_result.execution_status.value if fetch_result.execution_status else 'N/A'}")
        return fetch_result

    async def _orchestrate_business_processing(self, query_analysis: QueryAnalysisResult,
                                               data_acquisition_result: FetcherExecutionResult,
                                               strategy: OrchestratorProcessingStrategy) -> Dict[str, Any]:
        # ++++++++++++++ 添加此行 ++++++++++++++
        biz_proc_start_wall_time = time.time()
        # +++++++++++++++++++++++++++++++++++++

        logger.info(
            f"BusinessProcessing: Strategy '{strategy.value}', QueryType from Parser '{query_analysis.query_type.value}'")
        if not all([self.current_data_processor, self.historical_analysis_processor, self.prediction_processor,
                    self.financial_data_analyzer]):
            raise SystemError("One or more core processors/analyzers are not initialized for business processing.")

        processor_context = {'data_acquisition_result': data_acquisition_result, 'query_analysis': query_analysis}
        original_user_query = query_analysis.original_query

        processor_result_payload: Any = None
        processor_used_names: List[str] = []

        sum_of_reported_proc_times: float = 0.0
        confidences_from_procs: List[float] = []
        # biz_proc_metadata 已移到 return 语句中动态构建，以包含 ai_processing_time

        # --- SINGLE_PROCESSOR and DIRECT_RESPONSE ---
        if strategy == OrchestratorProcessingStrategy.DIRECT_RESPONSE:
            # ... (处理逻辑) ...
            res_obj = await self.current_data_processor.process_current_data_query(original_user_query)
            processor_result_payload = res_obj
            processor_used_names.append(self.current_data_processor.__class__.__name__)
            sum_of_reported_proc_times = getattr(res_obj, 'processing_time', 0.1)
            confidences_from_procs.append(getattr(res_obj, 'response_confidence', 0.7))

        elif strategy == OrchestratorProcessingStrategy.SINGLE_PROCESSOR:
            # ... (处理逻辑) ...
            qt_parser_enum: QueryParserQueryType = query_analysis.query_type
            selected_processor: Any = None

            if qt_parser_enum in [QueryParserQueryType.DATA_RETRIEVAL, QueryParserQueryType.CALCULATION,
                                  QueryParserQueryType.GENERAL_KNOWLEDGE]:  # 假设GENERAL_KNOWLEDGE已添加
                selected_processor = self.current_data_processor
                res_obj = await selected_processor.process_current_data_query(original_user_query)
            elif qt_parser_enum in [QueryParserQueryType.TREND_ANALYSIS, QueryParserQueryType.COMPARISON]:
                selected_processor = self.historical_analysis_processor
                res_obj = await selected_processor.process_historical_analysis_query(original_user_query,
                                                                                     processor_context)
            elif qt_parser_enum in [QueryParserQueryType.PREDICTION, QueryParserQueryType.SCENARIO_SIMULATION,
                                    QueryParserQueryType.RISK_ASSESSMENT]:
                selected_processor = self.prediction_processor
                res_obj = await selected_processor.process_prediction_query(original_user_query, processor_context)
            else:
                selected_processor = self.current_data_processor  # Default
                res_obj = await selected_processor.process_current_data_query(original_user_query)

            processor_result_payload = res_obj
            processor_used_names.append(selected_processor.__class__.__name__)
            sum_of_reported_proc_times += getattr(res_obj, 'processing_time', 0.2)
            confidences_from_procs.append(
                getattr(res_obj, 'confidence_score',
                        getattr(res_obj, 'response_confidence',
                                getattr(res_obj, 'analysis_confidence',
                                        getattr(res_obj, 'prediction_confidence', 0.6)))))

        elif strategy in [OrchestratorProcessingStrategy.MULTI_PROCESSOR, OrchestratorProcessingStrategy.FULL_PIPELINE]:
            # ... (多处理器/全流水线处理逻辑，与您之前提供的完整版本一致) ...
            multi_results_payload: Dict[str, Any] = {}
            tasks_to_run_coroutines_map: Dict[str, asyncio.Task] = {}  # type: ignore

            tasks_to_run_coroutines_map['current_data_analysis'] = asyncio.create_task(
                self.current_data_processor.process_current_data_query(original_user_query)
            )

            if query_analysis.query_type in [QueryParserQueryType.TREND_ANALYSIS, QueryParserQueryType.COMPARISON,
                                             QueryParserQueryType.PREDICTION, QueryParserQueryType.SCENARIO_SIMULATION,
                                             QueryParserQueryType.RISK_ASSESSMENT] or \
                    any(step.step_type == "trend_analysis" for step in query_analysis.execution_plan):
                tasks_to_run_coroutines_map['historical_analysis'] = asyncio.create_task(
                    self.historical_analysis_processor.process_historical_analysis_query(original_user_query,
                                                                                         processor_context)
                )

            if query_analysis.query_type in [QueryParserQueryType.PREDICTION, QueryParserQueryType.SCENARIO_SIMULATION,
                                             QueryParserQueryType.RISK_ASSESSMENT] or \
                    any(step.step_type == "prediction" for step in query_analysis.execution_plan):
                tasks_to_run_coroutines_map['prediction_analysis'] = asyncio.create_task(
                    self.prediction_processor.process_prediction_query(original_user_query, processor_context)
                )

            if strategy == OrchestratorProcessingStrategy.FULL_PIPELINE and self.financial_data_analyzer:
                scope_for_fda = query_analysis.business_scenario.value if query_analysis.business_scenario else 'financial_overview'
                time_range_days_fda = 30
                if query_analysis.time_requirements and query_analysis.time_requirements.get('time_range_days'):
                    time_range_days_fda = int(query_analysis.time_requirements['time_range_days'])

                tasks_to_run_coroutines_map['financial_deep_dive'] = asyncio.create_task(
                    self.financial_data_analyzer.analyze_business_performance(scope=scope_for_fda,
                                                                              time_range=time_range_days_fda
                                                                             )
                )

            logger.info(
                f"Executing {len(tasks_to_run_coroutines_map)} tasks for strategy {strategy.value}: {list(tasks_to_run_coroutines_map.keys())}")

            for name, task_coro in tasks_to_run_coroutines_map.items():
                try:
                    res_obj = await task_coro
                    multi_results_payload[name] = res_obj
                    processor_used_names.append(getattr(res_obj, '__class__', {}).__name__ or name)
                    sum_of_reported_proc_times += getattr(res_obj, 'processing_time', 0.1)
                    confidences_from_procs.append(
                        getattr(res_obj, 'confidence_score', getattr(res_obj, 'response_confidence', 0.6))
                    )
                except Exception as task_e:
                    multi_results_payload[name] = {"error": str(task_e), "details": traceback.format_exc()}
                    logger.error(f"Error processing task '{name}' in {strategy.value}: {task_e}")
                    confidences_from_procs.append(0.1)
            processor_result_payload = {"multi_results": multi_results_payload}
        else:
            logger.error(f"Unsupported processing strategy: {strategy.value}")
            raise ValueError(f"Unsupported processing strategy: {strategy.value}")

        # --- 在所有业务处理逻辑完成后计算实际耗时 ---
        combined_processing_time = time.time() - biz_proc_start_wall_time  # 现在 biz_proc_start_wall_time 已定义
        # --- ---

        overall_biz_confidence = sum(confidences_from_procs) / len(
            confidences_from_procs) if confidences_from_procs else 0.5

        # 累加各处理器报告的AI时间
        biz_proc_ai_time = 0.0
        if isinstance(processor_result_payload, dict) and 'multi_results' in processor_result_payload:
            for res_obj in processor_result_payload['multi_results'].values():
                if hasattr(res_obj, 'metadata') and isinstance(res_obj.metadata, dict):
                    biz_proc_ai_time += res_obj.metadata.get('ai_processing_time', 0.0)
        elif hasattr(processor_result_payload, 'metadata') and isinstance(processor_result_payload.metadata,
                                                                          dict):  # 单个对象
            biz_proc_ai_time = processor_result_payload.metadata.get('ai_processing_time', 0.0)

        return {
            'processing_type': strategy.value,
            'processor_used': ", ".join(list(set(processor_used_names))),
            'primary_result': processor_result_payload,
            'confidence_score': overall_biz_confidence,
            'processing_time': combined_processing_time,
            'metadata': {
                'data_input_source': 'from_smart_data_fetcher',
                'sum_of_individual_processor_reported_times': sum_of_reported_proc_times,
                'ai_processing_time': biz_proc_ai_time  # 记录业务处理阶段的AI耗时
            }
        }

    async def _orchestrate_insight_generation(self, business_result_data: Dict[str, Any],
                                              query_analysis: QueryAnalysisResult,
                                              data_acquisition_result: FetcherExecutionResult) -> List[BusinessInsight]:
        logger.debug("Orchestrator: Calling InsightGenerator...")
        if not self.insight_generator: raise SystemError("InsightGenerator not initialized.")

        analysis_objects_for_insights = []
        primary_res_payload = business_result_data.get('primary_result')  # This is object or dict of objects

        if isinstance(primary_res_payload, dict) and 'multi_results' in primary_res_payload:
            for res_key, res_obj in primary_res_payload['multi_results'].items():
                if res_obj and not (isinstance(res_obj, dict) and 'error' in res_obj):
                    analysis_objects_for_insights.append(res_obj)
        elif primary_res_payload and not (isinstance(primary_res_payload, dict) and 'error' in primary_res_payload):
            analysis_objects_for_insights.append(primary_res_payload)

        if not analysis_objects_for_insights:
            logger.warning("No valid business processing results to generate insights from.")
            return []

        # Prepare context for insight generator
        conv_id_for_db = None
        # conversation_id is from process_intelligent_query's parameters
        # If process_intelligent_query received a valid string conversation_id that was converted to int
        # that int conversation_id_for_db can be used here.
        # For simplicity, let's assume it's available if needed by get_context
        # This context logic needs to be robust based on how conv_id is managed.
        # For now, passing empty context if conv_id not available or invalid.
        parent_conversation_id_str = query_analysis.processing_metadata.get(
            'conversation_id_from_query_parsing') if query_analysis.processing_metadata else None
        if parent_conversation_id_str:
            try:
                conv_id_for_db = int(parent_conversation_id_str)
            except:
                pass

        user_context_for_insights = self.conversation_manager.get_context(
            conv_id_for_db) if conv_id_for_db and self.conversation_manager else {}

        insights, metadata = await self.insight_generator.generate_comprehensive_insights(
            analysis_results=analysis_objects_for_insights,
            user_context=user_context_for_insights,
            focus_areas=self._determine_focus_areas(query_analysis)  # Implemented
        )

        # Store AI time from insight generation
        ai_time_insights = metadata.get('ai_time', 0.0) if isinstance(metadata, dict) else 0.0
        business_result_data.setdefault('metadata', {}).update({'insight_gen_ai_time': ai_time_insights})
        business_result_data['insight_generation_metadata'] = metadata
        return insights

    async def _generate_intelligent_response(self, user_query: str, query_analysis: QueryAnalysisResult,
                                             business_result_data: Dict[str, Any],
                                             insights_list: List[BusinessInsight]) -> str:
        logger.debug("Orchestrator: Generating intelligent response text using CHINESE prompts.")
        if not self.claude_client and not self.gpt_client:
            logger.warning("AI客户端 (Claude/GPT) 未配置，使用基础模板生成响应。")
            return self._generate_basic_response(business_result_data, insights_list)  # Implemented

        summarized_analysis = self._summarize_business_result_for_ai(business_result_data)  # Implemented
        summarized_insights = []
        for insight in insights_list[:3]:
            title = getattr(insight, 'title', "重要洞察")
            summary = getattr(insight, 'summary', "...")
            summarized_insights.append(f"{title}: {summary}")

        # --- CHINESE PROMPT ---
        prompt = f"""
        作为一位资深的AI金融业务顾问，请根据以下信息，为用户生成一份全面、专业且易于理解的中文回答。

        用户的原始查询:
        "{user_query}"

        系统对查询的分析概要:
        - 用户意图核心: {query_analysis.query_type.value if query_analysis.query_type else "未知"}
        - 涉及业务场景: {query_analysis.business_scenario.value if query_analysis.business_scenario else "通用"}
        - 分析复杂度评估: {query_analysis.complexity.value if query_analysis.complexity else "中等"}

        核心分析结果精炼 (JSON格式):
        {json.dumps(summarized_analysis, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}

        关键业务洞察提要:
        {chr(10).join([f"  - {si}" for si in summarized_insights]) if summarized_insights else "  - 暂未生成特别的业务洞察。"}

        请严格按照以下结构和要求生成最终答复（使用Markdown格式）：

        ### **回答核心问题**
        [清晰、直接地回答用户查询的核心问题。]

        ### **关键数据与发现**
        [列出并解释1-3个最重要的图表或分析发现。例如：**总余额趋势：过去30天内增长了15%，达到¥XXX。**]

        ### **业务解读与洞察**
        [（如果洞察列表不为空）简要解释这些发现对业务的潜在影响或意义，自然地融入洞察的结论。例如：*洞察显示，近期用户活跃度提升可能与XX活动有关，建议持续关注并优化相关策略。*]

        ### **建议行动**
        [（如果洞察包含明确建议）提供1-2条最关键、最具可操作性的建议。]

        ### **总结与展望** (可选)
        [对整体情况进行简短总结或对未来趋势做初步展望。]

        **请注意：**
        - 语言风格需专业、客观、数据驱动，同时确保业务人员能够轻松理解。
        - 避免使用过于技术性的术语，必要时进行解释。
        - 如果分析结果中存在不确定性、风险或数据局限性，请在回答中明确、审慎地指出。
        - 如果数据不足或分析置信度较低，请坦诚说明，并可以礼貌地建议用户提供更多信息或调整查询范围。
        - 总回答长度建议控制在300-800字左右，根据信息量灵活调整。
        """

        ai_client_to_use = self.claude_client if self.claude_client else self.gpt_client  # 优先Claude
        if not ai_client_to_use: return self._generate_basic_response(business_result_data, insights_list)

        try:
            ai_response_data = None
            logger.info(f"使用 {ai_client_to_use.__class__.__name__} 生成最终响应文本。")
            # 假设AI客户端有通用的文本生成方法，或者需要根据类型选择
            if isinstance(ai_client_to_use, ClaudeClient) and hasattr(ai_client_to_use,
                                                                      'generate_text'):  # 假设Claude有此方法
                ai_response_data = await ai_client_to_use.generate_text(prompt, max_tokens=1500)  # Claude的典型方法
            elif isinstance(ai_client_to_use, OpenAIClient) and hasattr(ai_client_to_use, 'generate_completion'):
                ai_response_data = await ai_client_to_use.generate_completion(prompt, max_tokens=1500)  # OpenAI的典型方法
            else:  # Fallback for unknown client method signature
                logger.warning(f"AI客户端 {ai_client_to_use.__class__.__name__} 未找到标准的文本生成方法。")
                return self._generate_basic_response(business_result_data, insights_list)

            # 解析AI响应 (不同客户端的响应结构可能不同)
            if ai_response_data and ai_response_data.get('success', False):  # 假设响应中有 'success' 标志
                response_content = ai_response_data.get('text', ai_response_data.get('response', ai_response_data.get(
                    'completion')))  # 尝试不同可能的键
                if response_content and isinstance(response_content, str):
                    return response_content.strip()
                else:
                    logger.warning(f"AI响应内容为空或格式不正确: {response_content}")
                    return self._generate_basic_response(business_result_data, insights_list)
            else:
                logger.warning(
                    f"AI生成响应文本失败: {ai_response_data.get('error', '未知AI错误') if ai_response_data else 'AI客户端返回空'}")
                return self._generate_basic_response(business_result_data, insights_list)
        except Exception as e:
            logger.error(f"生成智能响应时发生异常: {str(e)}\n{traceback.format_exc()}")
            return self._generate_basic_response(business_result_data, insights_list)

    async def _generate_visualizations(self, business_result_data: Dict[str, Any],
                                       query_analysis: QueryAnalysisResult,
                                       insights_list: List[BusinessInsight]) -> List[Dict[str, Any]]:
        # (与上一版本完整实现一致，确保从真实业务结果中提取数据并调用 self.chart_generator)
        logger.debug("Orchestrator: Generating visualizations...")
        visualizations: List[Dict[str, Any]] = []
        if not self.chart_generator:
            logger.warning("ChartGenerator未初始化，跳过可视化生成。")
            return visualizations

        try:
            raw_numeric_metrics = self._extract_raw_numeric_metrics(business_result_data)

            if raw_numeric_metrics and len(raw_numeric_metrics) >= 2 and len(raw_numeric_metrics) <= 10:
                chart_labels = list(raw_numeric_metrics.keys())
                chart_values = list(raw_numeric_metrics.values())
                try:
                    bar_chart_result = self.chart_generator.generate_bar_chart(
                        data={'category': chart_labels, 'value': chart_values},
                        title="关键指标快照", x_column='category', y_columns=['value']
                    )
                    if bar_chart_result and not bar_chart_result.get("error"):
                        visualizations.append({
                            "type": VisChartType.BAR.value,
                            "title": bar_chart_result.get('title', "关键指标快照"),
                            "data_payload": bar_chart_result.get('data'),
                            "image_data_b64": bar_chart_result.get('image_data', {}).get('base64')
                        })
                except Exception as e_bar:
                    logger.error(f"Error generating bar chart for key metrics: {e_bar}")

            primary_res_payload = business_result_data.get('primary_result', {})
            # historical_analysis_res_obj should be the direct object from the processor or its dict representation
            historical_analysis_res_obj = None
            if isinstance(primary_res_payload, dict) and 'multi_results' in primary_res_payload:
                historical_analysis_res_obj = primary_res_payload['multi_results'].get('historical_analysis')
            elif isinstance(primary_res_payload, HistoricalAnalysisResponse):
                historical_analysis_res_obj = primary_res_payload
            elif isinstance(primary_res_payload, dict) and primary_res_payload.get(
                    'analysis_type') == QueryParserQueryType.TREND_ANALYSIS.value:
                historical_analysis_res_obj = primary_res_payload

            # Convert to dict if it's an object and has to_dict or __dict__
            historical_analysis_dict = None
            if historical_analysis_res_obj:
                if hasattr(historical_analysis_res_obj, 'to_dict') and callable(historical_analysis_res_obj.to_dict):
                    historical_analysis_dict = historical_analysis_res_obj.to_dict()
                elif hasattr(historical_analysis_res_obj, '__dict__'):
                    historical_analysis_dict = historical_analysis_res_obj.__dict__
                elif isinstance(historical_analysis_res_obj, dict):
                    historical_analysis_dict = historical_analysis_res_obj

            if historical_analysis_dict and isinstance(historical_analysis_dict.get('trends'), list):
                for trend_detail_item in historical_analysis_dict[
                    'trends']:  # trend_detail_item is FinancialTrendAnalysis or its dict
                    metric_name = "趋势"
                    data_points = []
                    if isinstance(trend_detail_item, FinancialTrendAnalysis):  # If it's the object
                        metric_name = getattr(trend_detail_item, 'metric_name', '趋势')
                        data_points = getattr(trend_detail_item, 'chart_data', [])
                    elif isinstance(trend_detail_item, dict):  # If it's a dict
                        metric_name = trend_detail_item.get('metric_name', trend_detail_item.get('metric', '趋势'))
                        data_points = trend_detail_item.get('chart_data',
                                                            trend_detail_item.get('data_points_for_chart', []))

                    if data_points and isinstance(data_points, list) and len(data_points) > 1:
                        if not (all(isinstance(dp, dict) and 'date' in dp and 'value' in dp for dp in data_points)):
                            logger.warning(
                                f"Trend data for {metric_name} is not in expected format for charting. Skipping.")
                            continue
                        try:
                            line_chart_result = self.chart_generator.generate_line_chart(
                                data={'dates': [dp['date'] for dp in data_points],
                                      metric_name: [dp['value'] for dp in data_points]},
                                title=f"{metric_name} 趋势分析", x_column='dates', y_columns=[metric_name]
                            )
                            if line_chart_result and not line_chart_result.get("error"):
                                visualizations.append({
                                    "type": VisChartType.LINE.value,
                                    "title": line_chart_result.get('title', f"{metric_name} 趋势分析"),
                                    "data_payload": line_chart_result.get('data'),
                                    "image_data_b64": line_chart_result.get('image_data', {}).get('base64')
                                })
                        except Exception as e_line:
                            logger.error(f"Error generating line chart for {metric_name}: {e_line}")

            logger.info(f"生成了 {len(visualizations)} 个可视化数据对象。")
        except Exception as e:
            logger.error(f"生成可视化时发生错误: {str(e)}\n{traceback.format_exc()}")

        return visualizations

    # --- 以下是所有之前讨论的、需要确保存在的辅助方法 ---
    # _extract_raw_numeric_metrics, _summarize_business_result_for_ai,
    # _determine_focus_areas, _get_processors_used, _get_ai_collaboration_summary,
    # _calculate_overall_confidence, _calculate_response_completeness,
    # _get_ai_models_invoked_summary, _log_ai_response_to_conversation,
    # _fallback_query_parsing, _generate_basic_response

    def _calculate_response_completeness(self, business_result_data: Dict[str, Any],
                                         insights_list: List[BusinessInsight]) -> float:
        """
        评估响应的完整性。
        考虑是否覆盖了关键分析结果和洞察。
        返回一个 0.0 到 1.0 之间的分数。
        """
        score = 0.0
        max_score = 1.0  # 总分值

        # 1. 是否有主要的业务处理结果 (占0.6分)
        if business_result_data and business_result_data.get('primary_result'):
            # 进一步检查 primary_result 是否真的有内容，而不是空壳或错误
            primary_res_content = business_result_data.get('primary_result')
            is_error_dict = isinstance(primary_res_content, dict) and 'error' in primary_res_content
            if primary_res_content and not is_error_dict:
                score += 0.6
            elif is_error_dict:  # 如果是错误，完整性会低一些
                score += 0.1
        else:  # 如果连 primary_result 都没有，则得分很低
            score += 0.05

        # 2. 是否生成了洞察 (占0.3分)
        if insights_list and isinstance(insights_list, list) and len(insights_list) > 0:
            score += 0.3

        # 3. 是否提取出关键指标 (占0.1分)
        # _extract_key_metrics 已经处理了各种情况并返回格式化指标
        key_metrics = self._extract_key_metrics(business_result_data)
        if key_metrics and isinstance(key_metrics, dict) and len(key_metrics) > 0:
            score += 0.1

        # 确保分数在0到1之间
        final_completeness = round(min(score, max_score), 3)
        logger.debug(f"Calculated response completeness: {final_completeness}")
        return final_completeness

    def _get_ai_models_invoked_summary(self, query_analysis: Optional[QueryAnalysisResult],
                                       business_result_data: Dict[str, Any],
                                       insights_list: List[BusinessInsight]) -> List[str]:
        """
        根据分析过程总结实际调用的AI模型 (例如 ["Claude", "GPT"])。
        更准确的统计需要各组件在其返回的元数据中明确报告AI使用情况。
        """
        models_invoked = set()

        # 1. 从AI协作摘要中获取（如果该摘要已准确记录）
        ai_collab_summary = self._get_ai_collaboration_summary(query_analysis, business_result_data, insights_list)
        if ai_collab_summary.get('claude_used_in_process'):
            models_invoked.add("Claude")
        if ai_collab_summary.get('gpt_used_in_process'):
            models_invoked.add("GPT")  # 或更具体的模型名如 "GPT-4o"

        # 2. (可选) 进一步检查各阶段的元数据，如果它们包含更细致的模型使用信息
        # 例如，query_analysis.processing_metadata.get('parser_ai_model')
        # business_result_data.get('metadata', {}).get('processor_ai_model')
        # insights_list[0].metadata.get('generator_ai_model') 等

        if not models_invoked:
            # 如果上述都未明确，但AI客户端存在，做一个基本假设
            if self.claude_client: models_invoked.add("Claude (assumed)")
            if self.gpt_client: models_invoked.add("GPT (assumed)")

        return list(models_invoked) if models_invoked else ["None_Specifically_Reported"]

    async def _handle_processing_error(self, session_id: str, query_id: str,
                                       user_query: str, error_msg: str,
                                       total_time_on_error: float,
                                       conversation_id: Optional[str] = None) -> ProcessingResult:
        """
        统一处理流程中的错误，生成错误结果对象。此为异步版本。
        """
        # total_queries 已经在 process_intelligent_query 开始时增加
        # 这里主要更新失败计数和错误类型
        error_type = self._classify_error(error_msg)  # _classify_error 是同步的

        logger.error(
            f"QueryID: {query_id} - ASYNC Handling processing error: {error_msg}, Type: {error_type}. Full Traceback (if DEBUG):\n{traceback.format_exc() if self.config.get('DEBUG') else 'Traceback hidden in production.'}")

        fallback_text = f"抱歉，在处理您的查询 '{user_query[:30]}...' 时系统遇到了一个预期之外的问题。"
        if self.config.get('fallback_response_enabled', True):
            try:
                # _generate_fallback_response 是异步的，因为它可能调用AI
                fallback_text = await self._generate_fallback_response(user_query, error_msg)
            except Exception as fallback_e:
                logger.error(f"QueryID: {query_id} - Generating fallback response itself also failed: {fallback_e}")
                # 如果AI生成降级响应也失败，使用更简单、硬编码的文本
                fallback_text = f"处理请求时发生错误。错误详情: {error_msg if self.config.get('DEBUG') else '内部系统错误，请联系技术支持。'}"

        error_processing_result = ProcessingResult(
            session_id=session_id,
            query_id=query_id,
            success=False,
            response_text=fallback_text,
            insights=[],  # 错误情况下，洞察等为空
            key_metrics={},
            visualizations=[],
            error_info={
                'error_type': error_type,
                'message': error_msg,
                'original_query': user_query,
                'trace': traceback.format_exc() if self.config.get("DEBUG") else "Error details hidden for security."
            },
            processing_strategy=OrchestratorProcessingStrategy.ERROR_HANDLING,
            processors_used=['AsyncErrorHandling'],  # 标记为异步错误处理器
            total_processing_time=total_time_on_error,
            conversation_id=conversation_id,  # 保持传入的 str ID
            timestamp=datetime.now().isoformat()
        )

        # 更新编排器统计（包括失败计数和错误类型）
        self._update_orchestrator_stats(error_processing_result)

        # 尝试记录错误响应到对话历史
        if conversation_id and self.conversation_manager:
            conv_id_for_db_err: Optional[int] = None
            try:
                conv_id_for_db_err = int(conversation_id)
            except ValueError:
                logger.warning(
                    f"QueryID: {query_id} - Cannot log error to conversation due to invalid conversation_id format: {conversation_id}")

            if conv_id_for_db_err is not None:
                self._log_ai_response_to_conversation(conv_id_for_db_err, error_processing_result, query_id)

        return error_processing_result

        # 在 IntelligentQAOrchestrator 类中修改此方法：
    async def _fallback_query_parsing(self, user_query: str) -> QueryAnalysisResult:
        """
        在AI查询解析失败时，提供一个基于规则的、基础的 QueryAnalysisResult。
        """
        logger.warning(f"Executing fallback query parsing for: '{user_query[:50]}...'")

        query_lower = user_query.lower()
        # 使用 query_parser.py 中定义的 QueryParserQueryType 和 QueryParserBusinessScenario
        qt = QueryParserQueryType.GENERAL_KNOWLEDGE if hasattr(QueryParserQueryType,
                                                               'GENERAL_KNOWLEDGE') else QueryParserQueryType.DEFINITION_EXPLANATION  # 假设有一个通用类型或解释类型
        bs = QueryParserBusinessScenario.UNKNOWN  # 默认使用 UNKNOWN
        qc = QueryParserComplexity.SIMPLE

        if any(kw in query_lower for kw in ["余额", "balance", "多少钱", "资金现状"]):
            qt = QueryParserQueryType.DATA_RETRIEVAL
            bs = QueryParserBusinessScenario.FINANCIAL_OVERVIEW  # 使用已定义的 FINANCIAL_OVERVIEW
            qc = QueryParserComplexity.SIMPLE
        elif any(kw in query_lower for kw in ["趋势", "历史", "trend", "history", "过去", "对比"]):
            qt = QueryParserQueryType.TREND_ANALYSIS
            bs = QueryParserBusinessScenario.HISTORICAL_PERFORMANCE  # 使用已定义的 HISTORICAL_PERFORMANCE
            qc = QueryParserComplexity.MEDIUM
        elif any(kw in query_lower for kw in
                 ["预测", "预计", "未来", "scenario", "如果", "what if"]):  # scenario 英文小写
            qt = QueryParserQueryType.PREDICTION
            bs = QueryParserBusinessScenario.FUTURE_PROJECTION  # 使用已定义的 FUTURE_PROJECTION
            qc = QueryParserComplexity.COMPLEX
        elif any(kw in query_lower for kw in ["用户", "会员", "user", "member"]):
            bs = QueryParserBusinessScenario.USER_ANALYSIS
        elif any(kw in query_lower for kw in ["产品", "product"]):
            bs = QueryParserBusinessScenario.PRODUCT_ANALYSIS

        return QueryAnalysisResult(
            original_query=user_query,
            complexity=qc,
            query_type=qt,
            business_scenario=bs,  # 使用修正后的 bs
            confidence_score=0.3,
            time_requirements={'parsed_by': 'fallback_rule', 'default_period': 'current'},
            date_parse_result=None,
            data_requirements={'apis_guessed': ['/api/sta/system']},
            required_apis=['/api/sta/system'],
            business_parameters={},
            calculation_requirements={},
            execution_plan=[
                QueryParserExecutionStep(  # 使用从 query_parser 导入的 QueryParserExecutionStep
                    step_id="fallback_fetch_system_data",
                    step_type="data_retrieval",
                    description="Fallback: 获取基础系统概览数据",
                    required_data=["system_overview"],
                    processing_method="direct_api_call_via_smart_fetcher",
                    dependencies=[],
                    estimated_time=1.5,
                    ai_model_preference="any"
                )
            ],
            processing_strategy="direct_response",
            ai_collaboration_plan={'strategy_type': 'none', 'reason': 'fallback_parsing_used'},
            analysis_timestamp=datetime.now().isoformat(),
            estimated_total_time=2.0,
            processing_metadata={"parser_status": "fallback_rule_based_incomplete"}
        )

    def _classify_error(self, error_msg: str) -> str:
        """
        根据错误消息文本分类错误类型。
        这是一个同步方法。
        """
        s_err = str(error_msg).lower()  # 确保是字符串并转为小写以便匹配

        if "timeout" in s_err: return "timeout_error"
        if "api key" in s_err or "authentication" in s_err or "unauthorized" in s_err: return "authentication_error"
        if "rate limit" in s_err or "quota exceeded" in s_err or "too many requests" in s_err: return "rate_limit_error"
        if "connection" in s_err or "network" in s_err or "dns" in s_err or "host not found" in s_err: return "network_error"
        if "database" in s_err or "sql" in s_err or "db query" in s_err: return "database_error"
        if "not initialized" in s_err or "not configured" in s_err: return "initialization_error"
        if any(ai_kw in s_err for ai_kw in
               ["claude", "openai", "gpt", "anthropic", "gemini", "ai model error"]): return "ai_service_error"
        if "attributeerror" in s_err: return "attribute_error_integration_issue"  # 通常是代码集成问题
        if "keyerror" in s_err: return "key_error_data_structure_issue"  # 访问了不存在的字典键
        if "indexerror" in s_err: return "index_error_data_structure_issue"  # 列表索引越界
        if "valueerror" in s_err or ("typeerror" in s_err and (
                "parameter" in s_err or "argument" in s_err)): return "validation_or_type_error"
        if "file not found" in s_err: return "file_not_found_error"
        if "permission denied" in s_err: return "permission_error"
        if "memory" in s_err and "out of" in s_err: return "out_of_memory_error"

        # 更通用的错误类型
        if "parsing failed" in s_err: return "parsing_error"
        if "data acquisition failed" in s_err: return "data_acquisition_error"
        if "processing failed" in s_err: return "business_processing_error"
        if "insight generation failed" in s_err: return "insight_generation_error"

        return "unknown_processing_error"  # 默认的未知错误
    def _extract_raw_numeric_metrics(self, business_result_data: Dict[str, Any]) -> Dict[str, float]:
        raw_metrics: Dict[str, float] = {}
        if not business_result_data or not isinstance(business_result_data, dict): return raw_metrics

        primary_res_content = business_result_data.get('primary_result')

        def _extract_from_single_res_raw(res_content: Any, prefix=""):
            if not res_content: return
            data_to_search = {}
            # Handle both dataclass objects and dictionaries
            if hasattr(res_content, 'to_dict') and callable(res_content.to_dict):
                data_to_search = res_content.to_dict()
            elif hasattr(res_content, '__dict__'):
                data_to_search = res_content.__dict__
            elif isinstance(res_content, dict):
                data_to_search = res_content
            else:
                return

            for metric_key_container in ['key_metrics', 'metrics', 'main_prediction', 'predictions']:
                if metric_key_container in data_to_search and isinstance(data_to_search[metric_key_container], dict):
                    for k, v_original in data_to_search[metric_key_container].items():
                        try:
                            v_str = str(v_original).replace('%', '').replace('¥', '').replace(',', '')
                            numeric_v = float(v_str)
                            metric_name_to_store = f"{prefix}{k}".strip('_')
                            if metric_name_to_store in raw_metrics and raw_metrics[metric_name_to_store] != numeric_v:
                                metric_name_to_store = f"{prefix}{metric_key_container}_{k}".strip('_')
                            raw_metrics[metric_name_to_store] = numeric_v
                        except (ValueError, TypeError):
                            continue
                    break

        if isinstance(primary_res_content, dict):
            if 'multi_results' in primary_res_content and isinstance(primary_res_content['multi_results'], dict):
                for proc_name, proc_res_obj_or_dict in primary_res_content['multi_results'].items():
                    if proc_res_obj_or_dict and not (
                            isinstance(proc_res_obj_or_dict, dict) and 'error' in proc_res_obj_or_dict):
                        _extract_from_single_res_raw(proc_res_obj_or_dict, prefix=f"{proc_name}_")
            else:
                _extract_from_single_res_raw(primary_res_content)
        elif primary_res_content:
            _extract_from_single_res_raw(primary_res_content, prefix="main_")
        return raw_metrics

    def _determine_focus_areas(self, query_analysis: Optional[QueryAnalysisResult]) -> List[str]:
        if not query_analysis: return ['general_overview']
        focus_areas = set()
        query_type_val = query_analysis.query_type.value if query_analysis.query_type else ""
        biz_scenario_val = query_analysis.business_scenario.value if query_analysis.business_scenario else ""
        if 'financial' in biz_scenario_val or 'risk' in query_type_val: focus_areas.add('financial_health')
        if 'growth' in biz_scenario_val or 'trend' in query_type_val: focus_areas.add('growth_analysis')
        if 'prediction' in query_type_val or 'scenario' in query_type_val: focus_areas.add(
            'future_outlook_and_opportunities')
        if 'risk' in query_type_val: focus_areas.add('risk_management')
        if 'user' in biz_scenario_val or 'user' in query_type_val: focus_areas.add('user_behavior_insights')
        if 'product' in biz_scenario_val or 'product' in query_type_val: focus_areas.add('product_performance_review')
        return list(focus_areas) if focus_areas else ['general_business_overview']

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        stats = self.orchestrator_stats.copy()
        total_q = stats.get('total_queries', 0)
        if total_q > 0:
            stats['success_rate'] = stats.get('successful_queries', 0) / total_q
            stats['failure_rate'] = stats.get('failed_queries', 0) / total_q
        else:
            stats['success_rate'] = 0.0;
            stats['failure_rate'] = 0.0
        cache_reqs = stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
        stats['cache_hit_rate'] = stats.get('cache_hits', 0) / cache_reqs if cache_reqs > 0 else 0.0
        return stats

    async def health_check(self) -> Dict[str, Any]:
        if not self.initialized:
            try:
                await self.initialize()
            except Exception as e:
                return {'status': 'unhealthy', 'reason': 'Orchestrator initialization failed during health check',
                        'error': str(e), 'timestamp': datetime.now().isoformat()}
        health_status = {
            'status': 'healthy', 'timestamp': datetime.now().isoformat(),
            'orchestrator_initialized': self.initialized, 'components_status': {},
            'ai_clients_status': {
                'claude_available': self.claude_client is not None and hasattr(self.claude_client, 'messages'),
                # Example check
                'gpt_available': self.gpt_client is not None and hasattr(self.gpt_client, 'chat')  # Example check
            },
            'db_connector_status': 'available' if self.db_connector else 'unavailable',
            'conversation_manager_status': 'available' if self.conversation_manager else 'unavailable',
            'statistics_snapshot': self.get_orchestrator_stats(),
            'active_sessions_count': len(self.active_sessions),
            'cache_current_size': len(self.result_cache)
        }
        components_to_check = {
            'QueryParser': self.query_parser, 'DataRequirementsAnalyzer': self.data_requirements_analyzer,
            'SmartDataFetcher': self.data_fetcher, 'FinancialDataAnalyzer': self.financial_data_analyzer,
            'InsightGenerator': self.insight_generator, 'CurrentDataProcessor': self.current_data_processor,
            'HistoricalAnalysisProcessor': self.historical_analysis_processor,
            'PredictionProcessor': self.prediction_processor,
            'APIConnector': getattr(self.data_fetcher, 'api_connector', None) if self.data_fetcher else None
        }
        all_components_healthy = True
        for name, comp_instance in components_to_check.items():
            if comp_instance and hasattr(comp_instance, 'health_check') and callable(
                    getattr(comp_instance, 'health_check')):
                try:
                    comp_health = await comp_instance.health_check()
                    status_from_comp = comp_health.get('status', 'unknown_status_key')
                    health_status['components_status'][name] = status_from_comp
                    if status_from_comp != 'healthy': all_components_healthy = False
                except Exception as e_hc:
                    health_status['components_status'][name] = f'error_calling_hc: {str(e_hc)}'
                    all_components_healthy = False
            elif comp_instance:
                health_status['components_status'][name] = 'available_no_hc_method'
            else:
                health_status['components_status'][name] = 'unavailable'; all_components_healthy = False

        if not self.db_connector and self.conversation_manager and getattr(self.conversation_manager, 'db',
                                                                           None) is None:
            health_status['components_status']['ConversationManager'] = 'running_in_memory_mode'
        elif self.db_connector and self.conversation_manager:
            health_status['components_status']['ConversationManager'] = 'healthy_with_db'

        if not all_components_healthy: health_status['status'] = 'degraded'
        ai_status = health_status['ai_clients_status']
        if not ai_status['claude_available'] and not ai_status['gpt_available']:
            health_status['status'] = 'critical_ai_unavailable'
        elif not ai_status['claude_available'] or not ai_status['gpt_available']:
            health_status['status'] = 'limited_one_ai_unavailable'
        return health_status

    async def close(self):
        logger.info("Closing IntelligentQAOrchestrator and its resources...")
        if self.data_fetcher and hasattr(self.data_fetcher, 'close') and callable(self.data_fetcher.close):
            if asyncio.iscoroutinefunction(self.data_fetcher.close):
                await self.data_fetcher.close()
            else:
                self.data_fetcher.close()
        if self.db_connector and hasattr(self.db_connector, 'close') and callable(self.db_connector.close):
            try:
                if asyncio.iscoroutinefunction(self.db_connector.close):
                    await self.db_connector.close()
                else:
                    self.db_connector.close()
            except Exception as db_close_err:
                logger.error(f"Error closing DatabaseConnector: {db_close_err}")
        self.result_cache.clear()
        self.active_sessions.clear()
        self.initialized = False
        logger.info("IntelligentQAOrchestrator closed successfully.")


# --- 全局实例管理 ---
_orchestrator_instance: Optional[IntelligentQAOrchestrator] = None


def get_orchestrator(claude_client_instance: Optional[ClaudeClient] = None,
                     gpt_client_instance: Optional[OpenAIClient] = None,
                     db_connector_instance: Optional[DatabaseConnector] = None,
                     app_config_instance: Optional[AppConfig] = None) -> IntelligentQAOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        logger.info("Creating new IntelligentQAOrchestrator singleton instance.")
        _orchestrator_instance = IntelligentQAOrchestrator(
            claude_client_instance, gpt_client_instance, db_connector_instance, app_config_instance
        )
    elif (claude_client_instance and _orchestrator_instance.claude_client != claude_client_instance) or \
            (gpt_client_instance and _orchestrator_instance.gpt_client != gpt_client_instance) or \
            (db_connector_instance and _orchestrator_instance.db_connector != db_connector_instance) or \
            (app_config_instance and _orchestrator_instance.app_config != app_config_instance):
        logger.info("Re-configuring existing orchestrator instance.")
        if claude_client_instance is not None: _orchestrator_instance.claude_client = claude_client_instance
        if gpt_client_instance is not None: _orchestrator_instance.gpt_client = gpt_client_instance
        if db_connector_instance is not None: _orchestrator_instance.db_connector = db_connector_instance
        if app_config_instance is not None:
            _orchestrator_instance.app_config = app_config_instance
            _orchestrator_instance.config = _orchestrator_instance._load_orchestrator_config()
        _orchestrator_instance.initialized = False
        logger.info("Orchestrator marked for re-initialization due to new configuration/clients.")

    return _orchestrator_instance

# (async def main() 已移除)