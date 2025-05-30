# core/orchestrator/intelligent_qa_orchestrator.py
"""
🚀 AI驱动的智能问答编排器
整个金融AI分析系统的核心大脑，负责协调所有组件协同工作

核心特点:
- 🧠 AI优先的智能路由与决策
- ⚡ 双AI协作的深度分析流程  
- 🔄 智能降级与容错处理
- 📊 完整的性能监控与统计
- 🎯 业务导向的结果输出
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib

# 导入所有已完成的核心组件
from core.analyzers.query_parser import SmartQueryParser, create_smart_query_parser
from core.analyzers.data_requirements_analyzer import DataRequirementsAnalyzer, create_data_requirements_analyzer
from core.analyzers.financial_data_analyzer import FinancialDataAnalyzer, create_financial_data_analyzer
from core.analyzers.insight_generator import InsightGenerator, create_insight_generator
from core.data_orchestration.smart_data_fetcher import SmartDataFetcher, create_smart_data_fetcher
from core.processors.current_data_processor import CurrentDataProcessor
from core.processors.historical_analysis_processor import HistoricalAnalysisProcessor
from core.processors.prediction_processor import PredictionProcessor

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """查询复杂度等级"""
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"
    EXPERT = "expert"


class ProcessingStrategy(Enum):
    """处理策略"""
    DIRECT_RESPONSE = "direct_response"  # 直接响应
    SINGLE_PROCESSOR = "single_processor"  # 单处理器
    MULTI_PROCESSOR = "multi_processor"  # 多处理器协作
    FULL_PIPELINE = "full_pipeline"  # 完整流水线


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    session_id: str  # 会话ID
    query_id: str  # 查询ID
    success: bool  # 处理是否成功
    
    # 核心结果
    response_text: str  # 主要响应文本
    insights: List[Dict[str, Any]]  # 业务洞察
    key_metrics: Dict[str, Any]  # 关键指标
    visualizations: List[Dict[str, Any]]  # 可视化数据
    
    # 处理信息
    processing_strategy: ProcessingStrategy  # 使用的处理策略
    processors_used: List[str]  # 使用的处理器
    ai_collaboration_summary: Dict[str, Any]  # AI协作摘要
    
    # 质量指标
    confidence_score: float  # 整体置信度
    data_quality_score: float  # 数据质量评分
    response_completeness: float  # 响应完整性
    
    # 性能指标
    total_processing_time: float  # 总处理时间
    ai_processing_time: float  # AI处理时间
    data_fetching_time: float  # 数据获取时间
    
    # 元数据
    processing_metadata: Dict[str, Any]  # 处理元数据
    error_info: Optional[Dict[str, Any]]  # 错误信息
    conversation_id: Optional[str]  # 对话ID
    timestamp: str  # 处理时间戳


class IntelligentQAOrchestrator:
    """
    🚀 AI驱动的智能问答编排器
    
    系统核心大脑，协调所有AI组件协同工作，
    提供智能的查询理解、数据获取、分析处理和洞察生成
    """
    
    def __init__(self, claude_client=None, gpt_client=None, config: Dict[str, Any] = None):
        """
        初始化智能编排器
        
        Args:
            claude_client: Claude客户端
            gpt_client: GPT客户端  
            config: 配置参数
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.config = config or self._load_default_config()
        
        # 初始化标志
        self.initialized = False
        
        # 核心组件 - 延迟初始化
        self.query_parser = None
        self.data_requirements_analyzer = None
        self.financial_data_analyzer = None
        self.insight_generator = None
        self.data_fetcher = None
        
        # 业务处理器
        self.current_data_processor = None
        self.historical_analysis_processor = None
        self.prediction_processor = None
        
        # 会话管理
        self.active_sessions = {}
        self.conversation_manager = ConversationManager()
        
        # 性能监控
        self.orchestrator_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time': 0.0,
            'avg_confidence_score': 0.0,
            'ai_collaboration_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processor_usage': {},
            'error_types': {}
        }
        
        # 智能缓存
        self.result_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 1800)  # 30分钟
        
        logger.info("IntelligentQAOrchestrator created, waiting for initialization")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            # 处理控制
            'max_processing_time': 120,  # 最大处理时间(秒)
            'enable_parallel_processing': True,
            'enable_intelligent_caching': True,
            'enable_ai_collaboration': True,
            
            # 质量控制
            'min_confidence_threshold': 0.6,
            'enable_result_validation': True,
            'enable_quality_monitoring': True,
            
            # 性能优化
            'cache_ttl': 1800,  # 缓存时间
            'max_concurrent_queries': 50,
            'enable_smart_routing': True,
            
            # AI配置
            'claude_timeout': 30,
            'gpt_timeout': 20,
            'max_ai_retries': 2,
            
            # 降级控制
            'enable_graceful_degradation': True,
            'fallback_response_enabled': True
        }
    
    async def initialize(self):
        """异步初始化所有组件"""
        if self.initialized:
            return
            
        try:
            logger.info("🚀 开始初始化智能编排器...")
            
            # 初始化核心分析组件
            self.query_parser = create_smart_query_parser(self.claude_client, self.gpt_client)
            self.data_requirements_analyzer = create_data_requirements_analyzer(self.claude_client, self.gpt_client)
            self.financial_data_analyzer = create_financial_data_analyzer(self.claude_client, self.gpt_client)
            self.insight_generator = create_insight_generator(self.claude_client, self.gpt_client)
            self.data_fetcher = create_smart_data_fetcher(self.claude_client, self.gpt_client, self.config)
            
            # 初始化业务处理器
            self.current_data_processor = CurrentDataProcessor(self.claude_client, self.gpt_client)
            self.historical_analysis_processor = HistoricalAnalysisProcessor(self.claude_client, self.gpt_client)
            self.prediction_processor = PredictionProcessor(self.claude_client, self.gpt_client)
            
            # 为处理器注入依赖
            self._inject_dependencies()
            
            self.initialized = True
            logger.info("✅ 智能编排器初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 智能编排器初始化失败: {str(e)}")
            raise
    
    def _inject_dependencies(self):
        """为各处理器注入依赖组件"""
        # 为处理器注入API连接器
        api_connector = getattr(self.data_fetcher, 'api_connector', None)
        if api_connector:
            self.current_data_processor.api_connector = api_connector
            self.historical_analysis_processor.api_connector = api_connector  
            self.prediction_processor.api_connector = api_connector
    
    # ============= 核心编排方法 =============
    
    async def process_intelligent_query(self, user_query: str, user_id: Optional[int] = None,
                                       conversation_id: Optional[str] = None,
                                       preferences: Dict[str, Any] = None) -> ProcessingResult:
        """
        🎯 智能查询处理 - 核心入口方法
        
        Args:
            user_query: 用户查询
            user_id: 用户ID
            conversation_id: 对话ID
            preferences: 用户偏好设置
            
        Returns:
            ProcessingResult: 完整的处理结果
        """
        # 确保已初始化
        if not self.initialized:
            await self.initialize()
        
        session_id = str(uuid.uuid4())
        query_id = f"query_{int(time.time())}"
        start_time = time.time()
        
        try:
            logger.info(f"🧠 开始智能查询处理: {user_query}")
            self.orchestrator_stats['total_queries'] += 1
            
            # 🔍 Step 1: 智能查询解析与理解
            parsing_start = time.time()
            query_analysis = await self._intelligent_query_parsing(user_query, conversation_id)
            parsing_time = time.time() - parsing_start
            
            # 🎯 Step 2: 智能路由决策
            routing_strategy = await self._determine_processing_strategy(query_analysis, preferences)
            
            # 📊 Step 3: 数据需求分析与获取
            data_start = time.time()
            data_acquisition_result = await self._orchestrate_data_acquisition(query_analysis, routing_strategy)
            data_time = time.time() - data_start
            
            # 🔬 Step 4: 智能业务处理
            processing_start = time.time()
            business_result = await self._orchestrate_business_processing(
                query_analysis, data_acquisition_result, routing_strategy
            )
            processing_time = time.time() - processing_start
            
            # 💡 Step 5: AI洞察生成与整合
            insight_start = time.time()
            insights = await self._orchestrate_insight_generation(
                business_result, query_analysis, data_acquisition_result
            )
            insight_time = time.time() - insight_start
            
            # 📝 Step 6: 响应格式化与优化
            response_text = await self._generate_intelligent_response(
                user_query, query_analysis, business_result, insights
            )
            
            # 📊 Step 7: 可视化数据生成
            visualizations = await self._generate_visualizations(business_result, query_analysis)
            
            total_time = time.time() - start_time
            
            # 构建成功结果
            result = ProcessingResult(
                session_id=session_id,
                query_id=query_id,
                success=True,
                
                response_text=response_text,
                insights=[insight.__dict__ if hasattr(insight, '__dict__') else insight for insight in insights],
                key_metrics=self._extract_key_metrics(business_result),
                visualizations=visualizations,
                
                processing_strategy=routing_strategy,
                processors_used=self._get_processors_used(routing_strategy),
                ai_collaboration_summary=self._get_ai_collaboration_summary(query_analysis, business_result),
                
                confidence_score=self._calculate_overall_confidence(query_analysis, business_result, insights),
                data_quality_score=data_acquisition_result.get('data_quality_score', 0.8),
                response_completeness=self._calculate_response_completeness(business_result, insights),
                
                total_processing_time=total_time,
                ai_processing_time=parsing_time + insight_time,
                data_fetching_time=data_time,
                
                processing_metadata={
                    'query_complexity': query_analysis.complexity.value,
                    'query_type': query_analysis.query_type.value,
                    'business_scenario': query_analysis.business_scenario.value,
                    'data_sources_used': data_acquisition_result.get('data_sources_used', []),
                    'processing_steps': ['parsing', 'routing', 'data_acquisition', 'business_processing', 'insight_generation', 'response_formatting'],
                    'ai_models_used': ['claude', 'gpt'] if self.claude_client and self.gpt_client else []
                },
                error_info=None,
                conversation_id=conversation_id,
                timestamp=datetime.now().isoformat()
            )
            
            # 更新统计信息
            self._update_orchestrator_stats(result)
            
            # 缓存结果
            if self.config.get('enable_intelligent_caching', True):
                await self._cache_result(user_query, result)
            
            # 更新对话历史
            if conversation_id:
                self.conversation_manager.add_exchange(conversation_id, user_query, result)
            
            logger.info(f"✅ 智能查询处理完成，耗时: {total_time:.2f}秒，置信度: {result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            # 错误处理和降级
            logger.error(f"❌ 智能查询处理失败: {str(e)}")
            return await self._handle_processing_error(
                session_id, query_id, user_query, str(e), time.time() - start_time
            )
    
    # ============= 智能解析与路由 =============
    
    async def _intelligent_query_parsing(self, user_query: str, conversation_id: Optional[str] = None):
        """AI驱动的智能查询解析"""
        try:
            # 获取对话上下文
            context = {}
            if conversation_id:
                context = self.conversation_manager.get_context(conversation_id)
            
            # 检查缓存
            cache_key = self._generate_query_cache_key(user_query, context)
            cached_result = self._get_cached_parsing_result(cache_key)
            if cached_result:
                self.orchestrator_stats['cache_hits'] += 1
                return cached_result
            
            # AI解析
            query_analysis = await self.query_parser.parse_complex_query(user_query, context)
            
            # 缓存解析结果
            self._cache_parsing_result(cache_key, query_analysis)
            self.orchestrator_stats['cache_misses'] += 1
            
            return query_analysis
            
        except Exception as e:
            logger.error(f"查询解析失败: {str(e)}")
            # 降级到基础解析
            return await self._fallback_query_parsing(user_query)
    
    async def _determine_processing_strategy(self, query_analysis, preferences: Dict[str, Any] = None) -> ProcessingStrategy:
        """智能确定处理策略"""
        try:
            complexity = query_analysis.complexity
            query_type = query_analysis.query_type
            
            # 🧠 使用AI决定处理策略
            if self.claude_client and self.config.get('enable_ai_collaboration', True):
                strategy = await self._ai_determine_strategy(query_analysis, preferences)
                if strategy:
                    return strategy
            
            # 基于规则的策略决策
            if complexity.value == "simple":
                return ProcessingStrategy.DIRECT_RESPONSE
            elif complexity.value == "medium":
                return ProcessingStrategy.SINGLE_PROCESSOR
            elif complexity.value == "complex":
                return ProcessingStrategy.MULTI_PROCESSOR
            else:  # expert
                return ProcessingStrategy.FULL_PIPELINE
                
        except Exception as e:
            logger.error(f"策略决策失败: {str(e)}")
            return ProcessingStrategy.SINGLE_PROCESSOR
    
    async def _ai_determine_strategy(self, query_analysis, preferences) -> Optional[ProcessingStrategy]:
        """AI辅助策略决策"""
        try:
            prompt = f"""
            作为智能系统架构师，请为以下查询分析选择最佳处理策略：

            查询分析结果:
            - 复杂度: {query_analysis.complexity.value}
            - 查询类型: {query_analysis.query_type.value}
            - 业务场景: {query_analysis.business_scenario.value}
            - 预估时间: {query_analysis.estimated_total_time}秒
            - 置信度: {query_analysis.confidence_score}

            用户偏好: {json.dumps(preferences or {}, ensure_ascii=False)}

            可选策略:
            1. direct_response - 直接响应(简单查询)
            2. single_processor - 单处理器(标准分析)  
            3. multi_processor - 多处理器协作(复杂分析)
            4. full_pipeline - 完整流水线(专家级分析)

            请选择最优策略，考虑效率、准确性和用户体验。
            只返回策略名称，如: "single_processor"
            """
            
            result = await self.claude_client.analyze_complex_query(prompt, {
                "query_analysis": query_analysis.__dict__ if hasattr(query_analysis, '__dict__') else query_analysis,
                "preferences": preferences
            })
            
            if result.get('success'):
                strategy_name = result.get('response', '').strip().lower()
                strategy_map = {
                    'direct_response': ProcessingStrategy.DIRECT_RESPONSE,
                    'single_processor': ProcessingStrategy.SINGLE_PROCESSOR,
                    'multi_processor': ProcessingStrategy.MULTI_PROCESSOR,
                    'full_pipeline': ProcessingStrategy.FULL_PIPELINE
                }
                return strategy_map.get(strategy_name)
                
        except Exception as e:
            logger.warning(f"AI策略决策失败: {str(e)}")
            
        return None
    
    # ============= 数据编排 =============
    
    async def _orchestrate_data_acquisition(self, query_analysis, strategy: ProcessingStrategy) -> Dict[str, Any]:
        """编排数据获取流程"""
        try:
            # 🎯 Step 1: 分析数据需求
            data_plan = await self.data_requirements_analyzer.analyze_data_requirements(query_analysis)
            
            # 🚀 Step 2: 执行数据获取
            data_result = await self.data_fetcher.execute_data_acquisition_plan(data_plan)
            
            return {
                'success': True,
                'data_plan': data_plan,
                'execution_result': data_result,
                'data_quality_score': data_result.confidence_level,
                'data_sources_used': data_result.data_sources_used,
                'fetched_data': data_result.fetched_data,
                'processed_data': data_result.processed_data,
                'time_series_data': data_result.time_series_data
            }
            
        except Exception as e:
            logger.error(f"数据获取编排失败: {str(e)}")
            # 降级到基础数据获取
            return await self._fallback_data_acquisition(query_analysis)
    
    async def _fallback_data_acquisition(self, query_analysis) -> Dict[str, Any]:
        """降级数据获取"""
        try:
            # 只获取基础系统数据
            api_connector = getattr(self.data_fetcher, 'api_connector', None)
            if api_connector:
                system_data = await api_connector.get_system_data()
                return {
                    'success': True,
                    'fallback': True,
                    'data_quality_score': 0.6,
                    'data_sources_used': ['system'],
                    'fetched_data': {'system': system_data},
                    'processed_data': {'system_data': system_data.get('data', {})},
                    'time_series_data': {}
                }
        except Exception as e:
            logger.error(f"降级数据获取也失败: {str(e)}")
            
        return {
            'success': False,
            'error': '数据获取失败',
            'data_quality_score': 0.0,
            'data_sources_used': [],
            'fetched_data': {},
            'processed_data': {},
            'time_series_data': {}
        }
    
    # ============= 业务处理编排 =============
    
    async def _orchestrate_business_processing(self, query_analysis, data_result, strategy: ProcessingStrategy) -> Dict[str, Any]:
        """编排业务处理流程"""
        try:
            complexity = query_analysis.complexity
            query_type = query_analysis.query_type
            
            # 根据策略选择处理方式
            if strategy == ProcessingStrategy.DIRECT_RESPONSE:
                return await self._direct_response_processing(query_analysis, data_result)
            elif strategy == ProcessingStrategy.SINGLE_PROCESSOR:
                return await self._single_processor_processing(query_analysis, data_result)
            elif strategy == ProcessingStrategy.MULTI_PROCESSOR:
                return await self._multi_processor_processing(query_analysis, data_result)
            else:  # FULL_PIPELINE
                return await self._full_pipeline_processing(query_analysis, data_result)
                
        except Exception as e:
            logger.error(f"业务处理编排失败: {str(e)}")
            return await self._fallback_business_processing(query_analysis, data_result)
    
    async def _direct_response_processing(self, query_analysis, data_result) -> Dict[str, Any]:
        """直接响应处理"""
        # 简单查询直接使用当前数据处理器
        result = await self.current_data_processor.process_current_data_query(query_analysis.original_query)
        return {
            'processing_type': 'direct_response',
            'primary_result': result,
            'confidence_score': result.response_confidence,
            'processing_time': result.processing_time
        }
    
    async def _single_processor_processing(self, query_analysis, data_result) -> Dict[str, Any]:
        """单处理器处理"""
        query_type = query_analysis.query_type
        
        # 根据查询类型选择合适的处理器
        if 'trend' in query_type.value or 'historical' in query_type.value:
            processor = self.historical_analysis_processor
            result = await processor.process_historical_analysis_query(
                query_analysis.original_query, {'data_result': data_result}
            )
        elif 'prediction' in query_type.value:
            processor = self.prediction_processor  
            result = await processor.process_prediction_query(
                query_analysis.original_query, {'data_result': data_result}
            )
        else:
            processor = self.current_data_processor
            result = await processor.process_current_data_query(query_analysis.original_query)
        
        return {
            'processing_type': 'single_processor',
            'processor_used': processor.__class__.__name__,
            'primary_result': result,
            'confidence_score': getattr(result, 'confidence_score', 0.8),
            'processing_time': getattr(result, 'processing_time', 0.0)
        }
    
    async def _multi_processor_processing(self, query_analysis, data_result) -> Dict[str, Any]:
        """多处理器协作处理"""
        try:
            # 并行运行多个处理器
            tasks = []
            
            # 当前数据处理
            tasks.append(self.current_data_processor.process_current_data_query(query_analysis.original_query))
            
            # 历史分析处理  
            tasks.append(self.historical_analysis_processor.process_historical_analysis_query(
                query_analysis.original_query, {'data_result': data_result}
            ))
            
            # 根据查询类型决定是否包含预测
            if 'prediction' in query_analysis.query_type.value:
                tasks.append(self.prediction_processor.process_prediction_query(
                    query_analysis.original_query, {'data_result': data_result}
                ))
            
            # 并行执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 整合结果
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            return {
                'processing_type': 'multi_processor',
                'processors_used': ['CurrentDataProcessor', 'HistoricalAnalysisProcessor', 'PredictionProcessor'],
                'current_data_result': successful_results[0] if len(successful_results) > 0 else None,
                'historical_result': successful_results[1] if len(successful_results) > 1 else None,
                'prediction_result': successful_results[2] if len(successful_results) > 2 else None,
                'confidence_score': sum(getattr(r, 'confidence_score', 0.8) for r in successful_results) / max(len(successful_results), 1),
                'processing_time': max(getattr(r, 'processing_time', 0.0) for r in successful_results) if successful_results else 0.0
            }
            
        except Exception as e:
            logger.error(f"多处理器协作失败: {str(e)}")
            # 降级到单处理器
            return await self._single_processor_processing(query_analysis, data_result)
    
    async def _full_pipeline_processing(self, query_analysis, data_result) -> Dict[str, Any]:
        """完整流水线处理"""
        # 专家级查询的完整处理流程
        pipeline_results = {}
        
        try:
            # Step 1: 当前状态分析
            current_result = await self.current_data_processor.process_current_data_query(query_analysis.original_query)
            pipeline_results['current_analysis'] = current_result
            
            # Step 2: 历史深度分析
            historical_result = await self.historical_analysis_processor.process_historical_analysis_query(
                query_analysis.original_query, {'data_result': data_result}
            )
            pipeline_results['historical_analysis'] = historical_result
            
            # Step 3: 预测与场景分析
            prediction_result = await self.prediction_processor.process_prediction_query(
                query_analysis.original_query, {'data_result': data_result}
            )
            pipeline_results['prediction_analysis'] = prediction_result
            
            # Step 4: 深度金融分析
            financial_analysis = await self.financial_data_analyzer.analyze_business_performance('comprehensive', 90)
            pipeline_results['financial_analysis'] = financial_analysis
            
            return {
                'processing_type': 'full_pipeline',
                'processors_used': ['CurrentDataProcessor', 'HistoricalAnalysisProcessor', 'PredictionProcessor', 'FinancialDataAnalyzer'],
                'pipeline_results': pipeline_results,
                'confidence_score': 0.9,  # 完整流水线的高置信度
                'processing_time': sum(getattr(r, 'processing_time', 1.0) for r in pipeline_results.values())
            }
            
        except Exception as e:
            logger.error(f"完整流水线处理失败: {str(e)}")
            # 降级到多处理器
            return await self._multi_processor_processing(query_analysis, data_result)
    
    async def _fallback_business_processing(self, query_analysis, data_result) -> Dict[str, Any]:
        """业务处理降级方案"""
        try:
            # 降级到最基础的当前数据处理
            result = await self.current_data_processor.process_current_data_query(query_analysis.original_query)
            return {
                'processing_type': 'fallback',
                'primary_result': result,
                'confidence_score': 0.6,
                'processing_time': getattr(result, 'processing_time', 0.0),
                'fallback_reason': '业务处理器失败，使用基础处理'
            }
        except Exception as e:
            logger.error(f"降级业务处理也失败: {str(e)}")
            return {
                'processing_type': 'error',
                'error': str(e),
                'confidence_score': 0.0,
                'processing_time': 0.0
            }
    
    # ============= 洞察生成编排 =============
    
    async def _orchestrate_insight_generation(self, business_result, query_analysis, data_result) -> List[Any]:
        """编排洞察生成流程"""
        try:
            # 准备分析结果用于洞察生成
            analysis_results = []
            
            # 从业务处理结果中提取分析结果
            if 'primary_result' in business_result:
                analysis_results.append(business_result['primary_result'])
            
            if 'pipeline_results' in business_result:
                analysis_results.extend(business_result['pipeline_results'].values())
            
            # 生成综合洞察
            insights, metadata = await self.insight_generator.generate_comprehensive_insights(
                analysis_results=analysis_results,
                user_context=None,
                focus_areas=self._determine_focus_areas(query_analysis)
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"洞察生成失败: {str(e)}")
            return []
    
    def _determine_focus_areas(self, query_analysis) -> List[str]:
        """根据查询分析确定洞察重点领域"""
        focus_areas = []
        
        query_type = query_analysis.query_type.value
        business_scenario = query_analysis.business_scenario.value
        
        if 'financial' in business_scenario or 'risk' in query_type:
            focus_areas.append('financial_health')
        if 'growth' in business_scenario or 'trend' in query_type:
            focus_areas.append('growth_analysis')
        if 'prediction' in query_type:
            focus_areas.append('opportunity')
        if 'risk' in query_type:
            focus_areas.append('risk_management')
            
        return focus_areas or ['financial_health']
    
    # ============= 响应生成与格式化 =============
    
    async def _generate_intelligent_response(self, user_query: str, query_analysis, 
                                           business_result: Dict[str, Any], 
                                           insights: List[Any]) -> str:
        """AI驱动的智能响应生成"""
        try:
            if not self.claude_client:
                return self._generate_basic_response(business_result, insights)
            
            # 使用Claude生成专业响应
            prompt = f"""
            作为专业的金融分析师，请基于以下分析结果生成用户友好的回答：

            用户问题: "{user_query}"

            查询分析:
            - 查询类型: {query_analysis.query_type.value}
            - 复杂度: {query_analysis.complexity.value}
            - 业务场景: {query_analysis.business_scenario.value}

            分析结果摘要:
            {json.dumps(self._summarize_business_result(business_result), ensure_ascii=False, indent=2)}

            业务洞察:
            {json.dumps([insight.get('title', '洞察') + ': ' + insight.get('summary', '') for insight in insights[:3]], ensure_ascii=False)}

            请生成：
            1. 直接回答用户问题
            2. 突出关键数据和发现
            3. 提供业务洞察和建议
            4. 语言专业但易懂

            要求简洁明了，重点突出，不超过500字。
            """
            
            result = await self.claude_client.analyze_complex_query(prompt, {
                'query_analysis': query_analysis.__dict__ if hasattr(query_analysis, '__dict__') else query_analysis,
                'business_result': business_result,
                'insights': insights[:3]
            })
            
            if result.get('success'):
                return result.get('response', '分析完成，请查看详细结果。')
            
        except Exception as e:
            logger.error(f"AI响应生成失败: {str(e)}")
        
        # 降级到基础响应
        return self._generate_basic_response(business_result, insights)
    
    def _generate_basic_response(self, business_result: Dict[str, Any], insights: List[Any]) -> str:
        """生成基础响应"""
        response_parts = ["根据您的查询，我已完成分析：\n"]
        
        # 添加处理类型信息
        processing_type = business_result.get('processing_type', 'unknown')
        response_parts.append(f"处理方式: {processing_type}")
        
        # 添加关键指标
        if 'primary_result' in business_result:
            result = business_result['primary_result']
            if hasattr(result, 'main_answer'):
                response_parts.append(f"\n主要发现: {result.main_answer}")
        
        # 添加洞察
        if insights:
            response_parts.append(f"\n业务洞察: 发现{len(insights)}条重要洞察")
            for i, insight in enumerate(insights[:3], 1):
                title = insight.get('title', f'洞察{i}')
                summary = insight.get('summary', '重要发现')
                response_parts.append(f"  {i}. {title}: {summary}")
        
        # 添加置信度
        confidence = business_result.get('confidence_score', 0.8)
        response_parts.append(f"\n分析置信度: {confidence:.1%}")
        
        return "\n".join(response_parts)
    
    def _summarize_business_result(self, business_result: Dict[str, Any]) -> Dict[str, Any]:
        """摘要业务结果用于AI处理"""
        summary = {
            'processing_type': business_result.get('processing_type'),
            'confidence_score': business_result.get('confidence_score'),
            'processing_time': business_result.get('processing_time')
        }
        
        # 添加主要结果摘要
        if 'primary_result' in business_result:
            result = business_result['primary_result']
            if hasattr(result, 'main_answer'):
                summary['main_answer'] = result.main_answer
            if hasattr(result, 'key_metrics'):
                summary['key_metrics'] = result.key_metrics
        
        return summary
    
    async def _generate_visualizations(self, business_result: Dict[str, Any], 
                                     query_analysis) -> List[Dict[str, Any]]:
        """生成可视化数据"""
        visualizations = []
        
        try:
            # 基于分析结果生成合适的可视化
            processing_type = business_result.get('processing_type')
            
            if processing_type in ['multi_processor', 'full_pipeline']:
                # 多维度数据可视化
                if 'historical_result' in business_result:
                    visualizations.append({
                        'type': 'line_chart',
                        'title': '历史趋势分析',
                        'data': self._extract_trend_data(business_result['historical_result']),
                        'config': {'x_axis': 'date', 'y_axis': 'value'}
                    })
                
                if 'prediction_result' in business_result:
                    visualizations.append({
                        'type': 'forecast_chart', 
                        'title': '预测分析',
                        'data': self._extract_prediction_data(business_result['prediction_result']),
                        'config': {'show_confidence_interval': True}
                    })
            
            # 基础指标图表
            if 'primary_result' in business_result:
                result = business_result['primary_result']
                if hasattr(result, 'key_metrics') and result.key_metrics:
                    visualizations.append({
                        'type': 'metrics_dashboard',
                        'title': '关键指标',
                        'data': result.key_metrics,
                        'config': {'layout': 'grid'}
                    })
                    
        except Exception as e:
            logger.warning(f"可视化生成失败: {str(e)}")
        
        return visualizations
    
    def _extract_trend_data(self, historical_result) -> Dict[str, Any]:
        """从历史结果中提取趋势数据"""
        # 简化实现，实际中会根据具体结果结构提取
        return {
            'dates': ['2024-05-01', '2024-05-15', '2024-05-30'],
            'values': [100, 120, 115],
            'trend': 'increasing'
        }
    
    def _extract_prediction_data(self, prediction_result) -> Dict[str, Any]:
        """从预测结果中提取预测数据"""
        # 简化实现
        return {
            'historical': [100, 120, 115],
            'predicted': [125, 130, 140],
            'confidence_upper': [135, 145, 160],
            'confidence_lower': [115, 120, 125]
        }
    
    # ============= 工具方法 =============
    
    def _extract_key_metrics(self, business_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键指标"""
        metrics = {}
        
        try:
            if 'primary_result' in business_result:
                result = business_result['primary_result']
                if hasattr(result, 'key_metrics'):
                    metrics.update(result.key_metrics)
            
            if 'pipeline_results' in business_result:
                for name, result in business_result['pipeline_results'].items():
                    if hasattr(result, 'key_metrics'):
                        metrics[f'{name}_metrics'] = result.key_metrics
                        
        except Exception as e:
            logger.warning(f"关键指标提取失败: {str(e)}")
        
        return metrics
    
    def _get_processors_used(self, strategy: ProcessingStrategy) -> List[str]:
        """获取使用的处理器列表"""
        if strategy == ProcessingStrategy.DIRECT_RESPONSE:
            return ['CurrentDataProcessor']
        elif strategy == ProcessingStrategy.SINGLE_PROCESSOR:
            return ['CurrentDataProcessor']  # 根据实际使用的处理器动态确定
        elif strategy == ProcessingStrategy.MULTI_PROCESSOR:
            return ['CurrentDataProcessor', 'HistoricalAnalysisProcessor', 'PredictionProcessor']
        else:  # FULL_PIPELINE
            return ['CurrentDataProcessor', 'HistoricalAnalysisProcessor', 'PredictionProcessor', 'FinancialDataAnalyzer']
    
    def _get_ai_collaboration_summary(self, query_analysis, business_result) -> Dict[str, Any]:
        """获取AI协作摘要"""
        return {
            'claude_used': self.claude_client is not None,
            'gpt_used': self.gpt_client is not None,
            'ai_enhanced_parsing': query_analysis.confidence_score > 0.8,
            'ai_enhanced_insights': True,
            'collaboration_level': 'high' if self.claude_client and self.gpt_client else 'basic'
        }
    
    def _calculate_overall_confidence(self, query_analysis, business_result, insights) -> float:
        """计算整体置信度"""
        confidence_factors = []
        
        # 查询解析置信度
        confidence_factors.append(query_analysis.confidence_score)
        
        # 业务处理置信度
        business_confidence = business_result.get('confidence_score', 0.8)
        confidence_factors.append(business_confidence)
        
        # 洞察质量
        insight_confidence = 0.8 if insights else 0.6
        confidence_factors.append(insight_confidence)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_response_completeness(self, business_result, insights) -> float:
        """计算响应完整性"""
        completeness_score = 0.5  # 基础分
        
        if business_result.get('primary_result'):
            completeness_score += 0.2
        
        if insights:
            completeness_score += 0.2
        
        if business_result.get('processing_type') in ['multi_processor', 'full_pipeline']:
            completeness_score += 0.1
        
        return min(completeness_score, 1.0)
    
    # ============= 缓存管理 =============
    
    def _generate_query_cache_key(self, query: str, context: Dict[str, Any] = None) -> str:
        """生成查询缓存键"""
        cache_data = f"{query}_{json.dumps(context or {}, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_cached_parsing_result(self, cache_key: str):
        """获取缓存的解析结果"""
        if cache_key in self.result_cache:
            cache_entry = self.result_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['data']
            else:
                del self.result_cache[cache_key]
        return None
    
    def _cache_parsing_result(self, cache_key: str, result):
        """缓存解析结果"""
        self.result_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        # 限制缓存大小
        if len(self.result_cache) > 1000:
            oldest_key = min(self.result_cache.keys(), 
                           key=lambda k: self.result_cache[k]['timestamp'])
            del self.result_cache[oldest_key]
    
    async def _cache_result(self, query: str, result: ProcessingResult):
        """缓存完整结果"""
        cache_key = f"result_{self._generate_query_cache_key(query)}"
        self.result_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
    
    # ============= 降级处理 =============
    
    async def _fallback_query_parsing(self, user_query: str):
        """降级查询解析"""
        # 创建基础查询分析结果
        from core.analyzers.query_parser import QueryComplexity, QueryType, BusinessScenario
        
        class MockQueryResult:
            def __init__(self):
                self.original_query = user_query
                self.complexity = QueryComplexity.MEDIUM
                self.query_type = QueryType.DATA_RETRIEVAL
                self.business_scenario = BusinessScenario.DAILY_OPERATIONS
                self.confidence_score = 0.6
                self.estimated_total_time = 10.0
                
        return MockQueryResult()
    
    async def _handle_processing_error(self, session_id: str, query_id: str, 
                                     user_query: str, error: str, 
                                     processing_time: float) -> ProcessingResult:
        """处理错误并返回降级结果"""
        self.orchestrator_stats['failed_queries'] += 1
        
        error_type = self._classify_error(error)
        self.orchestrator_stats['error_types'][error_type] = self.orchestrator_stats['error_types'].get(error_type, 0) + 1
        
        # 尝试降级响应
        fallback_response = "抱歉，处理您的查询时遇到了技术问题。"
        
        if self.config.get('fallback_response_enabled', True):
            try:
                # 尝试提供基础响应
                fallback_response = await self._generate_fallback_response(user_query, error)
            except:
                pass
        
        return ProcessingResult(
            session_id=session_id,
            query_id=query_id,
            success=False,
            
            response_text=fallback_response,
            insights=[],
            key_metrics={},
            visualizations=[],
            
            processing_strategy=ProcessingStrategy.DIRECT_RESPONSE,
            processors_used=[],
            ai_collaboration_summary={'error': True},
            
            confidence_score=0.0,
            data_quality_score=0.0,
            response_completeness=0.2,
            
            total_processing_time=processing_time,
            ai_processing_time=0.0,
            data_fetching_time=0.0,
            
            processing_metadata={'error_occurred': True},
            error_info={'error_message': error, 'error_type': error_type},
            conversation_id=None,
            timestamp=datetime.now().isoformat()
        )
    
    def _classify_error(self, error: str) -> str:
        """分类错误类型"""
        error_lower = error.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower:
            return 'connection'
        elif 'validation' in error_lower:
            return 'validation'
        elif 'ai' in error_lower or 'claude' in error_lower or 'gpt' in error_lower:
            return 'ai_service'
        else:
            return 'unknown'
    
    async def _generate_fallback_response(self, user_query: str, error: str) -> str:
        """生成降级响应"""
        if 'balance' in user_query.lower() or '余额' in user_query:
            return "系统暂时无法获取最新余额信息，请稍后重试或联系客服。"
        elif 'trend' in user_query.lower() or '趋势' in user_query:
            return "趋势分析功能暂时不可用，我们正在处理技术问题。"
        else:
            return "抱歉，暂时无法处理您的查询。系统正在恢复中，请稍后重试。"
    
    # ============= 统计与监控 =============
    
    def _update_orchestrator_stats(self, result: ProcessingResult):
        """更新编排器统计信息"""
        if result.success:
            self.orchestrator_stats['successful_queries'] += 1
        
        # 更新平均处理时间
        total_queries = self.orchestrator_stats['total_queries']
        current_avg_time = self.orchestrator_stats['avg_processing_time']
        new_avg_time = ((current_avg_time * (total_queries - 1)) + result.total_processing_time) / total_queries
        self.orchestrator_stats['avg_processing_time'] = new_avg_time
        
        # 更新平均置信度
        current_avg_confidence = self.orchestrator_stats['avg_confidence_score']
        new_avg_confidence = ((current_avg_confidence * (total_queries - 1)) + result.confidence_score) / total_queries
        self.orchestrator_stats['avg_confidence_score'] = new_avg_confidence
        
        # 更新处理器使用统计
        for processor in result.processors_used:
            self.orchestrator_stats['processor_usage'][processor] = \
                self.orchestrator_stats['processor_usage'].get(processor, 0) + 1
        
        # AI协作统计
        if result.ai_collaboration_summary.get('claude_used') and result.ai_collaboration_summary.get('gpt_used'):
            self.orchestrator_stats['ai_collaboration_count'] += 1
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """获取编排器统计信息"""
        stats = self.orchestrator_stats.copy()
        
        # 计算成功率
        if stats['total_queries'] > 0:
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']
            stats['failure_rate'] = stats['failed_queries'] / stats['total_queries']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # 缓存命中率
        cache_requests = stats['cache_hits'] + stats['cache_misses']
        if cache_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / cache_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'orchestrator_initialized': self.initialized,
            'components_status': {},
            'ai_clients_status': {
                'claude_available': self.claude_client is not None,
                'gpt_available': self.gpt_client is not None
            },
            'statistics': self.get_orchestrator_stats(),
            'active_sessions': len(self.active_sessions),
            'cache_size': len(self.result_cache)
        }
        
        # 检查各组件状态
        if self.initialized:
            components = {
                'query_parser': self.query_parser,
                'data_requirements_analyzer': self.data_requirements_analyzer,
                'financial_data_analyzer': self.financial_data_analyzer,
                'insight_generator': self.insight_generator,
                'data_fetcher': self.data_fetcher,
                'current_data_processor': self.current_data_processor,
                'historical_analysis_processor': self.historical_analysis_processor,
                'prediction_processor': self.prediction_processor
            }
            
            for name, component in components.items():
                try:
                    if hasattr(component, 'health_check'):
                        component_health = await component.health_check()
                        health_status['components_status'][name] = component_health.get('status', 'unknown')
                    else:
                        health_status['components_status'][name] = 'available' if component else 'unavailable'
                except Exception as e:
                    health_status['components_status'][name] = f'error: {str(e)}'
        
        # 确定整体状态
        if not self.initialized:
            health_status['status'] = 'initializing'
        elif any(status == 'error' for status in health_status['components_status'].values()):
            health_status['status'] = 'degraded'
        elif not health_status['ai_clients_status']['claude_available']:
            health_status['status'] = 'limited'
        
        return health_status
    
    # ============= 会话管理 =============
    
    async def close(self):
        """关闭编排器，清理资源"""
        try:
            if self.data_fetcher and hasattr(self.data_fetcher, 'close'):
                await self.data_fetcher.close()
            
            self.result_cache.clear()
            self.active_sessions.clear()
            
            logger.info("IntelligentQAOrchestrator closed successfully")
            
        except Exception as e:
            logger.error(f"关闭编排器时出错: {str(e)}")


class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.conversations = {}
        self.max_history = 50
    
    def add_exchange(self, conversation_id: str, query: str, result: ProcessingResult):
        """添加对话记录"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            'query': query,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # 限制历史记录长度
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        if conversation_id not in self.conversations:
            return {}
        
        history = self.conversations[conversation_id]
        recent_queries = [exchange['query'] for exchange in history[-5:]]
        
        return {
            'conversation_id': conversation_id,
            'recent_queries': recent_queries,
            'query_count': len(history)
        }
    
    def get_user_conversations(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """获取用户对话列表"""
        # 简化实现，实际中会根据user_id过滤
        all_conversations = []
        for conv_id, exchanges in self.conversations.items():
            if exchanges:
                last_exchange = exchanges[-1]
                all_conversations.append({
                    'conversation_id': conv_id,
                    'last_query': last_exchange['query'],
                    'last_timestamp': last_exchange['timestamp'],
                    'exchange_count': len(exchanges)
                })
        
        return all_conversations[offset:offset+limit]


# ============= 全局编排器实例管理 =============

_orchestrator_instance = None

def get_orchestrator(claude_client=None, gpt_client=None, config: Dict[str, Any] = None) -> IntelligentQAOrchestrator:
    """
    获取全局编排器实例（单例模式）
    
    Args:
        claude_client: Claude客户端
        gpt_client: GPT客户端
        config: 配置参数
        
    Returns:
        IntelligentQAOrchestrator: 编排器实例
    """
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = IntelligentQAOrchestrator(claude_client, gpt_client, config)
    
    return _orchestrator_instance

def create_orchestrator(claude_client=None, gpt_client=None, config: Dict[str, Any] = None) -> IntelligentQAOrchestrator:
    """
    创建新的编排器实例
    
    Args:
        claude_client: Claude客户端
        gpt_client: GPT客户端
        config: 配置参数
        
    Returns:
        IntelligentQAOrchestrator: 新的编排器实例
    """
    return IntelligentQAOrchestrator(claude_client, gpt_client, config)


# ============= 使用示例 =============

async def main():
    """使用示例"""
    
    print("=== AI驱动的智能问答编排器测试 ===")
    
    # 获取全局编排器实例
    orchestrator = get_orchestrator()
    
    try:
        # 初始化编排器
        await orchestrator.initialize()
        
        # 测试查询
        test_queries = [
            "今天系统总余额是多少？",
            "过去30天的入金趋势如何？", 
            "根据历史数据预测下月资金情况",
            "假设50%复投50%提现对资金的影响"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 测试查询 {i} ---")
            print(f"查询: {query}")
            
            # 处理查询
            result = await orchestrator.process_intelligent_query(
                user_query=query,
                user_id=1,
                conversation_id=f"test_conv_{i}"
            )
            
            print(f"成功: {'是' if result.success else '否'}")
            print(f"策略: {result.processing_strategy.value}")
            print(f"置信度: {result.confidence_score:.2f}")
            print(f"处理时间: {result.total_processing_time:.2f}秒")
            print(f"洞察数量: {len(result.insights)}")
            
            if result.response_text:
                print(f"响应: {result.response_text[:200]}...")
        
        # 系统健康检查
        print(f"\n=== 系统健康检查 ===")
        health = await orchestrator.health_check()
        print(f"状态: {health['status']}")
        print(f"AI客户端: Claude={health['ai_clients_status']['claude_available']}, GPT={health['ai_clients_status']['gpt_available']}")
        
        # 统计信息
        stats = orchestrator.get_orchestrator_stats()
        print(f"\n=== 统计信息 ===")
        print(f"总查询: {stats['total_queries']}")
        print(f"成功率: {stats.get('success_rate', 0):.1%}")
        print(f"平均处理时间: {stats['avg_processing_time']:.2f}秒")
        print(f"平均置信度: {stats['avg_confidence_score']:.2f}")
        
    finally:
        await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(main())