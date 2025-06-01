"""
精简重构后的智能问答编排器
核心改进：
1. 专注于流程编排，不再处理prompt构建和查询类型检测
2. 集成QueryTypeDetector、PromptManager、EnhancedAPIStrategyExtractor
3. 保持核心数据处理和响应生成功能
4. 简化代码结构，提高维护性
"""
import json
import logging
import os
import time
import uuid
import asyncio
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# 核心组件导入
from core.data_extraction.semantic_collector import SemanticDataCollector
from core.data_extraction.claude_extractor import ClaudeIntelligentExtractor
from core.data_orchestration.smart_data_fetcher import SmartDataFetcher, create_smart_data_fetcher
from utils.calculators.statistical_calculator import UnifiedCalculator, create_unified_calculator
from utils.formatters.financial_formatter import FinancialFormatter, create_financial_formatter
from data.models.conversation import ConversationManager, create_conversation_manager
from data.connectors.database_connector import DatabaseConnector, create_database_connector

# 🆕 新组件导入
from core.detectors.query_type_detector import QueryTypeDetector, QueryType, QueryTypeResult, create_query_type_detector
from core.prompts.prompt_manager import PromptManager, create_prompt_manager
from core.analyzers.ai_strategy_extractor import EnhancedAPIStrategyExtractor, ExtractedStrategy, create_enhanced_strategy_extractor

# AI 客户端导入
from core.models.claude_client import ClaudeClient
from core.models.openai_client import OpenAIClient
from config import Config as AppConfig

logger = logging.getLogger(__name__)

# 单例实例
_orchestrator_instance = None

def get_orchestrator():
    """获取智能问答编排器的单例实例"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        from core.models.claude_client import ClaudeClient
        from core.models.openai_client import OpenAIClient
        from config import Config
        from data.connectors.database_connector import create_database_connector

        config = Config()
        claude_client = ClaudeClient()
        gpt_client = OpenAIClient()
        db_connector = create_database_connector()

        _orchestrator_instance = IntelligentQAOrchestrator(
            claude_client=claude_client,
            gpt_client=gpt_client,
            db_connector=db_connector
        )

    return _orchestrator_instance

# ============= 数据类定义 =============

class QueryComplexity(Enum):
    """查询复杂度"""
    QUICK_RESPONSE = "quick_response"
    SIMPLE = "simple"
    COMPLEX = "complex"

class ProcessingStrategy(Enum):
    """处理策略"""
    QUICK_RESPONSE = "quick_response"
    STANDARD = "standard"
    FALLBACK = "fallback"
    COMPREHENSIVE = "comprehensive"

@dataclass
class QueryAnalysis:
    """查询分析结果（保持兼容性）"""
    original_query: str
    complexity: QueryComplexity
    is_quick_response: bool
    intent: str
    confidence: float
    api_calls: List[Dict[str, Any]] = field(default_factory=list)
    needs_calculation: bool = False
    calculation_type: Optional[str] = None
    calculation_params: Dict[str, Any] = field(default_factory=dict)
    query_type_info: Optional[QueryTypeResult] = None  # 🆕 新增

# 在 intelligent_qa_orchestrator.py 中更新ProcessingResult
@dataclass
class ProcessingResult:
    """处理结果"""
    session_id: str
    query_id: str
    success: bool
    response_text: str

    # 核心数据
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    calculation_results: Dict[str, Any] = field(default_factory=dict)

    # 🆕 图表相关字段
    charts_data: Dict[str, Any] = field(default_factory=dict)
    has_charts: bool = False
    chart_generation_time: float = 0.0
    chart_generation_method: str = "none"  # ai_intelligent, rule_based, failed, none

    # 处理信息
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    processing_path: str = "standard"
    confidence_score: float = 0.0
    total_processing_time: float = 0.0
    processing_strategy: ProcessingStrategy = ProcessingStrategy.STANDARD

    # 错误信息
    error_info: Optional[Dict[str, Any]] = None

    # 会话信息
    conversation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 兼容旧版API
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    insights: List[Any] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)  # 🆕 这里将包含图表数据
    data_quality_score: float = 0.0
    response_completeness: float = 0.0
    ai_processing_time: float = 0.0
    data_fetching_time: float = 0.0
    processors_used: List[str] = field(default_factory=list)
    ai_collaboration_summary: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

# ============= 主编排器类 =============

class IntelligentQAOrchestrator:
    """精简重构后的智能问答编排器 - 专注于流程编排"""

    def __init__(self,
                 claude_client: Optional[ClaudeClient] = None,
                 gpt_client: Optional[OpenAIClient] = None,
                 db_connector: Optional[DatabaseConnector] = None,
                 app_config: Optional[AppConfig] = None):

        # AI客户端
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.db_connector = db_connector
        self.app_config = app_config or AppConfig()

        # 核心组件
        self.data_fetcher: Optional[SmartDataFetcher] = None
        self.calculator: Optional[UnifiedCalculator] = None
        self.financial_formatter: Optional[FinancialFormatter] = None
        self.conversation_manager: Optional[ConversationManager] = None

        # 🆕 新架构组件
        self.query_type_detector: Optional[QueryTypeDetector] = None
        self.prompt_manager: Optional[PromptManager] = None
        self.strategy_extractor: Optional[EnhancedAPIStrategyExtractor] = None

        # 数据提取组件
        self.semantic_collector = SemanticDataCollector()
        self.claude_extractor = ClaudeIntelligentExtractor(self.claude_client)

        # 配置
        self.config = self._load_config()
        self.current_date = datetime.now()

        # 统计计数器
        self.stats = {
            'total_queries': 0,
            'quick_responses': 0,
            'comprehensive_analyses': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'calculation_calls': 0,
            'claude_generations': 0,
            'processing_time_total': 0.0
        }

        self.initialized = False
        logger.info("精简重构后的智能编排器创建完成")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        return {
            'max_processing_time': 120,
            'quick_response_timeout': 15,
            'claude_timeout': 30,
            'gpt_timeout': 25,
            'enable_dual_judgment': True,
            'current_date': '2025-06-01'
        }

    async def initialize(self):
        """初始化组件"""
        if self.initialized:
            return

        logger.info("初始化精简重构后的智能编排器...")

        try:
            # 🆕 初始化新架构组件
            self.query_type_detector = create_query_type_detector()
            self.prompt_manager = create_prompt_manager()
            self.strategy_extractor = create_enhanced_strategy_extractor(
                self.claude_client, self.query_type_detector, self.prompt_manager
            )

            # 数据获取器
            fetcher_config = {
                'base_url': 'https://api2.3foxzthtdgfy.com',
                'api_key': 'f22bf0ec9c61dce227d8f5d64998e883'
            }
            self.data_fetcher = create_smart_data_fetcher(
                self.claude_client, self.gpt_client, fetcher_config)

            # 统一计算器
            self.calculator = create_unified_calculator(self.gpt_client, precision=6)

            # 格式化器
            self.financial_formatter = create_financial_formatter()

            # 会话管理器
            if self.db_connector:
                self.conversation_manager = create_conversation_manager(self.db_connector)

            # 🆕 添加图表和报告生成器
            from utils.formatters.chart_generator import create_chart_generator, ChartType, ChartTheme
            from utils.formatters.report_generator import create_report_generator, ReportFormat

            self.chart_generator = create_chart_generator(
                theme=ChartTheme.FINANCIAL,
                claude_client=self.claude_client  # 🔧 修复：传递Claude客户端
            )
            self.report_generator = create_report_generator()

            logger.info("图表和报告生成器初始化完成")

            self.initialized = True
            logger.info("智能编排器初始化完成")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise


    # ================================================================
    # 核心方法：智能查询处理
    # ================================================================

    async def process_intelligent_query(self,
                                        user_query: str,
                                        session_id: str = "",
                                        user_id: int = 0,
                                        conversation_id: Optional[str] = None,
                                        preferences: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """智能查询处理 - 精简版"""

        if not self.initialized:
            await self.initialize()

        query_id = f"q_{int(time.time())}_{hash(user_query) % 10000}"
        start_time = time.time()

        logger.info(f"处理查询 [{query_id}]: {user_query}")
        self.stats['total_queries'] += 1

        try:
            # 🆕 阶段1: 使用新架构的双重快速响应判断
            quick_decision = await self._enhanced_quick_response_decision(user_query, query_id)
            print(quick_decision)
            if quick_decision['is_quick_response']:
                logger.info(f"快速响应路径: {quick_decision['reason']}")
                return await self._execute_quick_response_path(
                    user_query, quick_decision, session_id, query_id, start_time, conversation_id)

            # 🆕 阶段2: 使用增强策略提取器进行完整分析
            logger.info("执行完整分析路径")
            query_analysis = await self._enhanced_comprehensive_analysis(user_query, query_id, quick_decision)

            # 阶段3: 数据获取
            raw_data = await self._intelligent_data_acquisition(query_analysis)

            # 阶段4: 三层数据提取
            extracted_data = await self._claude_three_layer_data_extraction(
                raw_data, user_query, query_analysis)

            # 阶段5: 计算处理
            calculation_results = await self._statistical_calculation_processing(
                query_analysis, extracted_data)

            # 🆕 阶段6: 增强响应生成（包含图表生成）
            response_start_time = time.time()
            response_text = await self._enhanced_response_generation(
                user_query, query_analysis, extracted_data, calculation_results)
            response_generation_time = time.time() - response_start_time

            # 🆕 获取缓存的图表数据
            chart_data = getattr(self, '_last_chart_results', {
                'success': False,
                'chart_count': 0,
                'generation_method': 'none',
                'generated_charts': []
            })

            has_charts = chart_data.get('success', False) and chart_data.get('chart_count', 0) > 0
            chart_generation_time = chart_data.get('processing_time', 0.0)

            # 🆕 处理图表数据用于API响应
            processed_charts = []
            if has_charts:
                processed_charts = self._process_chart_results_for_api(chart_data)

            total_time = time.time() - start_time
            self.stats['processing_time_total'] += total_time

            result = ProcessingResult(
                session_id=session_id,
                query_id=query_id,
                success=True,
                response_text=response_text,
                extracted_data=extracted_data,
                calculation_results=calculation_results,

                # 🆕 图表相关字段
                charts_data=chart_data,
                has_charts=has_charts,
                chart_generation_time=chart_generation_time,
                chart_generation_method=chart_data.get('generation_method', 'none'),

                complexity=query_analysis.complexity,
                processing_path="comprehensive",
                confidence_score=query_analysis.confidence,
                total_processing_time=total_time,
                processing_strategy=ProcessingStrategy.COMPREHENSIVE,
                conversation_id=conversation_id,

                # 🆕 更新visualizations字段以包含图表
                visualizations=processed_charts
            )

            # 保存对话记录
            await self._save_conversation_if_needed(conversation_id, user_query, result)

            logger.info(f"查询处理成功 [{query_id}] 耗时: {total_time:.2f}s，图表: {'是' if has_charts else '否'}")
            return result

        except Exception as e:
            logger.error(f"查询处理失败 [{query_id}]: {e}\n{traceback.format_exc()}")
            return self._create_error_result(session_id, query_id, user_query, str(e),
                                             time.time() - start_time, conversation_id)

    def _process_chart_results_for_api(self, chart_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理图表结果用于API响应 - 适配前端ChartJS格式"""

        processed_charts = []
        generated_charts = chart_data.get('generated_charts', [])

        for i, chart in enumerate(generated_charts):
            try:
                chart_type = chart.get('chart_type', 'unknown')
                chart_success = chart.get('success', False)

                if not chart_success:
                    continue

                # 🎯 转换为前端ChartJS期望的格式
                processed_chart = {
                    'type': 'chart',  # 前端期望的类型标识
                    'title': chart.get('title', f'图表 {i + 1}'),
                    'data': self._convert_to_chartjs_format(chart)
                }

                processed_charts.append(processed_chart)

            except Exception as e:
                logger.error(f"处理图表 {i + 1} 时出错: {e}")
                continue

        return processed_charts

    def _convert_to_chartjs_format(self, chart: Dict[str, Any]) -> Dict[str, Any]:
        """将后端图表数据转换为前端ChartJS格式"""

        chart_type = chart.get('chart_type', 'pie')
        title = chart.get('title', '数据分析')
        data_summary = chart.get('data_summary', {})

        # 🎯 根据图表类型转换数据格式
        if chart_type in ['pie', 'donut']:
            return self._convert_pie_chart_to_chartjs(title, data_summary, chart_type)
        elif chart_type == 'bar':
            return self._convert_bar_chart_to_chartjs(title, data_summary)
        elif chart_type == 'line':
            return self._convert_line_chart_to_chartjs(title, data_summary)
        else:
            # 默认转换为饼图
            return self._convert_pie_chart_to_chartjs(title, data_summary, 'pie')

    def _convert_pie_chart_to_chartjs(self, title: str, data_summary: Dict[str, Any], chart_type: str) -> Dict[
        str, Any]:
        """转换饼图数据为ChartJS格式"""

        labels = data_summary.get('labels', [])
        values = data_summary.get('values', [])

        if not labels or not values or len(labels) != len(values):
            # 如果数据不完整，使用默认数据
            labels = ['数据1', '数据2']
            values = [50, 50]

        # 🎯 生成ChartJS期望的颜色
        colors = [
            '#0072B2',  # 蓝色
            '#E69F00',  # 橙色
            '#009E73',  # 绿色
            '#CC79A7',  # 粉色
            '#56B4E9',  # 天蓝色
            '#F0E442',  # 黄色
            '#D55E00',  # 红橙色
            '#999999',  # 灰色
        ]

        # 确保有足够的颜色
        chart_colors = []
        for i in range(len(values)):
            chart_colors.append(colors[i % len(colors)])

        chartjs_data = {
            'type': 'doughnut' if chart_type == 'donut' else 'pie',
            'title': title,
            'labels': labels,
            'datasets': [{
                'label': title,
                'data': values,
                'backgroundColor': chart_colors,
                'borderColor': chart_colors,
                'borderWidth': 2,
                'hoverOffset': 4
            }],
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {
                        'position': 'bottom',
                        'labels': {
                            'padding': 20,
                            'usePointStyle': True
                        }
                    },
                    'title': {
                        'display': True,
                        'text': title,
                        'font': {
                            'size': 16,
                            'weight': 'bold'
                        },
                        'padding': {
                            'bottom': 20
                        }
                    },
                    'tooltip': {
                        'callbacks': {
                            'label': f'function(context) {{ return context.label + ": ¥" + context.parsed.toLocaleString() + " (" + Math.round(context.parsed / context.dataset.data.reduce((a, b) => a + b, 0) * 100) + "%)"; }}'
                        }
                    }
                }
            }
        }

        return chartjs_data

    def _convert_bar_chart_to_chartjs(self, title: str, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """转换柱状图数据为ChartJS格式 - 优化版"""

        categories = data_summary.get('categories', [])
        series = data_summary.get('series', {})

        if not categories or not series:
            categories = ['类别1', '类别2', '类别3']
            series = {'数据系列': [100, 200, 150]}

        datasets = []
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00']

        for i, (series_name, series_data) in enumerate(series.items()):
            datasets.append({
                'label': series_name,
                'data': series_data,
                'backgroundColor': colors[i % len(colors)],
                'borderColor': colors[i % len(colors)],
                'borderWidth': 1,
                'borderRadius': 4,  # 🆕 圆角
                'borderSkipped': False,
            })

        return {
            'type': 'bar',
            'title': title,
            'labels': categories,
            'datasets': datasets,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {
                        'position': 'top',
                        'labels': {
                            'usePointStyle': True,  # 🆕 使用点样式
                            'padding': 20
                        }
                    },
                    'title': {
                        'display': True,
                        'text': title,
                        'font': {
                            'size': 16,
                            'weight': 'bold'
                        }
                    },
                    'tooltip': {
                        'mode': 'index',
                        'intersect': False,
                        'callbacks': {
                            'label': 'function(context) { return context.dataset.label + ": ¥" + context.parsed.y.toLocaleString(); }'
                        }
                    }
                },
                'scales': {
                    'x': {
                        'grid': {
                            'display': False  # 🆕 隐藏X轴网格线
                        }
                    },
                    'y': {
                        'beginAtZero': True,
                        'grid': {
                            'color': 'rgba(0, 0, 0, 0.1)'  # 🆕 淡化网格线
                        },
                        'ticks': {
                            'callback': 'function(value) { return "¥" + value.toLocaleString(); }'  # 🆕 格式化Y轴标签
                        }
                    }
                },
                'interaction': {
                    'mode': 'nearest',
                    'axis': 'x',
                    'intersect': False
                }
            }
        }

    def _convert_line_chart_to_chartjs(self, title: str, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """转换折线图数据为ChartJS格式"""

        x_axis = data_summary.get('x_axis', [])
        series = data_summary.get('series', {})

        if not x_axis or not series:
            x_axis = ['时间1', '时间2', '时间3']
            series = {'数据系列': [10, 20, 30]}

        datasets = []
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

        for i, (series_name, series_data) in enumerate(series.items()):
            datasets.append({
                'label': series_name,
                'data': series_data,
                'borderColor': colors[i % len(colors)],
                'backgroundColor': colors[i % len(colors)] + '20',  # 20% 透明度
                'borderWidth': 2,
                'fill': False,
                'tension': 0.1
            })

        return {
            'type': 'line',
            'title': title,
            'labels': x_axis,
            'datasets': datasets,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {
                        'position': 'top'
                    },
                    'title': {
                        'display': True,
                        'text': title
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            }
        }

    def _generate_chart_description(self, chart: Dict[str, Any]) -> str:
        """为图表生成描述文字"""

        chart_type = chart.get('chart_type', '图表')
        title = chart.get('title', '数据分析')
        data_summary = chart.get('data_summary', {})

        # 根据图表类型生成不同的描述
        if chart_type in ['pie', 'donut']:
            if 'values' in data_summary and 'labels' in data_summary:
                total = sum(data_summary['values']) if data_summary['values'] else 0
                if total > 0 and data_summary['values']:
                    max_item_idx = data_summary['values'].index(max(data_summary['values']))
                    max_item_label = data_summary['labels'][max_item_idx] if data_summary['labels'] else '未知'
                    formatted_total = self.financial_formatter.format_currency(
                        total) if self.financial_formatter else f'{total:,.2f}'
                    return f"{title}：总额 {formatted_total}，其中 {max_item_label} 占比最大"

        elif chart_type == 'bar':
            if 'series' in data_summary:
                series_count = len(data_summary['series'])
                return f"{title}：包含 {series_count} 个数据系列的对比分析"

        elif chart_type == 'line':
            if 'series' in data_summary:
                series_count = len(data_summary['series'])
                return f"{title}：{series_count} 个指标的趋势变化图"

        return f"{title}：{chart_type}图表"
    # ================================================================
    # 🆕 增强的阶段1：双重快速响应判断
    # ================================================================

    async def _enhanced_quick_response_decision(self, user_query: str, query_id: str) -> Dict[str, Any]:
        """增强版双重快速响应判断 - 修复聚合查询"""
        logger.debug(f"执行增强版快速响应判断: {user_query}")

        try:
            # 🆕 步骤0: 预检查明显的复合查询标志
            obvious_complex_keywords = ['平均', '月份', '对比', '趋势', '历史', '合计', '统计']
            if any(keyword in user_query for keyword in obvious_complex_keywords):
                logger.info(f"检测到明显的复合查询关键词，直接判断为复杂查询")
                return {
                    'is_quick_response': False,
                    'reason': f'检测到复合查询关键词: {[kw for kw in obvious_complex_keywords if kw in user_query]}',
                    'confidence': 0.9,
                    'query_type_info': None
                }

            # 步骤1: QueryTypeDetector检测
            query_type_result = self.query_type_detector.detect(user_query)

            # 🆕 步骤1.5: 特殊类型直接判断为复杂
            if query_type_result.type in [QueryType.AGGREGATION, QueryType.HISTORICAL_REVIEW,
                                          QueryType.COMPARISON, QueryType.REINVESTMENT]:
                return {
                    'is_quick_response': False,
                    'reason': f'检测到复杂查询类型: {query_type_result.type.value}',
                    'confidence': query_type_result.confidence,
                    'query_type_info': query_type_result
                }

            # 步骤2: Claude快速分析
            claude_judgment = await self._claude_quick_analysis(user_query, query_type_result)

            # 步骤3: 🆕 GPT交叉验证（对于边界情况）
            if claude_judgment.get('confidence', 0) < 0.8:
                gpt_judgment = await self._gpt_cross_validation(user_query, claude_judgment)

                # 综合判断
                final_decision = self._combine_ai_judgments(claude_judgment, gpt_judgment)
            else:
                final_decision = claude_judgment

            return {
                'is_quick_response': final_decision.get('is_quick', False),
                'reason': final_decision.get('reason', '基于AI双重验证'),
                'confidence': final_decision.get('confidence', 0.7),
                'query_type_info': query_type_result,
                'ai_analysis': final_decision  # 保存完整分析
            }

        except Exception as e:
            logger.error(f"双重验证失败: {e}")
            return {
                'is_quick_response': False,
                'reason': f'验证过程异常，使用保守策略: {str(e)}',
                'confidence': 0.0
            }

    async def _claude_quick_analysis(self, user_query: str, query_type_result) -> Dict[str, Any]:
        """Claude快速分析"""
        try:
            prompt = f"""
                   分析这个查询是否为复合查询（需要多个API调用）：

                   查询: "{user_query}"

                   🔍 **重要判断标准**：

                   **简单查询（单次API调用）**：
                   - "今天入金多少" → get_daily_data(今天)
                   - "总余额是多少" → get_system_data()
                   - "昨天注册人数" → get_daily_data(昨天)

                   **复合查询（多次API调用）**：
                   - "昨天的出金和今天的入金" → 需要2次API调用
                   - "5月份每日平均入金" → 需要31次API调用（5月1日-31日）+ 计算
                   - "本周和上周对比" → 需要14次API调用
                   - "最近7天的数据" → 需要7次API调用

                   **关键识别点**：
                   1. **时间范围**：月份/周/多天 = 复合查询
                   2. **计算需求**：平均/合计/对比 = 复合查询  
                   3. **数据聚合**：需要多个数据点的分析 = 复合查询

                   特别注意：
                   - "X月份" → 需要整月数据，必须是复合查询
                   - "平均" → 需要多个数据点计算，必须是复合查询
                   - "每日平均" → 需要多天数据，必须是复合查询

                   返回JSON：
                   {{
                       "is_quick": false,
                       "query_type": "aggregation",
                       "api_calls_needed": 31,
                       "time_points": ["2025-05-01", "2025-05-02", "...", "2025-05-31"], 
                       "requires_calculation": true,
                       "confidence": 0.95,
                       "reason": "需要获取5月份31天的每日数据并计算平均值"
                   }}
                   """

            result = await self.claude_client.generate_text(prompt, max_tokens=1000)

            if result.get('success'):
                response_text = result.get('text', '{}')

                # 解析JSON响应
                import re, json
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    claude_analysis = json.loads(json_match.group())
                    return claude_analysis
                else:
                    logger.warning("Claude响应无法解析为JSON")
                    return {
                        'is_quick': True,  # 解析失败时保守处理
                        'confidence': 0.5,
                        'reason': 'Claude响应解析失败'
                    }
            else:
                logger.error(f"Claude分析失败: {result.get('error')}")
                return {
                    'is_quick': True,
                    'confidence': 0.3,
                    'reason': f'Claude调用失败: {result.get("error")}'
                }

        except Exception as e:
            logger.error(f"Claude快速分析异常: {e}")
            return {
                'is_quick': True,
                'confidence': 0.2,
                'reason': f'Claude分析异常: {str(e)}'
            }

    async def _gpt_cross_validation(self, user_query: str, claude_judgment: Dict) -> Dict[str, Any]:
        """GPT交叉验证 - 使用现有方法"""
        try:
            # 构建验证查询
            validation_query = f"""
            请验证这个查询分析是否正确：

            用户查询: "{user_query}"
            Claude分析: {claude_judgment}

            请重新分析并给出你的判断。特别关注：
            1. 是否涉及多个时间点？
            2. 是否需要多次API调用？
            3. 复杂度如何评估？

            返回JSON格式的分析结果。
            """

            # 使用现有的 process_direct_query 方法
            result = await self.gpt_client.process_direct_query(validation_query, claude_judgment)

            if result.get('success'):
                response_text = result.get('response', '{}')

                # 解析JSON响应
                import re, json
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    gpt_analysis = json.loads(json_match.group())

                    # 确保返回格式一致
                    return {
                        'is_quick': gpt_analysis.get('is_quick', False),
                        'confidence': gpt_analysis.get('confidence', 0.5),
                        'reason': gpt_analysis.get('reason', 'GPT验证结果'),
                        'agreement_with_claude': gpt_analysis.get('agreement_with_claude', False)
                    }
                else:
                    logger.warning("GPT响应无法解析为JSON")
                    return {
                        'is_quick': False,
                        'confidence': 0.5,
                        'reason': 'GPT响应解析失败',
                        'agreement_with_claude': False
                    }
            else:
                logger.error(f"GPT验证失败: {result.get('error')}")
                return {
                    'is_quick': False,
                    'confidence': 0.3,
                    'reason': f'GPT调用失败: {result.get("error")}',
                    'agreement_with_claude': False
                }

        except Exception as e:
            logger.error(f"GPT交叉验证异常: {e}")
            return {
                'is_quick': False,
                'confidence': 0.2,
                'reason': f'GPT验证异常: {str(e)}',
                'agreement_with_claude': False
            }

    def _combine_ai_judgments(self, claude_result: Dict, gpt_result: Dict) -> Dict[str, Any]:
        """综合AI判断结果"""
        # 如果两个AI都认为是复合查询，则确认为复合
        claude_is_quick = claude_result.get('is_quick', True)
        gpt_is_quick = gpt_result.get('is_quick', True)

        if not claude_is_quick and not gpt_is_quick:
            return {
                'is_quick': False,
                'confidence': min(claude_result.get('confidence', 0.7), gpt_result.get('confidence', 0.7)),
                'reason': f"双AI确认为复合查询"
            }
        elif claude_is_quick and gpt_is_quick:
            return {
                'is_quick': True,
                'confidence': (claude_result.get('confidence', 0.7) + gpt_result.get('confidence', 0.7)) / 2,
                'reason': "双AI确认为简单查询"
            }
        else:
            # 意见不一致，采用保守策略
            return {
                'is_quick': False,
                'confidence': 0.6,
                'reason': "AI意见不一致，采用完整分析确保准确性"
            }

    # ================================================================
    # 🆕 增强的阶段2：完整分析路径
    # ================================================================

    async def _enhanced_comprehensive_analysis(self, user_query: str, query_id: str,
                                               quick_decision: Dict[str, Any] = None) -> QueryAnalysis:
        """增强版完整查询分析 - 修复日期覆盖问题"""
        logger.debug(f"执行增强版完整查询分析: {user_query}")

        try:
            # 🆕 使用增强版策略提取器
            context = {'quick_decision': quick_decision} if quick_decision else {}
            extraction_result = await self.strategy_extractor.extract_strategy(user_query, context)

            if not extraction_result.success:
                logger.warning(f"策略提取失败: {extraction_result.error_message}")
                return self._create_fallback_query_analysis(user_query, query_id)

            # 🆕 转换为QueryAnalysis格式（保持兼容性）
            query_analysis = self._convert_extraction_to_query_analysis(extraction_result, user_query)

            # 🔧 修复：检查是否是对比查询，如果是则跳过日期增强
            is_comparison_query = (
                    hasattr(query_analysis, 'query_type_info') and
                    query_analysis.query_type_info and
                    query_analysis.query_type_info.type.value == 'comparison'
            )

            # 🔧 修复：检查API调用是否已经包含完整的日期范围
            has_complete_date_range = self._check_api_calls_have_complete_dates(query_analysis.api_calls)

            logger.info(f"🔍 [DEBUG] 是否为对比查询: {is_comparison_query}")
            logger.info(f"🔍 [DEBUG] API调用是否已有完整日期: {has_complete_date_range}")
            logger.info(f"🔍 [DEBUG] API调用数量: {len(query_analysis.api_calls)}")

            # 🔧 修复：只在必要时进行日期识别增强
            if (query_analysis.query_type_info and
                    not is_comparison_query and
                    not has_complete_date_range):

                logger.info("执行日期识别增强...")
                date_analysis = await self._enhanced_date_recognition(user_query, query_analysis.query_type_info)
                if date_analysis.get('success'):
                    # 更新API调用中的日期参数
                    query_analysis.api_calls = self._update_api_dates_with_enhanced_result(
                        query_analysis.api_calls, date_analysis.get('date_info', {}))
            else:
                logger.info("跳过日期识别增强（已有完整日期或为对比查询）")

            return query_analysis

        except Exception as e:
            logger.error(f"增强版完整查询分析失败: {e}")
            return self._create_fallback_query_analysis(user_query, query_id)

    def _check_api_calls_have_complete_dates(self, api_calls: List[Dict[str, Any]]) -> bool:
        """检查API调用是否已经包含完整的日期配置"""
        if not api_calls:
            return False

        # 检查是否有多个不同的日期
        dates = set()
        for call in api_calls:
            params = call.get('params', {})
            date = params.get('date')
            if date:
                dates.add(date)

        # 如果有多个不同日期（比如周对比的14个日期），认为已经完整
        has_multiple_dates = len(dates) > 1

        # 检查是否有time_period标识（说明是有组织的时间序列查询）
        has_time_periods = any(call.get('time_period') for call in api_calls)

        logger.info(f"🔍 [DEBUG] 发现 {len(dates)} 个不同日期: {list(dates)[:5]}...")  # 只显示前5个
        logger.info(f"🔍 [DEBUG] 是否有时间周期标识: {has_time_periods}")

        return has_multiple_dates and has_time_periods
    async def _enhanced_date_recognition(self, user_query: str,
                                       query_type_result: QueryTypeResult) -> Dict[str, Any]:
        """增强版日期识别 - 使用PromptManager"""
        try:
            # 🆕 使用PromptManager构建日期识别prompt
            prompt = self.prompt_manager.build_date_recognition_prompt(user_query, query_type_result)

            result = await self.claude_client.generate_text(prompt, max_tokens=5000)

            if result.get('success'):
                response_text = result.get('text', '{}')
                import re, json
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    date_info = json.loads(json_match.group())
                    return {'success': True, 'date_info': date_info}

            return {'success': False, 'error': '日期识别解析失败'}

        except Exception as e:
            logger.error(f"增强版日期识别失败: {e}")
            return {'success': False, 'error': str(e)}

    def _convert_extraction_to_query_analysis(self, extraction_result: ExtractedStrategy,
                                              user_query: str) -> QueryAnalysis:
        """将ExtractedStrategy转换为QueryAnalysis（兼容性）- 修复计算类型"""

        query_analysis_data = extraction_result.query_analysis

        # 确定复杂度
        complexity_str = query_analysis_data.get('complexity', 'simple')
        if complexity_str == 'complex':
            complexity = QueryComplexity.COMPLEX
        elif complexity_str == 'medium':
            complexity = QueryComplexity.SIMPLE
        else:
            complexity = QueryComplexity.SIMPLE

        # 🔧 修复：正确设置计算类型和计算需求
        needs_calculation = query_analysis_data.get('calculation_required', False)
        calculation_type = None

        # 🆕 检查用户查询关键词来确定计算类型
        query_lower = user_query.lower()
        if '平均' in query_lower or '均值' in query_lower:
            calculation_type = "statistics"
            needs_calculation = True
        elif '对比' in query_lower or '比较' in query_lower:
            calculation_type = "comparison_analysis"
            needs_calculation = True
        elif '复投' in query_lower:
            calculation_type = "reinvestment_analysis"
            needs_calculation = True
        elif '预测' in query_lower or '还能运行' in query_lower:
            calculation_type = "cash_runway"
            needs_calculation = True

        # 🆕 从查询类型结果中获取
        if extraction_result.query_type_info:
            if extraction_result.query_type_info.type == QueryType.AGGREGATION:
                calculation_type = "statistics"
                needs_calculation = True
            special_calc_type = extraction_result.query_type_info.special_requirements.get('calculation_type')
            if special_calc_type:
                calculation_type = special_calc_type
                needs_calculation = True

        # 🔧 添加调试日志
        logger.info(f"🔍 [CALC_DEBUG] 计算类型设置: {calculation_type}, 需要计算: {needs_calculation}")

        return QueryAnalysis(
            original_query=user_query,
            complexity=complexity,
            is_quick_response=False,
            intent=query_analysis_data.get('intent', '数据查询'),
            confidence=extraction_result.confidence,
            api_calls=extraction_result.api_calls,
            needs_calculation=needs_calculation,
            calculation_type=calculation_type,
            query_type_info=extraction_result.query_type_info
        )

    def _update_api_dates_with_enhanced_result(self, api_calls: List[Dict[str, Any]],
                                               date_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用增强的日期识别结果更新API调用 - 修复版本"""

        # 🔧 修复：如果API调用已经有完整日期，不要覆盖
        if self._check_api_calls_have_complete_dates(api_calls):
            logger.info("🔍 [DEBUG] API调用已有完整日期配置，跳过日期更新")
            return api_calls

        if not date_info.get('has_dates', False):
            logger.info("🔍 [DEBUG] 日期分析未发现日期，保持原API调用")
            return api_calls

        updated_calls = []
        api_dates = date_info.get('api_dates', {})
        start_date = api_dates.get('start_date')
        end_date = api_dates.get('end_date')

        logger.info(f"🔍 [DEBUG] 应用日期更新: start_date={start_date}, end_date={end_date}")

        for call in api_calls:
            updated_call = call.copy()
            method = call.get('method', '')
            params = call.get('params', {}).copy()

            # 为需要日期的API添加日期参数（只有在原来没有日期时）
            if not params.get('date'):  # 🔧 修复：只在没有日期时才添加
                if ('daily' in method or 'day' in method) and start_date:
                    params['date'] = start_date
                    logger.info(f"🔍 [DEBUG] 为 {method} 添加日期: {start_date}")
                elif 'product_end_interval' in method and start_date and end_date:
                    params['start_date'] = start_date
                    params['end_date'] = end_date
                    logger.info(f"🔍 [DEBUG] 为 {method} 添加日期范围: {start_date} - {end_date}")
                elif 'product_end_data' in method and start_date:
                    params['date'] = start_date
                    logger.info(f"🔍 [DEBUG] 为 {method} 添加日期: {start_date}")

            updated_call['params'] = params
            updated_calls.append(updated_call)

        return updated_calls

    # ================================================================
    # 快速响应路径（保持原有逻辑，使用新组件优化）
    # ================================================================

    async def _execute_quick_response_path(self,
                                           user_query: str,
                                           quick_decision: Dict[str, Any],
                                           session_id: str,
                                           query_id: str,
                                           start_time: float,
                                           conversation_id: Optional[str]) -> ProcessingResult:
        """执行快速响应路径"""

        self.stats['quick_responses'] += 1

        try:
            # 从快速决策中获取查询类型信息
            query_type_info = quick_decision.get('query_type_info')

            # 提取API调用信息
            api_info = self._extract_quick_api_info_enhanced(quick_decision, user_query, query_type_info)

            # 执行API调用
            raw_data = await self._execute_quick_api_call(api_info)
            print(raw_data)
            # 快速数据提取
            extracted_data = await self._quick_data_extraction(raw_data, user_query)

            # 🆕 使用PromptManager生成快速响应
            response_text = await self._quick_response_generation_enhanced(user_query, extracted_data, query_type_info)

            total_time = time.time() - start_time

            result = ProcessingResult(
                session_id=session_id,
                query_id=query_id,
                success=True,
                response_text=response_text,
                extracted_data=extracted_data,
                complexity=QueryComplexity.QUICK_RESPONSE,
                processing_path="quick_response",
                confidence_score=quick_decision.get('confidence', 0.9),
                total_processing_time=total_time,
                processing_strategy=ProcessingStrategy.QUICK_RESPONSE,
                conversation_id=conversation_id
            )

            await self._save_conversation_if_needed(conversation_id, user_query, result)

            logger.info(f"快速响应完成 [{query_id}] 耗时: {total_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"快速响应路径失败: {e}")
            logger.info("降级到完整分析路径")
            return await self._execute_fallback_to_comprehensive(
                user_query, session_id, query_id, start_time, conversation_id)

    def _extract_quick_api_info_enhanced(self, quick_decision: Dict[str, Any],
                                       user_query: str,
                                       query_type_info: Optional[QueryTypeResult]) -> Dict[str, Any]:
        """增强版快速API信息提取"""

        # 🆕 如果有查询类型信息，优先使用
        if query_type_info and query_type_info.special_requirements:
            required_apis = query_type_info.special_requirements.get('requires_apis', [])
            if required_apis:
                return {
                    'method': required_apis[0],  # 使用第一个推荐的API
                    'params': {}
                }

        # 原有的关键词匹配逻辑作为降级
        return self._extract_quick_api_info_legacy(user_query)

    def _extract_quick_api_info_legacy(self, user_query: str) -> Dict[str, Any]:
        """原有的快速API信息提取逻辑"""
        query_lower = user_query.lower()

        # 日期关键词检测
        if ('昨天' in query_lower or '昨日' in query_lower) and any(
                kw in query_lower for kw in ['入金', '出金', '注册']):
            from datetime import timedelta
            yesterday = (self.current_date - timedelta(days=1)).strftime('%Y%m%d')
            return {'method': 'get_daily_data', 'params': {'date': yesterday}}

        elif ('今天' in query_lower or '今日' in query_lower) and any(
                kw in query_lower for kw in ['入金', '出金', '注册']):
            today = self.current_date.strftime('%Y%m%d')
            return {'method': 'get_daily_data', 'params': {'date': today}}

        else:
            return {'method': 'get_system_data', 'params': {}}

    async def _quick_response_generation_enhanced(self, user_query: str,
                                                extracted_data: Dict[str, Any],
                                                query_type_info: Optional[QueryTypeResult]) -> str:
        """增强版快速响应生成"""

        if not self.claude_client:
            return self._basic_response_generation(user_query, extracted_data)

        try:
            # 🆕 构建快速响应的简化prompt
            key_metrics = extracted_data.get('key_metrics', {})

            prompt = f"""
            用户查询: "{user_query}"
            查询类型: {query_type_info.type.value if query_type_info else '未知'}
            
            提取的关键数据:
            {json.dumps(key_metrics, ensure_ascii=False, indent=2)}
            
            请生成简洁、准确、易懂的回答：
            1. 直接回答用户问题
            2. 使用具体数字
            3. 格式友好易读
            4. 避免冗余信息
            
            示例风格:
            💰 总余额：¥8,223,695.07
            👥 活跃用户：3,911人
            """

            result = await self.claude_client.generate_text(prompt, max_tokens=5000)

            if result.get('success'):
                return result.get('text', '').strip()

            return self._basic_response_generation(user_query, extracted_data)

        except Exception as e:
            logger.error(f"增强版快速响应生成失败: {e}")
            return self._basic_response_generation(user_query, extracted_data)

    # ================================================================
    # 数据获取（保持原有逻辑）
    # ================================================================

    async def _intelligent_data_acquisition(self, query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """
        智能数据获取 - 完整版本（包含调试、数据缓存和事件循环修复）

        Args:
            query_analysis: 查询分析结果

        Returns:
            Dict[str, Any]: 包含API调用结果的字典
        """
        logger.info("🚀 开始执行智能数据获取")

        # 🔍 调试：打印API调用详情
        logger.info(f"🔍 [DEBUG] API调用总数: {len(query_analysis.api_calls)}")
        for i, api_call in enumerate(query_analysis.api_calls[:10]):  # 只打印前10个
            method = api_call.get('method', 'unknown')
            params = api_call.get('params', {})
            time_period = api_call.get('time_period', '')
            reason = api_call.get('reason', '')

            logger.info(f"🔍 [DEBUG] API调用 {i + 1}: method={method}, params={params}, time_period={time_period}")
            logger.info(f"🔍 [DEBUG]   原因: {reason}")

        if not self.data_fetcher or not self.data_fetcher.api_connector:
            logger.error("❌ 数据获取器不可用")
            raise Exception("数据获取器不可用")

        api_connector = self.data_fetcher.api_connector

        # 🔧 新增：API连接器健康检查
        try:
            health_check = await api_connector.health_check()
            logger.info(f"🔍 [DEBUG] API连接器健康状态: {health_check.get('status')}")
            if health_check.get('status') != 'healthy':
                logger.warning(f"⚠️ API连接器状态异常: {health_check.get('status')}")
        except Exception as e:
            logger.warning(f"⚠️ API连接器健康检查失败: {e}")

        results = {}
        start_time = time.time()
        failed_calls = []  # 🔧 新增：跟踪失败的调用
        consecutive_failures = 0  # 🔧 新增：连续失败计数
        max_consecutive_failures = 5  # 🔧 新增：最大连续失败数

        try:
            # 执行API调用策略
            for i, api_call in enumerate(query_analysis.api_calls):
                method = api_call.get('method', 'get_system_data')
                params = api_call.get('params', {})
                reason = api_call.get('reason', '数据获取')
                time_period = api_call.get('time_period', '')

                logger.info(f"🔄 执行API调用 {i + 1}/{len(query_analysis.api_calls)}")
                logger.info(f"  - 方法: {method}, 参数: {params}")
                logger.info(f"🔍 [DEBUG]   - 时间周期: {time_period}, 原因: {reason}")

                result = None

                # 🔧 新增：增加重试机制来处理事件循环问题
                max_method_retries = 3
                retry_delay_base = 0.5

                for retry in range(max_method_retries):
                    try:
                        # 🔧 新增：在重试前检查连续失败数
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"💥 连续失败次数过多 ({consecutive_failures})，跳过剩余API调用")
                            result = {
                                'success': False,
                                'message': f'连续失败次数过多，跳过API调用: {method}'
                            }
                            break

                        # 路由到对应的API方法
                        if method == 'get_system_data':
                            result = await api_connector.get_system_data()

                        elif method == 'get_daily_data':
                            date_param = params.get('date')
                            logger.info(f"🔍 [DEBUG] 调用get_daily_data，日期参数: {date_param}")
                            if date_param:
                                result = await api_connector.get_daily_data(date_param)
                            else:
                                result = await api_connector.get_daily_data(self.current_date.strftime('%Y%m%d'))

                        elif method == 'get_product_data':
                            result = await api_connector.get_product_data()

                        elif method == 'get_product_end_data':
                            date_param = params.get('date')
                            if date_param:
                                result = await api_connector.get_product_end_data(date_param)
                            else:
                                logger.warning(
                                    f"get_product_end_data缺少日期参数，使用今日: {self.current_date.strftime('%Y%m%d')}")
                                result = await api_connector.get_product_end_data(self.current_date.strftime('%Y%m%d'))

                        elif method == 'get_product_end_interval':
                            start_date = params.get('start_date')
                            end_date = params.get('end_date')
                            if start_date and end_date:
                                result = await api_connector.get_product_end_interval(start_date, end_date)
                            else:
                                logger.error(
                                    f"get_product_end_interval缺少日期范围参数: start_date={start_date}, end_date={end_date}")
                                result = {'success': False, 'message': '缺少日期范围参数'}

                        elif method == 'get_user_daily_data':
                            date_param = params.get('date')
                            if date_param:
                                result = await api_connector.get_user_daily_data(date_param)
                            else:
                                result = await api_connector.get_user_daily_data()

                        elif method == 'get_user_data':
                            page = params.get('page', 1)
                            result = await api_connector.get_user_data(page)

                        else:
                            logger.warning(f"⚠️ 未知的API方法: {method}")
                            result = {'success': False, 'message': f'未支持的API方法: {method}'}
                            break

                        # 🔍 调试：检查API调用结果
                        if result:
                            success = result.get('success', False)
                            logger.info(f"🔍 [DEBUG] API调用结果: success={success}")

                            if success:
                                # 🔧 重置连续失败计数
                                consecutive_failures = 0
                                data_keys = list(result.get('data', {}).keys()) if isinstance(result.get('data'),
                                                                                              dict) else 'non-dict'
                                logger.info(f"🔍 [DEBUG] 返回数据键: {data_keys}")
                                break  # 成功，跳出重试循环
                            else:
                                error_msg = result.get('message', '未知错误')
                                logger.warning(f"⚠️ API调用失败: {error_msg}")

                                # 🔧 检查是否是事件循环问题
                                if "Event loop" in error_msg:
                                    consecutive_failures += 1
                                    if retry < max_method_retries - 1:
                                        retry_delay = retry_delay_base * (2 ** retry)
                                        logger.warning(
                                            f"⚠️ 事件循环问题，等待 {retry_delay}s 后重试 {retry + 1}/{max_method_retries}")
                                        try:
                                            await asyncio.sleep(retry_delay)
                                        except asyncio.CancelledError:
                                            logger.warning("重试等待被取消")
                                            break
                                        continue
                                    else:
                                        logger.error(f"💥 事件循环问题重试失败，已达最大重试次数")
                                        break
                                else:
                                    # 其他类型的错误，直接跳出重试循环
                                    consecutive_failures += 1
                                    break
                        else:
                            logger.error(f"❌ API调用返回None: {method}")
                            consecutive_failures += 1
                            result = {'success': False, 'message': 'API调用返回None'}
                            break

                    except asyncio.CancelledError:
                        logger.warning(f"API调用被取消: {method}")
                        result = {'success': False, 'message': 'API调用被取消'}
                        break

                    except Exception as e:
                        error_str = str(e)
                        logger.error(f"💥 API方法调用异常: {error_str}")

                        # 🔧 检查是否是事件循环相关异常
                        if (
                                "Event loop" in error_str or "loop" in error_str.lower()) and retry < max_method_retries - 1:
                            consecutive_failures += 1
                            retry_delay = retry_delay_base * (2 ** retry)
                            logger.warning(
                                f"⚠️ 事件循环异常，等待 {retry_delay}s 后重试 {retry + 1}/{max_method_retries}: {e}")
                            try:
                                await asyncio.sleep(retry_delay)
                            except asyncio.CancelledError:
                                logger.warning("异常重试等待被取消")
                                break
                            continue
                        else:
                            consecutive_failures += 1
                            result = {'success': False, 'message': f'方法调用异常: {error_str}'}
                            break

                # 记录失败的调用
                if result and not result.get('success', False):
                    failed_call_info = {
                        'method': method,
                        'params': params,
                        'error': result.get('message', '未知错误'),
                        'time_period': time_period,
                        'retry_count': max_method_retries
                    }
                    failed_calls.append(failed_call_info)

                # 存储结果时使用更详细的键名
                if time_period:
                    date_suffix = params.get('date', params.get('start_date', 'no_date'))
                    result_key = f"{time_period}_{method}_{date_suffix}"
                else:
                    result_key = f"{method}_{i}"

                logger.info(f"🔍 [DEBUG] 结果存储键: {result_key}")
                results[result_key] = result

                # 🔧 如果连续失败太多，考虑提前终止
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"💥 连续失败次数过多 ({consecutive_failures})，提前终止API调用")
                    break

                # 🔧 如果总失败率过高，也提前终止
                total_processed = i + 1
                failed_count = len(failed_calls)
                if total_processed >= 5 and failed_count / total_processed > 0.7:  # 超过70%失败
                    logger.error(
                        f"💥 失败率过高 ({failed_count}/{total_processed} = {failed_count / total_processed:.1%})，提前终止")
                    break

                # 短暂延迟，避免API过载和给事件循环恢复时间
                if i < len(query_analysis.api_calls) - 1:
                    try:
                        await asyncio.sleep(0.2)  # 增加延迟时间
                    except asyncio.CancelledError:
                        logger.warning("API调用间隔等待被取消")
                        break

            # 🔍 调试：汇总API调用结果
            total_calls = len(query_analysis.api_calls)
            successful_calls = sum(1 for result in results.values() if result.get('success', False))
            failed_calls_count = len(failed_calls)
            actual_calls = len(results)

            logger.info(
                f"📊 API调用汇总: 计划={total_calls}, 实际={actual_calls}, 成功={successful_calls}, 失败={failed_calls_count}")

            if failed_calls:
                logger.warning(f"⚠️ 失败的API调用总数: {len(failed_calls)}")
                # 只显示前5个失败的详情，避免日志过长
                for idx, failed_call in enumerate(failed_calls[:5]):
                    logger.warning(f"⚠️ 失败 {idx + 1}: {failed_call['method']} - {failed_call['error']}")
                if len(failed_calls) > 5:
                    logger.warning(f"⚠️ 还有 {len(failed_calls) - 5} 个失败调用未显示...")

            # 🆕 缓存原始结果供计算器使用
            if results:
                self._last_api_results = results
                logger.info(f"🔍 [DEBUG] 缓存API结果，共{len(results)}个结果供后续使用")
            else:
                logger.warning("⚠️ 没有成功的API结果可缓存")

            # 🔧 改进成功判断逻辑
            is_success = successful_calls > 0 and successful_calls >= total_calls * 0.3  # 至少30%成功率

            final_result = {
                'success': is_success,
                'results': results,
                'api_calls_executed': actual_calls,
                'failed_calls': failed_calls,
                'execution_summary': {
                    'total_planned_calls': total_calls,
                    'actual_calls': actual_calls,
                    'successful_calls': successful_calls,
                    'failed_calls': failed_calls_count,
                    'success_rate': successful_calls / actual_calls if actual_calls > 0 else 0,
                    'completion_rate': actual_calls / total_calls if total_calls > 0 else 0,
                    'processing_time': time.time() - start_time,
                    'consecutive_failures_at_end': consecutive_failures,
                    'early_termination': actual_calls < total_calls
                }
            }

            if is_success:
                logger.info(
                    f"✅ 智能数据获取完成: 成功率 {successful_calls}/{actual_calls} ({successful_calls / actual_calls * 100:.1f}%)")
            else:
                logger.warning(
                    f"⚠️ 智能数据获取部分成功: 成功率 {successful_calls}/{actual_calls} ({successful_calls / actual_calls * 100:.1f}%)")

            return final_result

        except asyncio.CancelledError:
            processing_time = time.time() - start_time
            logger.error(f"💥 数据获取被取消")

            # 保存已获取的部分结果
            successful_results = {k: v for k, v in results.items() if v.get('success', False)}
            if successful_results:
                self._last_api_results = successful_results

            return {
                'success': len(successful_results) > 0,
                'error': 'Data acquisition cancelled',
                'results': results,
                'partial_results': successful_results,
                'failed_calls': failed_calls,
                'processing_time': processing_time,
                'execution_summary': {
                    'total_planned_calls': len(query_analysis.api_calls),
                    'completed_calls': len(results),
                    'successful_calls': len(successful_results),
                    'failed_calls': len(results) - len(successful_results),
                    'cancellation_occurred': True
                }
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 数据获取异常: {str(e)}")
            logger.error(f"💥 异常详情: {traceback.format_exc()}")

            # 即使出现异常，也要保存已获取的部分结果
            successful_results = {k: v for k, v in results.items() if v.get('success', False)}
            logger.info(f"🔄 保存部分成功结果: {len(successful_results)} 个")

            # 缓存部分结果
            if successful_results:
                self._last_api_results = successful_results

            return {
                'success': len(successful_results) > 0,  # 🔧 如果有部分成功结果，仍然算作成功
                'error': str(e),
                'results': results,  # 包含成功和失败的结果
                'partial_results': successful_results,
                'failed_calls': failed_calls,
                'exception_type': type(e).__name__,
                'processing_time': processing_time,
                'execution_summary': {
                    'total_planned_calls': len(query_analysis.api_calls),
                    'completed_calls': len(results),
                    'successful_calls': len(successful_results),
                    'failed_calls': len(results) - len(successful_results),
                    'exception_occurred': True,
                    'exception_message': str(e)
                }
            }

    # ================================================================
    # 数据提取（保持原有逻辑）
    # ================================================================

    async def _claude_three_layer_data_extraction(self,
                                                  raw_data: Dict[str, Any],
                                                  user_query: str,
                                                  query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """三层数据提取主方法 - 保持原有逻辑"""
        logger.debug("执行三层Claude数据提取")

        try:
            # 第一层：语义化数据收集
            semantic_result = self.semantic_collector.organize_semantic_data(raw_data, query_analysis)

            if 'error' in semantic_result:
                logger.error(f"语义化收集失败: {semantic_result['error']}")
                return self._fallback_data_extraction(raw_data)

            # 第二层：Claude智能提取
            extraction_result = await self.claude_extractor.extract_with_intelligence(
                semantic_result, user_query, query_analysis
            )

            if extraction_result.get('success', True):
                self.stats['successful_extractions'] += 1
                return extraction_result
            else:
                logger.warning("Claude提取失败，使用增强降级方案")
                return self._enhanced_fallback_data_extraction(raw_data, query_analysis)

        except Exception as e:
            logger.error(f"三层数据提取失败: {e}")
            return self._enhanced_fallback_data_extraction(raw_data, query_analysis)

    def _enhanced_fallback_data_extraction(self, raw_data: Dict[str, Any],
                                         query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """增强版降级数据提取"""
        self.stats['failed_extractions'] += 1

        # 使用查询类型信息进行智能降级
        if hasattr(query_analysis, 'query_type_info') and query_analysis.query_type_info:
            query_type = query_analysis.query_type_info.type
            if query_type == QueryType.COMPARISON:
                return self._extract_comparison_data_fallback(raw_data)
            elif query_type == QueryType.REINVESTMENT:
                return self._extract_reinvestment_data_fallback(raw_data)

        # 通用降级提取
        return self._fallback_data_extraction(raw_data)

    def _extract_comparison_data_fallback(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """对比查询的降级数据提取"""
        if not raw_data.get('success') or not raw_data.get('results'):
            return {'extracted_metrics': {}, 'extraction_method': 'comparison_fallback'}

        # 按时间段分组
        current_week_data = {}
        last_week_data = {}

        for key, result in raw_data['results'].items():
            if result.get('success') and result.get('data'):
                if 'current_week' in key:
                    current_week_data[key] = result['data']
                elif 'last_week' in key:
                    last_week_data[key] = result['data']

        # 聚合计算
        current_totals = self._aggregate_period_data(current_week_data)
        last_totals = self._aggregate_period_data(last_week_data)

        # 计算比较
        comparison_analysis = {}
        for metric in ['入金', '出金']:
            if metric in current_totals and metric in last_totals:
                current_val = current_totals[metric]
                last_val = last_totals[metric]
                if last_val > 0:
                    change_rate = (current_val - last_val) / last_val
                    comparison_analysis[metric] = {
                        'current_value': current_val,
                        'baseline_value': last_val,
                        'absolute_change': current_val - last_val,
                        'percentage_change': change_rate,
                        'change_direction': '增长' if change_rate > 0 else '下降' if change_rate < 0 else '持平'
                    }

        return {
            'extracted_metrics': {**current_totals, **last_totals},
            'comparison_analysis': comparison_analysis,
            'extraction_method': 'comparison_fallback',
            'data_quality_score': 0.7
        }

    def _extract_reinvestment_data_fallback(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """复投查询的降级数据提取"""
        extracted_metrics = {}

        for key, result in raw_data['results'].items():
            if result.get('success') and result.get('data'):
                data = result['data']
                if '到期金额' in data:
                    extracted_metrics['到期金额'] = float(data['到期金额'])
                if '总余额' in data:
                    extracted_metrics['总余额'] = float(data['总余额'])

        return {
            'extracted_metrics': extracted_metrics,
            'extraction_method': 'reinvestment_fallback',
            'data_quality_score': 0.6
        }

    def _aggregate_period_data(self, period_data: Dict[str, Any]) -> Dict[str, float]:
        """聚合时间段数据"""
        totals = {}

        for data_entry in period_data.values():
            if isinstance(data_entry, dict):
                for field in ['入金', '出金', '注册人数']:
                    if field in data_entry:
                        try:
                            value = float(data_entry[field])
                            totals[field] = totals.get(field, 0) + value
                        except (ValueError, TypeError):
                            pass

        return totals

    # ================================================================
    # 计算处理（保持原有逻辑）
    # ================================================================

    async def _statistical_calculation_processing(self,
                                                  query_analysis: QueryAnalysis,
                                                  extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """统计计算处理 - 增加数据调试"""

        if not query_analysis.needs_calculation:
            return {'needs_calculation': False}

        if not self.calculator:
            logger.error("统一计算器不可用")
            return {'error': '计算器不可用'}

        try:
            self.stats['calculation_calls'] += 1

            calculation_type = query_analysis.calculation_type
            calculation_params = getattr(query_analysis, 'calculation_params', {})

            logger.info(f"执行计算: {calculation_type}")

            # 🔍 调试：打印传递给计算器的数据结构
            logger.info(f"🔍 [DEBUG] extracted_data的顶级键: {list(extracted_data.keys())}")

            # 详细打印extracted_data的结构
            for key, value in extracted_data.items():
                if isinstance(value, dict):
                    logger.info(f"🔍 [DEBUG] {key}: {list(value.keys())}")
                    if key == 'extracted_metrics' and isinstance(value, dict):
                        logger.info(f"🔍 [DEBUG] extracted_metrics内容: {value}")
                else:
                    logger.info(f"🔍 [DEBUG] {key}: {type(value)} - {str(value)[:100]}")

            # 准备计算数据
            calc_data = self._prepare_calculation_data(extracted_data, query_analysis)

            # 🔍 调试：打印计算器实际接收的数据
            logger.info(f"🔍 [DEBUG] 传递给计算器的calc_data键: {list(calc_data.keys())}")
            for key, value in calc_data.items():
                if isinstance(value, dict):
                    logger.info(f"🔍 [DEBUG] calc_data[{key}]: {list(value.keys())}")
                else:
                    logger.info(f"🔍 [DEBUG] calc_data[{key}]: {type(value)}")

            # 调用统一计算器
            calc_result = await self.calculator.calculate(
                calculation_type=calculation_type,
                data=calc_data,
                params=calculation_params
            )

            return {
                'needs_calculation': True,
                'calculation_type': calculation_type,
                'success': calc_result.success if calc_result else False,
                'result': calc_result,
                'confidence': calc_result.confidence if calc_result else 0.0
            }

        except Exception as e:
            logger.error(f"统计计算处理失败: {e}")

            # 降级处理：如果计算失败，仍然可以继续
            return {
                'needs_calculation': True,
                'success': False,
                'error': str(e),
                'fallback_message': '计算模块出现问题，但数据提取成功，Claude将基于原始数据进行分析'
            }

    # 在 intelligent_qa_orchestrator.py 中修复 _prepare_calculation_data 方法

    def _prepare_calculation_data(self,
                                  extracted_data: Dict[str, Any],
                                  query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """准备计算数据 - 修复Claude数据传递"""

        logger.info("🔍 [DEBUG] 开始准备计算数据...")

        # 🔍 详细调试Claude提取的数据
        logger.info(f"🔍 [DEBUG] extracted_data的所有键: {list(extracted_data.keys())}")

        # 🆕 优先使用Claude提取器的对比分析结果
        claude_comparison = extracted_data.get('comparison_analysis', {})
        claude_detailed_calculations = extracted_data.get('detailed_calculations', {})
        claude_business_insights = extracted_data.get('business_insights', [])

        logger.info(f"🔍 [DEBUG] Claude对比分析结果: {len(claude_comparison)} 项")
        logger.info(f"🔍 [DEBUG] Claude详细计算: {len(claude_detailed_calculations)} 项")
        logger.info(f"🔍 [DEBUG] Claude业务洞察: {len(claude_business_insights)} 条")

        # 🆕 如果Claude已经做了对比分析，直接使用其结果
        if claude_comparison or claude_detailed_calculations:
            logger.info("✅ 发现Claude对比分析结果，直接使用")

            calc_data = {
                'extracted_metrics': extracted_data.get('extracted_metrics', {}),
                'derived_metrics': extracted_data.get('derived_metrics', {}),
                'comparison_analysis': claude_comparison or claude_detailed_calculations,  # 🆕 使用Claude的分析
                'business_insights': claude_business_insights,  # 🆕 传递业务洞察
                'user_query': query_analysis.original_query,
                'claude_analysis_available': True  # 🆕 标记Claude分析可用
            }

            # 🔍 调试Claude分析内容
            if claude_comparison:
                logger.info(f"🔍 [DEBUG] Claude对比分析内容: {list(claude_comparison.keys())}")
            if claude_detailed_calculations:
                logger.info(f"🔍 [DEBUG] Claude详细计算内容: {list(claude_detailed_calculations.keys())}")

        else:
            # 降级到原有逻辑：传递原始数据给计算器重新分析
            logger.info("⚠️ 未发现Claude对比分析，使用原始数据")

            calc_data = {
                'extracted_metrics': extracted_data.get('extracted_metrics', {}),
                'derived_metrics': extracted_data.get('derived_metrics', {}),
                'comparison_analysis': {},
                'user_query': query_analysis.original_query,
                'claude_analysis_available': False
            }

        # 🔍 调试基础数据
        logger.info(
            f"🔍 [DEBUG] 基础calc_data: extracted_metrics={len(calc_data['extracted_metrics'])}, comparison_analysis={len(calc_data['comparison_analysis'])}")

        calculation_type = query_analysis.calculation_type

        # 🆕 特殊处理对比分析数据
        if calculation_type == 'comparison_analysis':
            logger.info("🔍 [DEBUG] 为对比分析准备特殊数据...")

            # 🆕 如果Claude已经分析过，就不需要原始results了
            if not calc_data.get('claude_analysis_available', False):
                # 🆕 尝试从三层提取的完整结果中获取原始API数据
                raw_results = extracted_data.get('raw_results', {})
                if raw_results:
                    logger.info(f"🔍 [DEBUG] 发现raw_results: {len(raw_results)} 个")
                    calc_data['results'] = raw_results

                # 🆕 如果没有raw_results，尝试从缓存获取
                elif hasattr(self, '_last_api_results'):
                    logger.info("🔍 [DEBUG] 使用缓存的API结果")
                    calc_data['results'] = self._last_api_results

                # 🆕 确保有原始API结果用于深度查找
                if 'results' not in calc_data:
                    logger.warning("🔍 [DEBUG] 缺少原始API结果，对比分析可能不完整")
            else:
                logger.info("✅ Claude已完成分析，跳过原始数据传递")

        elif calculation_type == 'reinvestment_analysis':
            calc_data.update({
                'system_data': {
                    '总余额': extracted_data.get('extracted_metrics', {}).get('总余额', 0),
                    '总入金': extracted_data.get('extracted_metrics', {}).get('总入金', 0),
                    '总出金': extracted_data.get('extracted_metrics', {}).get('总出金', 0)
                }
            })

        elif calculation_type == 'cash_runway':
            calc_data.update({
                'system_data': {
                    '总余额': extracted_data.get('extracted_metrics', {}).get('总余额', 0)
                },
                'daily_data': [{
                    '出金': extracted_data.get('extracted_metrics', {}).get('出金', 0)
                }]
            })

        logger.info(f"🔍 [DEBUG] 最终calc_data键: {list(calc_data.keys())}")
        return calc_data

    # ================================================================
    # 🆕 增强的响应生成
    # ================================================================

    async def _enhanced_response_generation(self,
                                            user_query: str,
                                            query_analysis: QueryAnalysis,
                                            extracted_data: Dict[str, Any],
                                            calculation_results: Dict[str, Any]) -> str:
        """增强版响应生成 - 集成图表生成完整版"""

        self.stats['claude_generations'] += 1

        if not self.claude_client:
            logger.warning("⚠️ Claude客户端不可用，使用降级响应生成")
            return self._enhanced_response_generation_fallback(user_query, extracted_data, calculation_results)

        try:
            # 🆕 通用：如果有缓存的API结果，自动添加到extracted_data
            if hasattr(self, '_last_api_results') and self._last_api_results:
                extracted_data['raw_api_results'] = self._last_api_results

                # 🆕 智能判断数据类型并格式化
                data_count = len(self._last_api_results)
                logger.info(f"🔍 [DEBUG] API结果数量: {data_count}")

                if data_count > 1:  # 多数据自动格式化
                    extracted_data['multi_data_detected'] = True
                    extracted_data['data_count'] = data_count
                    logger.info(f"🔍 [DEBUG] 检测到多数据场景: {data_count} 条数据")

                    # 🆕 为Claude格式化详细数据
                    formatted_details = {}
                    daily_data_list = []

                    for key, result in self._last_api_results.items():
                        if result.get('success') and result.get('data'):
                            data = result['data']

                            # 根据数据类型进行不同的格式化
                            if isinstance(data, dict) and ('日期' in data or 'date' in key.lower()):
                                # 时间序列数据
                                daily_data_list.append({
                                    'api_key': key,
                                    'date': data.get('日期', ''),
                                    'date_formatted': self._format_date_for_display(data.get('日期', '')),
                                    'inflow': float(data.get('入金', 0)),
                                    'outflow': float(data.get('出金', 0)),
                                    'net_flow': float(data.get('入金', 0)) - float(data.get('出金', 0)),
                                    'registrations': int(data.get('注册人数', 0)),
                                    'purchases': int(data.get('购买产品数量', 0)),
                                    'holdings': int(data.get('持仓人数', 0)),
                                    'expirations': int(data.get('到期产品数量', 0))
                                })
                            elif isinstance(data, dict) and ('产品' in str(data) or 'product' in key.lower()):
                                # 产品数据
                                if '产品列表' in data:
                                    formatted_details['product_list'] = data.get('产品列表', [])
                            elif isinstance(data, dict) and ('用户' in str(data) or 'user' in key.lower()):
                                # 用户数据
                                if '用户列表' in data:
                                    formatted_details['user_list'] = data.get('用户列表', [])
                                if '每日数据' in data:
                                    formatted_details['user_daily_data'] = data.get('每日数据', [])

                    # 按日期排序时间序列数据
                    if daily_data_list:
                        daily_data_list.sort(key=lambda x: x.get('date', ''))
                        formatted_details['daily_data_list'] = daily_data_list
                        logger.info(f"🔍 [DEBUG] 格式化每日数据: {len(daily_data_list)} 天")

                    extracted_data['formatted_details'] = formatted_details

                else:
                    extracted_data['multi_data_detected'] = False
                    extracted_data['data_count'] = data_count
                    logger.info(f"🔍 [DEBUG] 单数据场景: {data_count} 条数据")

            # 🆕 Step 1: 启动并行任务 - 文本生成和图表生成
            logger.info("🚀 启动并行任务：文本生成 + 图表生成")

            # 创建并行任务
            text_generation_task = self._generate_text_response_core(
                user_query, query_analysis, extracted_data, calculation_results
            )

            chart_generation_task = self._generate_intelligent_charts(
                extracted_data, user_query, query_analysis
            )

            # 🆕 Step 2: 等待两个任务完成
            try:
                text_result, chart_result = await asyncio.gather(
                    text_generation_task,
                    chart_generation_task,
                    return_exceptions=True
                )
            except Exception as e:
                logger.error(f"并行任务执行失败: {e}")
                # 降级到串行执行
                text_result = await self._generate_text_response_core(
                    user_query, query_analysis, extracted_data, calculation_results
                )
                chart_result = Exception(f"图表生成跳过: {str(e)}")

            # 🆕 Step 3: 处理文本生成结果
            if isinstance(text_result, Exception):
                logger.error(f"文本生成失败: {text_result}")
                response_text = self._enhanced_response_generation_fallback(
                    user_query, extracted_data, calculation_results
                )
            else:
                response_text = text_result

            # 🆕 Step 4: 处理图表生成结果
            chart_section = ""
            if not isinstance(chart_result, Exception) and isinstance(chart_result, dict):
                if chart_result.get('success') and chart_result.get('chart_count', 0) > 0:
                    logger.info(f"✅ 图表生成成功: {chart_result.get('chart_count')} 个图表")

                    # 缓存图表结果供API响应使用
                    self._cache_chart_results(chart_result)

                    # 生成图表描述文本
                    chart_section = self._format_charts_for_text_response(chart_result)

                    if chart_section:
                        logger.info("📊 图表描述已添加到响应文本")
                    else:
                        logger.warning("⚠️ 图表生成成功但描述为空")
                else:
                    logger.warning(f"⚠️ 图表生成未成功: {chart_result.get('reason', '未知原因')}")
                    # 仍然缓存结果（即使失败）以便API返回错误信息
                    self._cache_chart_results(chart_result)
            else:
                error_msg = str(chart_result) if isinstance(chart_result, Exception) else "图表生成返回格式异常"
                logger.error(f"❌ 图表生成异常: {error_msg}")
                # 缓存失败结果
                self._cache_chart_results({
                    'success': False,
                    'error': error_msg,
                    'generation_method': 'failed',
                    'chart_count': 0,
                    'generated_charts': []
                })

            # 🆕 Step 5: 合并文本响应和图表描述
            final_response = response_text
            if chart_section:
                final_response += chart_section

            # 后处理：格式化货币显示
            if self.financial_formatter:
                final_response = self._format_currency_in_response(final_response)

            logger.info(f"✅ 增强响应生成完成，包含图表: {'是' if chart_section else '否'}")
            return final_response

        except Exception as e:
            logger.error(f"增强版响应生成失败: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")

            # 确保即使出现异常也缓存失败状态
            self._cache_chart_results({
                'success': False,
                'error': str(e),
                'generation_method': 'failed',
                'chart_count': 0,
                'generated_charts': []
            })

            return self._enhanced_response_generation_fallback(user_query, extracted_data, calculation_results)

    async def _generate_text_response_core(self,
                                           user_query: str,
                                           query_analysis: QueryAnalysis,
                                           extracted_data: Dict[str, Any],
                                           calculation_results: Dict[str, Any]) -> str:
        """
        核心文本响应生成 - 从原有逻辑提取
        """
        try:
            # 🔧 构建response_data用于prompt
            response_data = {
                'extracted_metrics': extracted_data.get('extracted_metrics', {}),
                'derived_metrics': extracted_data.get('derived_metrics', {}),
                'key_insights': extracted_data.get('key_insights', []),
                'business_health_indicators': extracted_data.get('business_health_indicators', {}),
                'detailed_daily_analysis': extracted_data.get('detailed_daily_analysis', {}),
                'weekly_pattern_analysis': extracted_data.get('weekly_pattern_analysis', {}),
                'recommendations': extracted_data.get('recommendations', []),
                'data_quality_assessment': extracted_data.get('data_quality_assessment', {}),
                'direct_answer': extracted_data.get('direct_answer', ''),
                'calculation_summary': self._summarize_calculation_results(calculation_results),
                'data_sources': extracted_data.get('source_data_summary', {}),
                'extraction_method': extracted_data.get('extraction_method', 'unknown'),
                'business_insights': extracted_data.get('business_insights', []),
                'detailed_calculations': extracted_data.get('detailed_calculations', {}),
                'raw_api_results': extracted_data.get('raw_api_results', {}),
                'multi_data_detected': extracted_data.get('multi_data_detected', False),
                'data_count': extracted_data.get('data_count', 0),
                'formatted_details': extracted_data.get('formatted_details', {})
            }

            # 🆕 使用PromptManager构建响应生成prompt
            prompt = self.prompt_manager.build_response_generation_prompt(
                user_query=user_query,
                query_analysis=query_analysis,
                extracted_data=extracted_data,
                calculation_results=calculation_results,
                query_type_result=getattr(query_analysis, 'query_type_info', None)
            )

            result = await self.claude_client.generate_text(prompt, max_tokens=6000)

            if result.get('success'):
                response_text = result.get('text', '').strip()
                logger.info("✅ Claude文本生成成功")
                return response_text
            else:
                error_msg = result.get('error', 'Claude调用失败')
                logger.warning(f"⚠️ Claude文本生成失败: {error_msg}")
                raise Exception(f"Claude生成失败: {error_msg}")

        except Exception as e:
            logger.error(f"核心文本生成异常: {e}")
            raise e

    def _cache_chart_results(self, chart_results: Dict[str, Any]):
        """缓存图表结果供ProcessingResult使用"""
        self._last_chart_results = chart_results
        logger.debug(
            f"🔄 缓存图表结果: success={chart_results.get('success')}, count={chart_results.get('chart_count', 0)}")

    def _format_charts_for_text_response(self, chart_results: Dict[str, Any]) -> str:
        """格式化图表结果用于文本响应"""

        if not chart_results.get('success') or chart_results.get('chart_count', 0) == 0:
            return ""

        charts = chart_results.get('generated_charts', [])

        if not charts:
            return ""

        chart_section = "\n\n📊 **数据可视化分析**："

        for i, chart in enumerate(charts, 1):
            if not chart.get('success', False):
                continue

            chart_type = chart.get('chart_type', '图表')
            title = chart.get('title', f'图表 {i}')

            chart_section += f"\n\n**{i}. {title}**"

            # 添加图表类型说明
            chart_type_desc = {
                'pie': '饼图',
                'donut': '环形图',
                'bar': '柱状图',
                'line': '折线图',
                'area': '面积图',
                'scatter': '散点图'
            }.get(chart_type, chart_type)

            chart_section += f" ({chart_type_desc})"

            # 添加数据摘要
            data_summary = chart.get('data_summary', {})
            if data_summary:
                if 'values' in data_summary and 'labels' in data_summary:
                    # 饼图类型的摘要
                    total_value = sum(data_summary['values']) if data_summary['values'] else 0
                    if total_value > 0:
                        chart_section += f"\n   📈 总计: {self.financial_formatter.format_currency(total_value) if self.financial_formatter else f'{total_value:,.2f}'}"
                        chart_section += "\n   📋 数据明细："

                        for label, value in zip(data_summary['labels'], data_summary['values']):
                            percentage = (value / total_value * 100) if total_value > 0 else 0
                            formatted_value = self.financial_formatter.format_currency(
                                value) if self.financial_formatter else f'{value:,.2f}'
                            chart_section += f"\n   • {label}: {formatted_value} ({percentage:.1f}%)"

                elif 'series' in data_summary:
                    # 柱状图/折线图类型的摘要
                    series_count = len(data_summary['series'])
                    chart_section += f"\n   📊 包含 {series_count} 个数据系列"

                    # 显示系列名称
                    if series_count <= 3:  # 只显示前3个系列
                        for series_name in list(data_summary['series'].keys())[:3]:
                            series_data = data_summary['series'][series_name]
                            if isinstance(series_data, list) and len(series_data) > 0:
                                avg_value = sum(series_data) / len(series_data) if series_data else 0
                                formatted_avg = self.financial_formatter.format_currency(
                                    avg_value) if self.financial_formatter else f'{avg_value:,.2f}'
                                chart_section += f"\n   • {series_name}: 平均值 {formatted_avg}"

        # 添加AI推荐说明
        if chart_results.get('recommendation_used', False):
            reasoning = chart_results.get('ai_reasoning', '')
            if reasoning:
                chart_section += f"\n\n💡 **AI智能分析**: {reasoning}"

        generation_method = chart_results.get('generation_method', 'rule_based')
        method_desc = {
            'ai_intelligent': '🤖 *图表由AI智能推荐生成，基于查询内容和数据特征自动选择最佳可视化方式*',
            'rule_based': '📋 *图表基于数据特征和业务规则自动生成*',
            'failed': '❌ *图表生成遇到问题*'
        }.get(generation_method, '📊 *图表已自动生成*')

        chart_section += f"\n\n{method_desc}"

        return chart_section

    def _format_date_for_display(self, date_str: str) -> str:
        """格式化日期用于显示"""
        try:
            if len(date_str) == 8 and date_str.isdigit():
                # YYYYMMDD格式转换
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{month}月{day}日"
            return date_str
        except:
            return date_str
    # 在 IntelligentQAOrchestrator 中添加图表生成方法
    async def _generate_intelligent_charts(self,
                                           extracted_data: Dict[str, Any],
                                           user_query: str,
                                           query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """
        🎨 生成智能图表

        Args:
            extracted_data: 提取的数据
            user_query: 用户查询
            query_analysis: 查询分析结果

        Returns:
            Dict[str, Any]: 图表生成结果
        """

        if not self.chart_generator:
            logger.warning("图表生成器不可用，跳过图表生成")
            return {
                'success': False,
                'reason': 'chart_generator_unavailable',
                'generated_charts': [],
                'chart_count': 0
            }

        try:
            logger.info("🎨 开始智能图表生成...")

            # 🎯 调用ChartGenerator的智能图表生成方法
            chart_results = await self.chart_generator.intelligent_chart_generation(
                extracted_data=extracted_data,
                user_query=user_query,
                auto_select=True
            )

            # 🔍 调试日志
            success = chart_results.get('success', False)
            chart_count = chart_results.get('chart_count', 0)
            generation_method = chart_results.get('generation_method', 'unknown')

            logger.info(f"📊 图表生成结果: success={success}, count={chart_count}, method={generation_method}")

            # 🆕 如果图表生成成功，处理图表数据
            if success and chart_count > 0:
                processed_charts = self._process_chart_results(chart_results)
                chart_results['processed_charts'] = processed_charts

            return chart_results

        except Exception as e:
            logger.error(f"❌ 图表生成异常: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")

            return {
                'success': False,
                'error': str(e),
                'generated_charts': [],
                'chart_count': 0,
                'generation_method': 'failed'
            }

    def _process_chart_results(self, chart_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理图表结果，转换为API友好的格式

        Args:
            chart_results: 原始图表结果

        Returns:
            List[Dict[str, Any]]: 处理后的图表数据
        """

        processed_charts = []
        generated_charts = chart_results.get('generated_charts', [])

        for i, chart in enumerate(generated_charts):
            try:
                processed_chart = {
                    'id': f"chart_{i + 1}",
                    'type': chart.get('chart_type', 'unknown'),
                    'title': chart.get('title', f'图表 {i + 1}'),
                    'success': chart.get('success', False)
                }

                # 🎯 处理图片数据
                if 'image_data' in chart and chart['image_data']:
                    image_data = chart['image_data']
                    if isinstance(image_data, str):
                        # 如果直接是base64字符串
                        processed_chart['image'] = {
                            'format': 'base64_png',
                            'data': image_data
                        }
                    elif isinstance(image_data, dict):
                        # 如果是包含格式信息的字典
                        processed_chart['image'] = {
                            'format': image_data.get('format', 'png'),
                            'data': image_data.get('base64') or image_data.get('data', ''),
                            'binary_available': 'binary' in image_data,
                            'svg_available': 'svg' in image_data,
                            'html_available': 'html' in image_data
                        }

                # 🎯 处理数据摘要
                if 'data_summary' in chart:
                    processed_chart['data_summary'] = chart['data_summary']

                # 🎯 添加图表描述和洞察
                processed_chart['description'] = self._generate_chart_description(chart)

                processed_charts.append(processed_chart)

            except Exception as e:
                logger.error(f"处理图表 {i + 1} 时出错: {e}")
                # 添加错误的图表项
                processed_charts.append({
                    'id': f"chart_{i + 1}_error",
                    'type': 'error',
                    'title': f'图表 {i + 1} 处理失败',
                    'success': False,
                    'error': str(e)
                })

        return processed_charts

    def _generate_chart_description(self, chart: Dict[str, Any]) -> str:
        """为图表生成描述文字"""

        chart_type = chart.get('chart_type', '图表')
        title = chart.get('title', '数据分析')
        data_summary = chart.get('data_summary', {})

        # 根据图表类型生成不同的描述
        if chart_type in ['pie', 'donut']:
            if 'values' in data_summary and 'labels' in data_summary:
                total = sum(data_summary['values']) if data_summary['values'] else 0
                max_item_idx = data_summary['values'].index(max(data_summary['values'])) if data_summary[
                    'values'] else 0
                max_item_label = data_summary['labels'][max_item_idx] if data_summary['labels'] else '未知'
                return f"{title}：总额 {total:,.2f}，其中 {max_item_label} 占比最大"

        elif chart_type == 'bar':
            if 'series' in data_summary:
                series_count = len(data_summary['series'])
                return f"{title}：包含 {series_count} 个数据系列的对比分析"

        elif chart_type == 'line':
            if 'series' in data_summary:
                series_count = len(data_summary['series'])
                return f"{title}：{series_count} 个指标的趋势变化图"

        return f"{title}：{chart_type}图表"

    def _format_charts_for_text_response(self, chart_results: Dict[str, Any]) -> str:
        """格式化图表结果用于文本响应"""

        if not chart_results.get('success') or chart_results.get('chart_count', 0) == 0:
            return ""

        charts = chart_results.get('generated_charts', [])
        processed_charts = chart_results.get('processed_charts', [])

        chart_section = "\n\n📊 **数据可视化分析**："

        for i, (chart, processed_chart) in enumerate(zip(charts, processed_charts), 1):
            if not chart.get('success', False):
                continue

            title = processed_chart.get('title', f'图表 {i}')
            description = processed_chart.get('description', '')

            chart_section += f"\n\n**{i}. {title}**"
            if description:
                chart_section += f"\n   {description}"

            # 添加数据摘要
            data_summary = processed_chart.get('data_summary', {})
            if data_summary:
                if 'values' in data_summary and 'labels' in data_summary:
                    # 饼图类型的摘要
                    chart_section += "\n   数据明细："
                    for label, value in zip(data_summary['labels'], data_summary['values']):
                        percentage = (value / sum(data_summary['values']) * 100) if sum(
                            data_summary['values']) > 0 else 0
                        chart_section += f"\n   • {label}: {value:,.2f} ({percentage:.1f}%)"

        # 添加AI推荐说明
        if chart_results.get('recommendation_used', False):
            reasoning = chart_results.get('ai_reasoning', '')
            if reasoning:
                chart_section += f"\n\n💡 **AI分析**: {reasoning}"

        generation_method = chart_results.get('generation_method', 'rule_based')
        if generation_method == 'ai_intelligent':
            chart_section += f"\n\n🤖 *图表由AI智能推荐生成*"
        elif generation_method == 'rule_based':
            chart_section += f"\n\n📋 *图表基于数据特征自动生成*"

        return chart_section
    def _format_date_for_display(self, date_str: str) -> str:
        """格式化日期用于显示"""
        try:
            if len(date_str) == 8 and date_str.isdigit():
                # YYYYMMDD格式转换
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{month}月{day}日"
            return date_str
        except:
            return date_str

    def _enhanced_response_generation_fallback(self,
                                               user_query: str,
                                               extracted_data: Dict[str, Any],
                                               calculation_results: Dict[str, Any]) -> str:
        """增强版降级响应生成"""

        # 检查是否是对比类查询
        if any(keyword in user_query for keyword in ['比较', '相比', '变化', '对比']):
            return self._generate_detailed_comparison_response(extracted_data, user_query)
        else:
            return self._generate_detailed_general_response(extracted_data, user_query)

    # ================================================================
    # 辅助方法（保持原有逻辑或简化）
    # ================================================================

    async def _execute_quick_api_call(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行快速API调用"""
        if not self.data_fetcher or not self.data_fetcher.api_connector:
            raise Exception("数据获取器不可用")

        method = api_info['method']
        params = api_info.get('params', {})
        api_connector = self.data_fetcher.api_connector

        if method == 'get_system_data':
            return await api_connector.get_system_data()
        elif method == 'get_daily_data':
            if 'date' in params:
                return await api_connector.get_daily_data(params['date'])
            else:
                return await api_connector.get_daily_data(self.current_date.strftime('%Y%m%d'))
        elif method == 'get_product_data':
            return await api_connector.get_product_data()
        else:
            raise Exception(f"不支持的快速API方法: {method}")

    async def _quick_data_extraction(self, raw_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """快速数据提取"""
        if not raw_data.get('success') or not raw_data.get('data'):
            return {'key_metrics': {}, 'extracted_info': '数据为空', 'data_quality': 0.0}

        data = raw_data['data']
        key_metrics = {}

        # 提取常见字段
        common_fields = ['总余额', '总入金', '总出金', '入金', '出金', '注册人数', '活跃用户数', '总用户数', '持仓人数']

        for field in common_fields:
            if field in data:
                try:
                    if field in ['入金', '出金', '总余额', '总入金', '总出金']:
                        key_metrics[field] = float(data[field])
                    else:
                        key_metrics[field] = int(data[field])
                except (ValueError, TypeError):
                    pass

        # 从用户统计中提取
        if '用户统计' in data and isinstance(data['用户统计'], dict):
            user_stats = data['用户统计']
            for field in ['总用户数', '活跃用户数']:
                if field in user_stats:
                    try:
                        key_metrics[field] = int(user_stats[field])
                    except (ValueError, TypeError):
                        pass

        return {
            'key_metrics': key_metrics,
            'extracted_info': f'提取了{len(key_metrics)}个关键指标',
            'data_quality': 0.9 if len(key_metrics) >= 3 else (0.7 if key_metrics else 0.3)
        }

    def _basic_response_generation(self, user_query: str, extracted_data: Dict[str, Any]) -> str:
        """基础响应生成"""
        key_metrics = extracted_data.get('key_metrics', {})

        if not key_metrics:
            return "抱歉，未能获取到相关数据。"

        response_parts = ["根据最新数据："]

        # 优先显示重要指标
        priority_fields = ['总余额', '入金', '出金', '活跃用户数', '注册人数']

        for field in priority_fields:
            if field in key_metrics:
                value = key_metrics[field]
                if isinstance(value, (int, float)):
                    if '余额' in field or '金额' in field or '入金' in field or '出金' in field:
                        response_parts.append(f"💰 {field}：¥{value:,.2f}")
                    elif '用户' in field or '人数' in field:
                        response_parts.append(f"👥 {field}：{int(value):,}人")
                    else:
                        response_parts.append(f"📊 {field}：{value}")

        return "\n".join(response_parts)

    def _generate_detailed_comparison_response(self, extracted_data: Dict[str, Any], user_query: str) -> str:
        """生成详细的对比分析响应"""
        response_parts = ["📊 **详细对比分析报告**\n"]

        # 对比分析
        comparison_analysis = extracted_data.get('comparison_analysis', {})
        if comparison_analysis:
            response_parts.append("📈 **对比分析结果**：")
            for metric, analysis in comparison_analysis.items():
                if isinstance(analysis, dict):
                    current = analysis.get('current_value', 0)
                    baseline = analysis.get('baseline_value', 0)
                    change_rate = analysis.get('percentage_change', 0)
                    direction = analysis.get('change_direction', '持平')

                    response_parts.append(f"💰 **{metric}**：")
                    response_parts.append(f"  - 当前值：¥{current:,.2f}")
                    response_parts.append(f"  - 对比值：¥{baseline:,.2f}")
                    response_parts.append(f"  - 变化：{direction} {change_rate:.2%}")

        # 数据质量评估
        data_quality = extracted_data.get('data_quality_score', 0)
        response_parts.append(f"\n🔍 **数据质量评分**：{data_quality:.1%}")

        return "\n".join(response_parts)

    def _generate_detailed_general_response(self, extracted_data: Dict[str, Any], user_query: str) -> str:
        """生成详细的通用响应"""
        response_parts = ["📊 **数据分析报告**\n"]

        # 直接答案
        direct_answer = extracted_data.get('direct_answer', '')
        if direct_answer:
            response_parts.append(f"🎯 **核心结论**：{direct_answer}\n")

        # 提取的指标
        extracted_metrics = extracted_data.get('extracted_metrics', {})
        if extracted_metrics:
            response_parts.append("💰 **核心业务指标**：")
            for key, value in extracted_metrics.items():
                if isinstance(value, (int, float)):
                    if '余额' in key or '金额' in key or '入金' in key or '出金' in key:
                        response_parts.append(f"• {key}：¥{value:,.2f}")
                    elif '人数' in key or '数量' in key:
                        response_parts.append(f"• {key}：{int(value):,}")

        # 关键洞察
        key_insights = extracted_data.get('key_insights', [])
        if key_insights:
            response_parts.append("\n💡 **关键洞察**：")
            for insight in key_insights:
                response_parts.append(f"• {insight}")

        return "\n".join(response_parts)

    def _format_currency_in_response(self, response_text: str) -> str:
        """格式化响应中的货币显示"""
        if not self.financial_formatter:
            return response_text
        return response_text

    def _summarize_calculation_results(self, calculation_results: Dict[str, Any]) -> Dict[str, Any]:
        """总结计算结果"""
        if not calculation_results.get('success'):
            return {'has_calculation': False}

        return {
            'has_calculation': True,
            'calculation_type': calculation_results.get('calculation_type'),
            'success': True,
            'confidence': calculation_results.get('confidence', 0.0)
        }

    def _fallback_data_extraction(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """基础降级数据提取"""
        if not raw_data.get('success') or not raw_data.get('results'):
            return {
                'extracted_metrics': {},
                'data_quality_score': 0.0,
                'extraction_method': 'fallback'
            }

        extracted_metrics = {}
        all_data = {}

        for key, result in raw_data['results'].items():
            if result.get('success') and result.get('data'):
                data = result['data']
                if isinstance(data, dict):
                    all_data.update(data)

        # 提取常见指标
        common_fields = ['总余额', '总入金', '总出金', '入金', '出金', '注册人数', '活跃用户数']

        for field in common_fields:
            if field in all_data:
                try:
                    extracted_metrics[field] = float(all_data[field])
                except (ValueError, TypeError):
                    pass

        return {
            'extracted_metrics': extracted_metrics,
            'data_quality_score': 0.7 if extracted_metrics else 0.2,
            'extraction_method': 'fallback'
        }

    async def _execute_fallback_to_comprehensive(self,
                                                 user_query: str,
                                                 session_id: str,
                                                 query_id: str,
                                                 start_time: float,
                                                 conversation_id: Optional[str]) -> ProcessingResult:
        """快速响应失败时的降级处理"""
        logger.info("快速响应失败，降级到完整分析")

        try:
            query_analysis = self._create_fallback_query_analysis(user_query, query_id)
            raw_data = await self._intelligent_data_acquisition(query_analysis)
            extracted_data = self._fallback_data_extraction(raw_data)
            response_text = self._basic_response_generation(user_query, extracted_data)

            total_time = time.time() - start_time

            return ProcessingResult(
                session_id=session_id,
                query_id=query_id,
                success=True,
                response_text=response_text,
                extracted_data=extracted_data,
                complexity=QueryComplexity.SIMPLE,
                processing_path="fallback",
                confidence_score=0.6,
                total_processing_time=total_time,
                processing_strategy=ProcessingStrategy.FALLBACK,
                conversation_id=conversation_id
            )

        except Exception as e:
            logger.error(f"降级处理也失败了: {e}")
            return self._create_error_result(session_id, query_id, user_query, str(e),
                                             time.time() - start_time, conversation_id)

    def _create_fallback_query_analysis(self, user_query: str, query_id: str) -> QueryAnalysis:
        """创建降级查询分析"""
        return QueryAnalysis(
            original_query=user_query,
            complexity=QueryComplexity.SIMPLE,
            is_quick_response=False,
            intent='数据查询',
            confidence=0.4,
            api_calls=[{'method': 'get_system_data', 'params': {}, 'reason': '降级数据获取'}]
        )

    async def _save_conversation_if_needed(self,
                                           conversation_id: Optional[str],
                                           user_query: str,
                                           result: ProcessingResult):
        """保存对话记录"""
        if not conversation_id or not self.conversation_manager:
            return

        try:
            conv_id = int(conversation_id)

            # 保存用户消息
            self.conversation_manager.add_message(conv_id, True, user_query)

            # 保存AI回复消息
            ai_message_id = self.conversation_manager.add_message(conv_id, False, result.response_text)

            # 🆕 保存图表数据到数据库
            if result.visualizations and ai_message_id:
                for i, visualization in enumerate(result.visualizations):
                    try:
                        self.conversation_manager.add_visual(
                            message_id=ai_message_id,
                            visual_type=visualization.get('type', 'chart'),
                            visual_order=i,
                            title=visualization.get('title', f'图表 {i + 1}'),
                            data=visualization.get('data', {})
                        )
                        logger.info(f"✅ 保存图表数据到数据库: message_id={ai_message_id}, visual_order={i}")
                    except Exception as e:
                        logger.error(f"❌ 保存图表数据失败: {e}")

        except Exception as e:
            logger.error(f"保存对话记录失败: {e}")

    def _create_error_result(self,
                             session_id: str,
                             query_id: str,
                             user_query: str,
                             error_msg: str,
                             processing_time: float,
                             conversation_id: Optional[str]) -> ProcessingResult:
        """创建错误结果"""
        return ProcessingResult(
            session_id=session_id,
            query_id=query_id,
            success=False,
            response_text=f"抱歉，处理您的查询时遇到问题。请稍后重试或联系技术支持。",
            complexity=QueryComplexity.SIMPLE,
            processing_path="error",
            confidence_score=0.0,
            total_processing_time=processing_time,
            processing_strategy=ProcessingStrategy.FALLBACK,
            error_info={
                'message': error_msg,
                'query': user_query,
                'timestamp': datetime.now().isoformat()
            },
            conversation_id=conversation_id
        )

    # ================================================================
    # 🆕 管理方法 - 增强统计
    # ================================================================

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """获取编排器统计信息"""
        total = self.stats['total_queries']
        stats = self.stats.copy()

        if total > 0:
            stats['quick_response_rate'] = self.stats['quick_responses'] / total
            stats['comprehensive_rate'] = self.stats['comprehensive_analyses'] / total
            stats['average_processing_time'] = self.stats['processing_time_total'] / total

        # 🆕 添加组件统计
        if hasattr(self, 'strategy_extractor') and self.strategy_extractor:
            # 获取策略提取器的统计信息，但不直接将字典赋值给stats
            extractor_stats = self.strategy_extractor.get_stats()
            # 将策略提取器的关键统计数据作为单独的条目添加到stats中
            if isinstance(extractor_stats, dict):
                for key, value in extractor_stats.items():
                    if isinstance(value, (int, float)):
                        stats[f'strategy_extractor_{key}'] = value

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.initialized else 'not_initialized',
            'components': {
                'claude_available': self.claude_client is not None,
                'gpt_available': self.gpt_client is not None,
                'data_fetcher_ready': self.data_fetcher is not None,
                'calculator_ready': self.calculator is not None,
                'conversation_manager_ready': self.conversation_manager is not None,
                # 🆕 新组件状态
                'query_type_detector_ready': self.query_type_detector is not None,
                'prompt_manager_ready': self.prompt_manager is not None,
                'strategy_extractor_ready': self.strategy_extractor is not None,
            },
            'features': {
                'enhanced_query_detection': True,
                'prompt_management': True,
                'intelligent_strategy_extraction': True,
                'three_layer_data_extraction': True
            },
            'stats': self.get_orchestrator_stats(),
            'current_date': self.current_date.isoformat(),
            'timestamp': datetime.now().isoformat()
        }

    async def close(self):
        """关闭编排器"""
        logger.info("关闭精简重构后的智能编排器...")

        if self.data_fetcher and hasattr(self.data_fetcher, 'close'):
            await self.data_fetcher.close()

        if self.db_connector and hasattr(self.db_connector, 'close'):
            await self.db_connector.close()

        self.initialized = False
        logger.info("智能编排器已关闭")

# ================================================================
# 工厂函数
# ================================================================

def create_enhanced_orchestrator(claude_client: Optional[ClaudeClient] = None,
                                gpt_client: Optional[OpenAIClient] = None,
                                db_connector: Optional[DatabaseConnector] = None,
                                app_config: Optional[AppConfig] = None) -> IntelligentQAOrchestrator:
    """创建增强版智能编排器"""
    return IntelligentQAOrchestrator(claude_client, gpt_client, db_connector, app_config)