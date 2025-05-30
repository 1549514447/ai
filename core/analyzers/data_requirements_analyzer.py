# core/analyzers/data_requirements_analyzer.py
"""
🔍 智能数据需求分析器
基于查询解析结果，精确分析和规划数据获取策略

核心特点:
- 精细化API调用策略规划
- 智能数据依赖关系分析
- 动态数据质量要求评估
- 优化的数据获取时序安排
- 数据成本效益分析
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json

# 导入我们的工具类和类型
from utils.helpers.date_utils import DateUtils, create_date_utils
from utils.helpers.validation_utils import ValidationUtils, create_validation_utils

logger = logging.getLogger(__name__)


class DataPriority(Enum):
    """数据优先级"""
    CRITICAL = "critical"  # 关键数据，必须获取
    HIGH = "high"  # 高优先级，强烈建议获取
    MEDIUM = "medium"  # 中等优先级，建议获取
    LOW = "low"  # 低优先级，可选获取
    OPTIONAL = "optional"  # 可选数据，资源充足时获取


class DataFreshness(Enum):
    """数据新鲜度要求"""
    REALTIME = "realtime"  # 实时数据 (<1分钟)
    FRESH = "fresh"  # 新鲜数据 (<5分钟)
    CURRENT = "current"  # 当前数据 (<30分钟)
    RECENT = "recent"  # 最近数据 (<2小时)
    DAILY = "daily"  # 日度数据 (<24小时)


class DataScope(Enum):
    """数据范围类型"""
    POINT = "point"  # 单点数据 (特定日期/时刻)
    RANGE = "range"  # 范围数据 (时间段)
    COMPREHENSIVE = "comprehensive"  # 综合数据 (多维度)
    COMPARATIVE = "comparative"  # 对比数据 (多时期)


@dataclass
class APICallPlan:
    """API调用计划"""
    api_endpoint: str  # API端点
    call_method: str  # 调用方法名
    parameters: Dict[str, Any]  # 调用参数
    priority: DataPriority  # 优先级
    estimated_time: float  # 预估耗时(秒)
    estimated_cost: float  # 预估成本(相对值)
    data_volume_expected: str  # 预期数据量 (small/medium/large)
    depends_on: List[str]  # 依赖的其他调用
    cache_strategy: str  # 缓存策略
    retry_strategy: str  # 重试策略
    validation_rules: List[str]  # 数据验证规则


@dataclass
class DataRequirement:
    """数据需求定义"""
    requirement_id: str  # 需求ID
    requirement_type: str  # 需求类型
    data_source: str  # 数据源
    data_scope: DataScope  # 数据范围
    freshness_requirement: DataFreshness  # 新鲜度要求
    priority: DataPriority  # 优先级
    quality_threshold: float  # 质量阈值 (0-1)
    fallback_options: List[str]  # 降级选项
    processing_requirements: Dict[str, Any]  # 处理要求
    business_justification: str  # 业务理由


@dataclass
class DataAcquisitionPlan:
    """数据获取计划"""
    plan_id: str  # 计划ID
    query_analysis_summary: Dict[str, Any]  # 查询分析摘要
    data_requirements: List[DataRequirement]  # 数据需求列表
    api_call_plans: List[APICallPlan]  # API调用计划
    execution_sequence: List[str]  # 执行序列
    parallel_groups: List[List[str]]  # 并行执行组
    total_estimated_time: float  # 总预估时间
    total_estimated_cost: float  # 总预估成本
    success_criteria: Dict[str, Any]  # 成功标准
    fallback_strategies: Dict[str, Any]  # 降级策略
    optimization_notes: List[str]  # 优化建议
    plan_confidence: float  # 计划置信度


class DataRequirementsAnalyzer:
    """
    🔍 智能数据需求分析器

    功能架构:
    1. 基于查询解析结果的精细化需求分析
    2. 多维度数据依赖关系分析
    3. 优化的API调用策略规划
    4. 动态数据质量和成本控制
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化数据需求分析器

        Args:
            claude_client: Claude客户端，用于业务逻辑分析
            gpt_client: GPT客户端，用于数据策略优化
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.date_utils = create_date_utils(claude_client)
        self.validator = create_validation_utils(claude_client, gpt_client)

        # API配置信息
        self.api_catalog = self._build_api_catalog()

        # 数据成本模型
        self.cost_model = self._build_cost_model()

        # 分析统计
        self.analysis_stats = {
            'total_analyses': 0,
            'optimization_suggestions': 0,
            'cost_savings_achieved': 0.0,
            'avg_plan_confidence': 0.0
        }

        logger.info("DataRequirementsAnalyzer initialized")

    def _build_api_catalog(self) -> Dict[str, Any]:
        """构建API目录信息"""
        return {
            "system": {
                "endpoint": "/api/sta/system",
                "method": "get_system_data",
                "data_type": "realtime_overview",
                "typical_response_time": 2.0,
                "data_volume": "small",
                "cache_recommended": True,
                "cache_ttl": 60,
                "provides": [
                    "总余额", "总入金", "总出金", "总投资金额", "总奖励发放",
                    "用户统计", "产品统计", "到期概览"
                ],
                "best_for": ["current_status", "overview", "realtime_queries"]
            },
            "daily": {
                "endpoint": "/api/sta/day",
                "method": "get_daily_data",
                "data_type": "historical_daily",
                "typical_response_time": 1.5,
                "data_volume": "small",
                "cache_recommended": True,
                "cache_ttl": 300,
                "provides": [
                    "日期", "注册人数", "持仓人数", "购买产品数量",
                    "到期产品数量", "入金", "出金"
                ],
                "best_for": ["trend_analysis", "historical_data", "daily_metrics"]
            },
            "product": {
                "endpoint": "/api/sta/product",
                "method": "get_product_data",
                "data_type": "product_catalog",
                "typical_response_time": 3.0,
                "data_volume": "medium",
                "cache_recommended": True,
                "cache_ttl": 600,
                "provides": [
                    "产品列表", "产品价格", "期限天数", "每日利率",
                    "总购买次数", "持有情况", "即将到期数"
                ],
                "best_for": ["product_analysis", "expiry_prediction", "portfolio_analysis"]
            },
            "user_daily": {
                "endpoint": "/api/sta/user_daily",
                "method": "get_user_daily_data",
                "data_type": "user_behavior_daily",
                "typical_response_time": 2.0,
                "data_volume": "small",
                "cache_recommended": True,
                "cache_ttl": 300,
                "provides": [
                    "每日新增用户", "VIP等级分布", "用户行为数据"
                ],
                "best_for": ["user_analysis", "behavior_trends", "segmentation"]
            },
            "user": {
                "endpoint": "/api/sta/user",
                "method": "get_user_data",
                "data_type": "user_details",
                "typical_response_time": 4.0,
                "data_volume": "large",
                "cache_recommended": False,
                "cache_ttl": 0,
                "provides": [
                    "用户详情", "投资金额", "累计奖励", "投报比"
                ],
                "best_for": ["user_profiling", "detailed_analysis", "compliance"]
            },
            "product_end_single": {
                "endpoint": "/api/sta/product_end",
                "method": "get_product_end_data",
                "data_type": "single_day_expiry",
                "typical_response_time": 1.5,
                "data_volume": "small",
                "cache_recommended": True,
                "cache_ttl": 180,
                "provides": [
                    "指定日期到期产品", "到期金额", "产品分布"
                ],
                "best_for": ["specific_date_expiry", "cash_flow_planning"]
            },
            "product_end_interval": {
                "endpoint": "/api/sta/product_end_interval",
                "method": "get_product_end_interval",
                "data_type": "range_expiry",
                "typical_response_time": 2.5,
                "data_volume": "medium",
                "cache_recommended": True,
                "cache_ttl": 240,
                "provides": [
                    "时间段到期产品", "区间到期金额", "到期分布统计"
                ],
                "best_for": ["period_expiry_analysis", "cash_flow_forecasting"]
            }
        }

    def _build_cost_model(self) -> Dict[str, Any]:
        """构建数据成本模型"""
        return {
            "api_call_cost": {
                "system": 1.0,  # 基准成本
                "daily": 1.2,  # 稍高，因为可能需要批量调用
                "product": 1.5,  # 中等，数据量较大
                "user_daily": 1.1,  # 较低
                "user": 3.0,  # 最高，大数据量且分页
                "product_end_single": 1.0,
                "product_end_interval": 1.3
            },
            "batch_cost_multiplier": {
                "1-5": 1.0,  # 1-5个调用，正常成本
                "6-20": 0.9,  # 6-20个调用，小幅优化
                "21-50": 0.8,  # 21-50个调用，中等优化
                "50+": 0.7  # 50+个调用，最大优化
            },
            "cache_cost_benefit": 0.3,  # 缓存可节省30%成本
            "parallel_efficiency": 0.15  # 并行执行可提升15%效率
        }

    # ============= 核心需求分析方法 =============

    async def analyze_data_requirements(self, query_analysis_result: Any) -> DataAcquisitionPlan:
        """
        🎯 分析数据需求 - 核心入口方法

        Args:
            query_analysis_result: 查询解析结果 (来自SmartQueryParser)

        Returns:
            DataAcquisitionPlan: 完整的数据获取计划
        """
        try:
            logger.info("🔍 开始分析数据需求")
            self.analysis_stats['total_analyses'] += 1

            # 第1步: 提取查询分析关键信息
            query_summary = self._extract_query_summary(query_analysis_result)

            # 第2步: 基于复杂度和类型确定数据需求
            base_requirements = await self._determine_base_requirements(query_analysis_result)

            # 第3步: 分析时间维度的数据需求
            temporal_requirements = await self._analyze_temporal_requirements(query_analysis_result)

            # 第4步: 分析业务参数相关的数据需求
            business_requirements = await self._analyze_business_requirements(query_analysis_result)

            # 第5步: 合并和去重数据需求
            consolidated_requirements = self._consolidate_requirements(
                base_requirements, temporal_requirements, business_requirements
            )

            # 第6步: 生成API调用计划
            api_call_plans = await self._generate_api_call_plans(consolidated_requirements)

            # 第7步: 优化执行序列和并行策略
            execution_strategy = await self._optimize_execution_strategy(api_call_plans)

            # 第8步: 成本效益分析和优化建议
            cost_analysis = self._analyze_cost_and_optimize(api_call_plans, execution_strategy)

            # 第9步: 构建最终数据获取计划
            acquisition_plan = self._build_acquisition_plan(
                query_summary, consolidated_requirements, api_call_plans,
                execution_strategy, cost_analysis
            )

            logger.info(f"✅ 数据需求分析完成: {len(consolidated_requirements)}个需求, {len(api_call_plans)}个API调用")

            return acquisition_plan

        except Exception as e:
            logger.error(f"❌ 数据需求分析失败: {str(e)}")
            return self._create_fallback_plan(query_analysis_result, str(e))

    # ============= 需求分析层 =============

    def _extract_query_summary(self, query_analysis_result: Any) -> Dict[str, Any]:
        """提取查询分析关键信息"""

        return {
            "original_query": query_analysis_result.original_query,
            "complexity": query_analysis_result.complexity.value,
            "query_type": query_analysis_result.query_type.value,
            "business_scenario": query_analysis_result.business_scenario.value,
            "confidence_score": query_analysis_result.confidence_score,
            "estimated_time": query_analysis_result.estimated_total_time,
            "execution_steps": len(query_analysis_result.execution_plan),
            "ai_collaboration_level": query_analysis_result.ai_collaboration_plan.get("collaboration_level", "minimal")
        }

    async def _determine_base_requirements(self, query_analysis_result: Any) -> List[DataRequirement]:
        """确定基础数据需求"""

        base_requirements = []
        complexity = query_analysis_result.complexity
        query_type = query_analysis_result.query_type

        # 🎯 根据复杂度确定基础需求
        if complexity.value == "simple":
            # 简单查询：通常只需要当前状态数据
            base_requirements.append(DataRequirement(
                requirement_id="simple_current_state",
                requirement_type="current_status",
                data_source="system",
                data_scope=DataScope.POINT,
                freshness_requirement=DataFreshness.CURRENT,
                priority=DataPriority.CRITICAL,
                quality_threshold=0.9,
                fallback_options=["cached_system_data"],
                processing_requirements={"format": "summary"},
                business_justification="获取当前系统状态用于直接回答"
            ))

        elif complexity.value == "medium":
            # 中等复杂度：需要当前数据 + 一些历史数据
            base_requirements.extend([
                DataRequirement(
                    requirement_id="medium_current_state",
                    requirement_type="current_status",
                    data_source="system",
                    data_scope=DataScope.POINT,
                    freshness_requirement=DataFreshness.FRESH,
                    priority=DataPriority.CRITICAL,
                    quality_threshold=0.85,
                    fallback_options=[],
                    processing_requirements={"format": "detailed"},
                    business_justification="当前状态作为分析基线"
                ),
                DataRequirement(
                    requirement_id="medium_recent_history",
                    requirement_type="historical_trend",
                    data_source="daily",
                    data_scope=DataScope.RANGE,
                    freshness_requirement=DataFreshness.DAILY,
                    priority=DataPriority.HIGH,
                    quality_threshold=0.8,
                    fallback_options=["shorter_history"],
                    processing_requirements={"time_range": "recent_30_days"},
                    business_justification="最近趋势分析所需历史数据"
                )
            ])

        elif complexity.value in ["complex", "expert"]:
            # 复杂查询：需要综合数据
            base_requirements.extend([
                DataRequirement(
                    requirement_id="complex_comprehensive_current",
                    requirement_type="comprehensive_current",
                    data_source="system",
                    data_scope=DataScope.COMPREHENSIVE,
                    freshness_requirement=DataFreshness.FRESH,
                    priority=DataPriority.CRITICAL,
                    quality_threshold=0.9,
                    fallback_options=[],
                    processing_requirements={"include_metadata": True},
                    business_justification="复杂分析的当前状态基础"
                ),
                DataRequirement(
                    requirement_id="complex_extended_history",
                    requirement_type="extended_historical",
                    data_source="daily",
                    data_scope=DataScope.RANGE,
                    freshness_requirement=DataFreshness.DAILY,
                    priority=DataPriority.HIGH,
                    quality_threshold=0.85,
                    fallback_options=["medium_history"],
                    processing_requirements={"time_range": "extended_90_days"},
                    business_justification="深度分析所需扩展历史数据"
                ),
                DataRequirement(
                    requirement_id="complex_product_analysis",
                    requirement_type="product_portfolio",
                    data_source="product",
                    data_scope=DataScope.COMPREHENSIVE,
                    freshness_requirement=DataFreshness.CURRENT,
                    priority=DataPriority.MEDIUM,
                    quality_threshold=0.8,
                    fallback_options=[],
                    processing_requirements={"include_predictions": True},
                    business_justification="产品组合分析和到期预测"
                )
            ])

            # 专家级还需要用户数据
            if complexity.value == "expert":
                base_requirements.append(DataRequirement(
                    requirement_id="expert_user_behavior",
                    requirement_type="user_behavior_analysis",
                    data_source="user_daily",
                    data_scope=DataScope.RANGE,
                    freshness_requirement=DataFreshness.DAILY,
                    priority=DataPriority.MEDIUM,
                    quality_threshold=0.75,
                    fallback_options=["aggregated_user_stats"],
                    processing_requirements={"time_range": "user_behavior_60_days"},
                    business_justification="用户行为模式分析"
                ))

        # 🎯 根据查询类型添加特定需求
        if query_type.value == "prediction":
            base_requirements.append(DataRequirement(
                requirement_id="prediction_baseline",
                requirement_type="prediction_foundation",
                data_source="daily",
                data_scope=DataScope.RANGE,
                freshness_requirement=DataFreshness.DAILY,
                priority=DataPriority.CRITICAL,
                quality_threshold=0.9,
                fallback_options=[],
                processing_requirements={"min_data_points": 60},
                business_justification="预测模型需要充足的历史数据"
            ))

        elif query_type.value == "risk_assessment":
            base_requirements.append(DataRequirement(
                requirement_id="risk_comprehensive_data",
                requirement_type="risk_analysis_data",
                data_source="comprehensive",
                data_scope=DataScope.COMPREHENSIVE,
                freshness_requirement=DataFreshness.FRESH,
                priority=DataPriority.CRITICAL,
                quality_threshold=0.95,
                fallback_options=[],
                processing_requirements={"include_stress_scenarios": True},
                business_justification="风险评估需要全面准确的数据"
            ))

        return base_requirements

    async def _analyze_temporal_requirements(self, query_analysis_result: Any) -> List[DataRequirement]:
        """分析时间维度数据需求"""

        temporal_requirements = []
        time_requirements = query_analysis_result.time_requirements
        date_parse_result = query_analysis_result.date_parse_result

        # 如果查询中包含明确的时间信息
        if date_parse_result and date_parse_result.has_time_info:

            # 处理具体日期需求
            for date in date_parse_result.dates:
                temporal_requirements.append(DataRequirement(
                    requirement_id=f"specific_date_{date.replace('-', '_')}",
                    requirement_type="specific_date_data",
                    data_source="daily",
                    data_scope=DataScope.POINT,
                    freshness_requirement=DataFreshness.DAILY,
                    priority=DataPriority.HIGH,
                    quality_threshold=0.85,
                    fallback_options=["interpolated_data"],
                    processing_requirements={"target_date": date},
                    business_justification=f"用户明确要求{date}的数据"
                ))

            # 处理日期范围需求
            for date_range in date_parse_result.ranges:
                range_id = f"range_{date_range.start_date.replace('-', '_')}_to_{date_range.end_date.replace('-', '_')}"
                temporal_requirements.append(DataRequirement(
                    requirement_id=range_id,
                    requirement_type="date_range_data",
                    data_source="daily",
                    data_scope=DataScope.RANGE,
                    freshness_requirement=DataFreshness.DAILY,
                    priority=DataPriority.HIGH,
                    quality_threshold=0.8,
                    fallback_options=["partial_range"],
                    processing_requirements={
                        "start_date": date_range.start_date,
                        "end_date": date_range.end_date,
                        "range_type": date_range.range_type
                    },
                    business_justification=f"用户要求时间范围: {date_range.description}"
                ))

        # 分析是否需要到期数据
        original_query = query_analysis_result.original_query.lower()
        if any(keyword in original_query for keyword in ["到期", "expiry", "mature", "到期"]):

            # 确定到期数据的时间范围
            if date_parse_result and date_parse_result.ranges:
                # 使用查询中的时间范围
                for date_range in date_parse_result.ranges:
                    temporal_requirements.append(DataRequirement(
                        requirement_id=f"expiry_range_{date_range.start_date.replace('-', '_')}",
                        requirement_type="expiry_range_data",
                        data_source="product_end_interval",
                        data_scope=DataScope.RANGE,
                        freshness_requirement=DataFreshness.CURRENT,
                        priority=DataPriority.CRITICAL,
                        quality_threshold=0.9,
                        fallback_options=["daily_expiry_aggregation"],
                        processing_requirements={
                            "start_date": date_range.start_date,
                            "end_date": date_range.end_date
                        },
                        business_justification="用户查询涉及产品到期分析"
                    ))
            else:
                # 默认获取未来一周的到期数据
                today = datetime.now()
                week_end = today + timedelta(days=7)
                temporal_requirements.append(DataRequirement(
                    requirement_id="default_expiry_week",
                    requirement_type="default_expiry_data",
                    data_source="product_end_interval",
                    data_scope=DataScope.RANGE,
                    freshness_requirement=DataFreshness.CURRENT,
                    priority=DataPriority.HIGH,
                    quality_threshold=0.85,
                    fallback_options=["single_day_expiry"],
                    processing_requirements={
                        "start_date": today.strftime("%Y-%m-%d"),
                        "end_date": week_end.strftime("%Y-%m-%d")
                    },
                    business_justification="到期相关查询的默认时间窗口"
                ))

        return temporal_requirements

    async def _analyze_business_requirements(self, query_analysis_result: Any) -> List[DataRequirement]:
        """分析业务参数相关数据需求"""

        business_requirements = []
        business_parameters = query_analysis_result.business_parameters
        calculation_requirements = query_analysis_result.calculation_requirements
        query_type = query_analysis_result.query_type

        # 分析查询中的关键业务概念
        original_query = query_analysis_result.original_query.lower()

        # 🔍 复投/提现相关查询
        if any(keyword in original_query for keyword in ["复投", "提现", "reinvest", "cashout"]):
            business_requirements.append(DataRequirement(
                requirement_id="reinvestment_analysis_data",
                requirement_type="reinvestment_calculation_base",
                data_source="product",
                data_scope=DataScope.COMPREHENSIVE,
                freshness_requirement=DataFreshness.CURRENT,
                priority=DataPriority.CRITICAL,
                quality_threshold=0.9,
                fallback_options=[],
                processing_requirements={
                    "include_product_rates": True,
                    "include_expiry_schedule": True
                },
                business_justification="复投/提现计算需要完整的产品和到期信息"
            ))

        # 🔍 用户相关分析
        if any(keyword in original_query for keyword in ["用户", "会员", "user", "member"]):
            if query_type.value in ["user_behavior", "growth_analysis"]:
                business_requirements.append(DataRequirement(
                    requirement_id="user_comprehensive_analysis",
                    requirement_type="user_analysis_base",
                    data_source="user_daily",
                    data_scope=DataScope.RANGE,
                    freshness_requirement=DataFreshness.DAILY,
                    priority=DataPriority.HIGH,
                    quality_threshold=0.8,
                    fallback_options=["aggregated_user_metrics"],
                    processing_requirements={"include_vip_breakdown": True},
                    business_justification="用户行为和增长分析需要用户详细数据"
                ))

        # 🔍 产品表现分析
        if any(keyword in original_query for keyword in ["产品", "product", "portfolio"]):
            business_requirements.append(DataRequirement(
                requirement_id="product_performance_data",
                requirement_type="product_analysis_base",
                data_source="product",
                data_scope=DataScope.COMPREHENSIVE,
                freshness_requirement=DataFreshness.CURRENT,
                priority=DataPriority.HIGH,
                quality_threshold=0.85,
                fallback_options=[],
                processing_requirements={
                    "include_performance_metrics": True,
                    "include_holding_analysis": True
                },
                business_justification="产品表现分析需要完整的产品数据"
            ))

        # 🔍 财务规划相关
        if query_analysis_result.business_scenario.value == "financial_planning":
            business_requirements.append(DataRequirement(
                requirement_id="financial_planning_comprehensive",
                requirement_type="financial_planning_base",
                data_source="comprehensive",
                data_scope=DataScope.COMPREHENSIVE,
                freshness_requirement=DataFreshness.FRESH,
                priority=DataPriority.CRITICAL,
                quality_threshold=0.9,
                fallback_options=[],
                processing_requirements={"include_all_metrics": True},
                business_justification="财务规划需要全面的财务数据"
            ))

        return business_requirements

    # ============= 需求整合和API规划层 =============

    def _consolidate_requirements(self, *requirement_lists: List[DataRequirement]) -> List[DataRequirement]:
        """合并和去重数据需求"""

        all_requirements = []
        for req_list in requirement_lists:
            all_requirements.extend(req_list)

        # 去重和合并相似需求
        consolidated = {}

        for req in all_requirements:
            key = f"{req.data_source}_{req.requirement_type}"

            if key in consolidated:
                # 合并相似需求，取优先级更高的
                existing = consolidated[key]
                if req.priority.value == "critical" or existing.priority.value != "critical":
                    consolidated[key] = req
            else:
                consolidated[key] = req

        # 按优先级排序
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "optional": 4}
        sorted_requirements = sorted(
            consolidated.values(),
            key=lambda x: priority_order[x.priority.value]
        )

        logger.info(f"📋 合并需求: {len(all_requirements)} -> {len(sorted_requirements)}")
        return sorted_requirements

    async def _generate_api_call_plans(self, requirements: List[DataRequirement]) -> List[APICallPlan]:
        """生成API调用计划"""

        api_call_plans = []

        for req in requirements:
            # 🎯 根据数据源和需求类型确定API调用
            if req.data_source == "system":
                api_call_plans.append(self._create_system_api_plan(req))

            elif req.data_source == "daily":
                if req.data_scope == DataScope.POINT:
                    api_call_plans.append(self._create_daily_point_api_plan(req))
                elif req.data_scope == DataScope.RANGE:
                    api_call_plans.extend(self._create_daily_range_api_plans(req))

            elif req.data_source == "product":
                api_call_plans.append(self._create_product_api_plan(req))

            elif req.data_source == "user_daily":
                if req.data_scope == DataScope.RANGE:
                    api_call_plans.extend(self._create_user_daily_range_api_plans(req))

            elif req.data_source == "product_end_interval":
                api_call_plans.append(self._create_product_end_interval_api_plan(req))

            elif req.data_source == "comprehensive":
                # 综合数据需求，需要多个API调用
                api_call_plans.extend(self._create_comprehensive_api_plans(req))

        # 添加依赖关系分析
        self._analyze_api_dependencies(api_call_plans)

        logger.info(f"🔗 生成API调用计划: {len(api_call_plans)}个")
        return api_call_plans

    def _create_system_api_plan(self, req: DataRequirement) -> APICallPlan:
        """创建系统API调用计划"""

        api_info = self.api_catalog["system"]

        return APICallPlan(
            api_endpoint=api_info["endpoint"],
            call_method=api_info["method"],
            parameters={},
            priority=req.priority,
            estimated_time=api_info["typical_response_time"],
            estimated_cost=self.cost_model["api_call_cost"]["system"],
            data_volume_expected=api_info["data_volume"],
            depends_on=[],
            cache_strategy="aggressive" if api_info["cache_recommended"] else "none",
            retry_strategy="standard",
            validation_rules=["non_empty_response", "positive_balance"]
        )

    def _create_daily_point_api_plan(self, req: DataRequirement) -> APICallPlan:
        """创建单日数据API调用计划"""

        api_info = self.api_catalog["daily"]
        target_date = req.processing_requirements.get("target_date", "")

        # 转换日期格式为API需要的YYYYMMDD
        if self.date_utils and target_date:
            api_date = self.date_utils.date_to_api_format(target_date)
        else:
            api_date = datetime.now().strftime("%Y%m%d")

        return APICallPlan(
            api_endpoint=api_info["endpoint"],
            call_method=api_info["method"],
            parameters={"date": api_date},
            priority=req.priority,
            estimated_time=api_info["typical_response_time"],
            estimated_cost=self.cost_model["api_call_cost"]["daily"],
            data_volume_expected=api_info["data_volume"],
            depends_on=[],
            cache_strategy="standard",
            retry_strategy="standard",
            validation_rules=["valid_date_format", "non_negative_amounts"]
        )

    def _create_daily_range_api_plans(self, req: DataRequirement) -> List[APICallPlan]:
        """创建日期范围数据API调用计划"""

        api_info = self.api_catalog["daily"]

        # 获取时间范围
        start_date = req.processing_requirements.get("start_date", "")
        end_date = req.processing_requirements.get("end_date", "")

        if not start_date or not end_date:
            # 使用默认范围
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=30)
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")

        # 生成日期列表
        if self.date_utils:
            dates = self.date_utils.generate_date_range(start_date, end_date, "api")
        else:
            # 降级方案
            dates = [datetime.now().strftime("%Y%m%d")]

        # 为每个日期创建API调用计划
        api_plans = []
        for date in dates:
            api_plans.append(APICallPlan(
                api_endpoint=api_info["endpoint"],
                call_method=api_info["method"],
                parameters={"date": date},
                priority=req.priority,
                estimated_time=api_info["typical_response_time"],
                estimated_cost=self.cost_model["api_call_cost"]["daily"],
                data_volume_expected=api_info["data_volume"],
                depends_on=[],
                cache_strategy="standard",
                retry_strategy="batch_retry",
                validation_rules=["valid_date_format", "non_negative_amounts"]
            ))

        return api_plans

    def _create_product_api_plan(self, req: DataRequirement) -> APICallPlan:
        """创建产品API调用计划"""

        api_info = self.api_catalog["product"]

        return APICallPlan(
            api_endpoint=api_info["endpoint"],
            call_method=api_info["method"],
            parameters={},
            priority=req.priority,
            estimated_time=api_info["typical_response_time"],
            estimated_cost=self.cost_model["api_call_cost"]["product"],
            data_volume_expected=api_info["data_volume"],
            depends_on=[],
            cache_strategy="extended" if api_info["cache_recommended"] else "none",
            retry_strategy="standard",
            validation_rules=["product_list_not_empty", "valid_product_data"]
        )

    def _create_user_daily_range_api_plans(self, req: DataRequirement) -> List[APICallPlan]:
        """创建用户每日数据范围API调用计划"""

        api_info = self.api_catalog["user_daily"]

        # 简化处理，只获取最近30天的用户数据
        api_plans = []
        for i in range(30):
            date_dt = datetime.now() - timedelta(days=i)
            api_date = date_dt.strftime("%Y%m%d")

            api_plans.append(APICallPlan(
                api_endpoint=api_info["endpoint"],
                call_method=api_info["method"],
                parameters={"date": api_date},
                priority=req.priority,
                estimated_time=api_info["typical_response_time"],
                estimated_cost=self.cost_model["api_call_cost"]["user_daily"],
                data_volume_expected=api_info["data_volume"],
                depends_on=[],
                cache_strategy="standard",
                retry_strategy="batch_retry",
                validation_rules=["valid_user_data"]
            ))

        return api_plans[:10]  # 限制为最近10天以控制成本

    def _create_product_end_interval_api_plan(self, req: DataRequirement) -> APICallPlan:
        """创建产品区间到期API调用计划"""

        api_info = self.api_catalog["product_end_interval"]

        start_date = req.processing_requirements.get("start_date", "")
        end_date = req.processing_requirements.get("end_date", "")

        # 转换为API格式
        if self.date_utils and start_date and end_date:
            api_start = self.date_utils.date_to_api_format(start_date)
            api_end = self.date_utils.date_to_api_format(end_date)
        else:
            # 默认未来一周
            today = datetime.now()
            week_end = today + timedelta(days=7)
            api_start = today.strftime("%Y%m%d")
            api_end = week_end.strftime("%Y%m%d")

        return APICallPlan(
            api_endpoint=api_info["endpoint"],
            call_method=api_info["method"],
            parameters={"start_date": api_start, "end_date": api_end},
            priority=req.priority,
            estimated_time=api_info["typical_response_time"],
            estimated_cost=self.cost_model["api_call_cost"]["product_end_interval"],
            data_volume_expected=api_info["data_volume"],
            depends_on=[],
            cache_strategy="extended",
            retry_strategy="standard",
            validation_rules=["valid_date_range", "non_negative_expiry_amounts"]
        )

    def _create_comprehensive_api_plans(self, req: DataRequirement) -> List[APICallPlan]:
        """创建综合数据API调用计划"""

        # 综合数据需求包括多个基础API调用
        comprehensive_plans = []

        # 添加系统数据
        comprehensive_plans.append(self._create_system_api_plan(req))

        # 添加产品数据
        comprehensive_plans.append(self._create_product_api_plan(req))

        # 添加最近的每日数据
        recent_req = DataRequirement(
            requirement_id="comprehensive_recent_daily",
            requirement_type="recent_daily_for_comprehensive",
            data_source="daily",
            data_scope=DataScope.RANGE,
            freshness_requirement=req.freshness_requirement,
            priority=req.priority,
            quality_threshold=req.quality_threshold,
            fallback_options=req.fallback_options,
            processing_requirements={
                "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d")
            },
            business_justification="综合分析的最近数据组件"
        )
        comprehensive_plans.extend(self._create_daily_range_api_plans(recent_req))

        return comprehensive_plans

    def _analyze_api_dependencies(self, api_call_plans: List[APICallPlan]):
        """分析API调用依赖关系"""

        # 简单的依赖关系分析：
        # 1. 系统数据通常是其他分析的基础
        # 2. 产品数据在到期分析之前需要获取

        system_calls = [plan for plan in api_call_plans if "system" in plan.call_method]
        product_calls = [plan for plan in api_call_plans if
                         "product" in plan.call_method and "end" not in plan.call_method]
        expiry_calls = [plan for plan in api_call_plans if "product_end" in plan.call_method]

        # 设置依赖关系
        for expiry_call in expiry_calls:
            if product_calls:
                expiry_call.depends_on = [product_calls[0].call_method]

        # 其他调用可以并行执行
        logger.info("🔗 API依赖关系分析完成")

    # ============= 执行策略优化层 =============

    async def _optimize_execution_strategy(self, api_call_plans: List[APICallPlan]) -> Dict[str, Any]:
        """优化执行策略"""

        try:
            # 🎯 分析并行执行可能性
            parallel_groups = self._identify_parallel_groups(api_call_plans)

            # 🎯 生成执行序列
            execution_sequence = self._generate_execution_sequence(api_call_plans, parallel_groups)

            # 🎯 优化批量调用
            batch_optimizations = self._identify_batch_opportunities(api_call_plans)

            # 🎯 缓存策略优化
            cache_strategy = self._optimize_cache_strategy(api_call_plans)

            return {
                "execution_sequence": execution_sequence,
                "parallel_groups": parallel_groups,
                "batch_optimizations": batch_optimizations,
                "cache_strategy": cache_strategy,
                "estimated_total_time": self._calculate_optimized_time(api_call_plans, parallel_groups),
                "optimization_notes": self._generate_optimization_notes(parallel_groups, batch_optimizations)
            }

        except Exception as e:
            logger.error(f"执行策略优化失败: {str(e)}")
            # 返回基础策略
            return {
                "execution_sequence": [plan.call_method for plan in api_call_plans],
                "parallel_groups": [],
                "batch_optimizations": {},
                "cache_strategy": "standard",
                "estimated_total_time": sum(plan.estimated_time for plan in api_call_plans),
                "optimization_notes": ["使用基础顺序执行策略"]
            }

    def _identify_parallel_groups(self, api_call_plans: List[APICallPlan]) -> List[List[str]]:
        """识别可并行执行的API组"""

        parallel_groups = []
        processed = set()

        for plan in api_call_plans:
            if plan.call_method in processed:
                continue

            # 找到可以与当前plan并行执行的其他plan
            parallel_group = [plan.call_method]
            processed.add(plan.call_method)

            for other_plan in api_call_plans:
                if (other_plan.call_method not in processed and
                        not other_plan.depends_on and
                        not plan.depends_on and
                        other_plan.call_method != plan.call_method):
                    parallel_group.append(other_plan.call_method)
                    processed.add(other_plan.call_method)

            if len(parallel_group) > 1:
                parallel_groups.append(parallel_group)

        return parallel_groups

    def _generate_execution_sequence(self, api_call_plans: List[APICallPlan],
                                     parallel_groups: List[List[str]]) -> List[str]:
        """生成优化的执行序列"""

        sequence = []

        # 按优先级排序
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "optional": 4}
        sorted_plans = sorted(api_call_plans, key=lambda x: priority_order[x.priority.value])

        # 构建执行序列
        processed = set()

        for plan in sorted_plans:
            if plan.call_method not in processed:
                sequence.append(plan.call_method)
                processed.add(plan.call_method)

        return sequence

    def _identify_batch_opportunities(self, api_call_plans: List[APICallPlan]) -> Dict[str, Any]:
        """识别批量调用机会"""

        batch_opportunities = {}

        # 按API端点分组
        endpoint_groups = {}
        for plan in api_call_plans:
            endpoint = plan.api_endpoint
            if endpoint not in endpoint_groups:
                endpoint_groups[endpoint] = []
            endpoint_groups[endpoint].append(plan)

        # 识别批量机会
        for endpoint, plans in endpoint_groups.items():
            if len(plans) > 1:
                batch_opportunities[endpoint] = {
                    "call_count": len(plans),
                    "estimated_savings": len(plans) * 0.1,  # 假设每个调用节省0.1秒
                    "batch_method": "parallel_batch" if len(plans) <= 10 else "sequential_batch"
                }

        return batch_opportunities

    def _optimize_cache_strategy(self, api_call_plans: List[APICallPlan]) -> str:
        """优化缓存策略"""

        cache_beneficial_count = sum(1 for plan in api_call_plans if plan.cache_strategy != "none")
        total_plans = len(api_call_plans)

        if cache_beneficial_count / total_plans > 0.7:
            return "aggressive_caching"
        elif cache_beneficial_count / total_plans > 0.3:
            return "standard_caching"
        else:
            return "minimal_caching"

    def _calculate_optimized_time(self, api_call_plans: List[APICallPlan],
                                  parallel_groups: List[List[str]]) -> float:
        """计算优化后的总执行时间"""

        total_time = 0
        processed = set()

        # 计算并行组的时间
        for group in parallel_groups:
            group_times = []
            for call_method in group:
                plan = next((p for p in api_call_plans if p.call_method == call_method), None)
                if plan:
                    group_times.append(plan.estimated_time)
                    processed.add(call_method)

            if group_times:
                total_time += max(group_times)  # 并行执行取最长时间

        # 计算剩余顺序执行的时间
        for plan in api_call_plans:
            if plan.call_method not in processed:
                total_time += plan.estimated_time

        return total_time

    def _generate_optimization_notes(self, parallel_groups: List[List[str]],
                                     batch_optimizations: Dict[str, Any]) -> List[str]:
        """生成优化建议"""

        notes = []

        if parallel_groups:
            parallel_count = sum(len(group) for group in parallel_groups)
            notes.append(f"识别到{len(parallel_groups)}个并行组，共{parallel_count}个API调用可并行执行")

        if batch_optimizations:
            total_savings = sum(opt["estimated_savings"] for opt in batch_optimizations.values())
            notes.append(f"批量调用优化可节省约{total_savings:.1f}秒")

        notes.append("建议启用智能缓存以提高性能")

        return notes

    # ============= 成本分析和最终构建层 =============

    def _analyze_cost_and_optimize(self, api_call_plans: List[APICallPlan],
                                   execution_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """成本效益分析和优化"""

        # 计算基础成本
        total_cost = sum(plan.estimated_cost for plan in api_call_plans)

        # 应用批量优化折扣
        batch_count = len(api_call_plans)
        if batch_count >= 50:
            cost_multiplier = self.cost_model["batch_cost_multiplier"]["50+"]
        elif batch_count >= 21:
            cost_multiplier = self.cost_model["batch_cost_multiplier"]["21-50"]
        elif batch_count >= 6:
            cost_multiplier = self.cost_model["batch_cost_multiplier"]["6-20"]
        else:
            cost_multiplier = self.cost_model["batch_cost_multiplier"]["1-5"]

        optimized_cost = total_cost * cost_multiplier

        # 缓存节省
        cache_savings = total_cost * self.cost_model["cache_cost_benefit"]

        # 并行执行效率提升
        parallel_efficiency = self.cost_model["parallel_efficiency"]

        return {
            "base_cost": total_cost,
            "optimized_cost": optimized_cost,
            "cost_savings": total_cost - optimized_cost + cache_savings,
            "efficiency_gain": parallel_efficiency,
            "optimization_recommendations": [
                "启用智能缓存",
                "使用并行执行",
                "考虑批量调用优化"
            ]
        }

    def _build_acquisition_plan(self, query_summary: Dict[str, Any],
                                requirements: List[DataRequirement],
                                api_call_plans: List[APICallPlan],
                                execution_strategy: Dict[str, Any],
                                cost_analysis: Dict[str, Any]) -> DataAcquisitionPlan:
        """构建最终的数据获取计划"""

        # 计算计划置信度
        plan_confidence = self._calculate_plan_confidence(requirements, api_call_plans)

        # 生成成功标准
        success_criteria = {
            "minimum_critical_data": "所有CRITICAL优先级数据必须成功获取",
            "data_quality_threshold": 0.8,
            "maximum_acceptable_failures": len([p for p in api_call_plans if p.priority.value in ["low", "optional"]]),
            "response_time_target": execution_strategy["estimated_total_time"] * 1.2
        }

        # 生成降级策略
        fallback_strategies = {
            "api_failure_fallback": "使用缓存数据或降级到基础数据集",
            "timeout_handling": "优先获取CRITICAL数据，跳过OPTIONAL数据",
            "data_quality_fallback": "降低质量阈值但保证核心业务逻辑",
            "complete_failure_fallback": "返回基础系统状态数据"
        }

        return DataAcquisitionPlan(
            plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_analysis_summary=query_summary,
            data_requirements=requirements,
            api_call_plans=api_call_plans,
            execution_sequence=execution_strategy["execution_sequence"],
            parallel_groups=execution_strategy["parallel_groups"],
            total_estimated_time=execution_strategy["estimated_total_time"],
            total_estimated_cost=cost_analysis["optimized_cost"],
            success_criteria=success_criteria,
            fallback_strategies=fallback_strategies,
            optimization_notes=execution_strategy["optimization_notes"],
            plan_confidence=plan_confidence
        )

    def _calculate_plan_confidence(self, requirements: List[DataRequirement],
                                   api_call_plans: List[APICallPlan]) -> float:
        """计算计划置信度"""

        # 基于数据需求的重要性和API调用的可靠性计算置信度
        total_weight = 0
        weighted_confidence = 0

        for req in requirements:
            weight = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4, "optional": 0.2}[req.priority.value]
            confidence = req.quality_threshold

            total_weight += weight
            weighted_confidence += weight * confidence

        return weighted_confidence / total_weight if total_weight > 0 else 0.5

    def _create_fallback_plan(self, query_analysis_result: Any, error: str) -> DataAcquisitionPlan:
        """创建降级数据获取计划"""

        logger.warning(f"创建降级计划: {error}")

        # 创建基础的系统数据需求
        fallback_requirement = DataRequirement(
            requirement_id="fallback_system_data",
            requirement_type="basic_system_status",
            data_source="system",
            data_scope=DataScope.POINT,
            freshness_requirement=DataFreshness.CURRENT,
            priority=DataPriority.CRITICAL,
            quality_threshold=0.7,
            fallback_options=[],
            processing_requirements={},
            business_justification="降级方案的基础数据"
        )

        # 创建基础API调用计划
        fallback_api_plan = self._create_system_api_plan(fallback_requirement)

        return DataAcquisitionPlan(
            plan_id=f"fallback_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_analysis_summary={"error": error, "fallback": True},
            data_requirements=[fallback_requirement],
            api_call_plans=[fallback_api_plan],
            execution_sequence=[fallback_api_plan.call_method],
            parallel_groups=[],
            total_estimated_time=fallback_api_plan.estimated_time,
            total_estimated_cost=fallback_api_plan.estimated_cost,
            success_criteria={"minimum_data": "基础系统状态"},
            fallback_strategies={"error_handling": "返回错误信息和基础数据"},
            optimization_notes=["这是错误降级计划"],
            plan_confidence=0.3
        )

    # ============= 工具方法 =============

    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return self.analysis_stats.copy()

    def validate_acquisition_plan(self, plan: DataAcquisitionPlan) -> Dict[str, Any]:
        """验证数据获取计划"""

        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }

        # 检查关键数据需求
        critical_requirements = [req for req in plan.data_requirements if req.priority == DataPriority.CRITICAL]
        if not critical_requirements:
            validation_result["warnings"].append("没有CRITICAL优先级的数据需求")

        # 检查API调用计划
        if not plan.api_call_plans:
            validation_result["is_valid"] = False
            validation_result["issues"].append("没有API调用计划")

        # 检查时间预估合理性
        if plan.total_estimated_time > 300:  # 5分钟
            validation_result["warnings"].append("预估执行时间过长")

        return validation_result


# ============= 工厂函数 =============

def create_data_requirements_analyzer(claude_client=None, gpt_client=None) -> DataRequirementsAnalyzer:
    """
    创建数据需求分析器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        DataRequirementsAnalyzer: 数据需求分析器实例
    """
    return DataRequirementsAnalyzer(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建数据需求分析器
    analyzer = create_data_requirements_analyzer()

    print("=== 数据需求分析器测试 ===")

    # 模拟查询分析结果
    from dataclasses import dataclass
    from enum import Enum

    @dataclass
    class MockQueryResult:
        original_query: str = "过去30天入金趋势分析"
        complexity: Any = None
        query_type: Any = None
        business_scenario: Any = None
        confidence_score: float = 0.85
        time_requirements: Dict = None
        date_parse_result: Any = None
        business_parameters: Dict = None
        calculation_requirements: Dict = None

        def __post_init__(self):
            if self.time_requirements is None:
                self.time_requirements = {"complexity": "medium"}
            if self.business_parameters is None:
                self.business_parameters = {}
            if self.calculation_requirements is None:
                self.calculation_requirements = {}

    # 创建模拟的查询结果
    mock_result = MockQueryResult()

    # 分析数据需求
    acquisition_plan = await analyzer.analyze_data_requirements(mock_result)

    print(f"计划ID: {acquisition_plan.plan_id}")
    print(f"数据需求数量: {len(acquisition_plan.data_requirements)}")
    print(f"API调用数量: {len(acquisition_plan.api_call_plans)}")
    print(f"预估时间: {acquisition_plan.total_estimated_time:.1f}秒")
    print(f"计划置信度: {acquisition_plan.plan_confidence:.2f}")

    # 验证计划
    validation = analyzer.validate_acquisition_plan(acquisition_plan)
    print(f"计划验证: {'通过' if validation['is_valid'] else '失败'}")

    # 统计信息
    stats = analyzer.get_analysis_stats()
    print(f"总分析次数: {stats['total_analyses']}")


if __name__ == "__main__":
    asyncio.run(main())