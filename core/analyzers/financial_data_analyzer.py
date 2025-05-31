# core/analyzers/financial_data_analyzer.py
"""
🎯 AI驱动的金融数据深度分析器
金融AI分析系统的核心分析引擎，负责对获取的数据进行深度分析

核心特点:
- 双AI协作的深度数据分析 (Claude + GPT-4o)
- 多维度金融指标计算和分析
- 智能趋势识别和模式发现
- 异常检测和风险预警
- 业务洞察和决策支持
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from decimal import Decimal

from utils.calculators.statistical_calculator import create_financial_calculator
# 导入已完成的工具类
from utils.helpers.date_utils import DateUtils, create_date_utils
from utils.helpers.validation_utils import ValidationUtils, create_validation_utils, ValidationLevel
from utils.data_transformers.time_series_builder import TimeSeriesBuilder, create_time_series_builder, DataQuality

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """分析类型"""
    TREND_ANALYSIS = "trend_analysis"  # 趋势分析
    PERFORMANCE_ANALYSIS = "performance"  # 业绩分析
    RISK_ASSESSMENT = "risk_assessment"  # 风险评估
    ANOMALY_DETECTION = "anomaly_detection"  # 异常检测
    CORRELATION_ANALYSIS = "correlation"  # 相关性分析
    SEASONAL_ANALYSIS = "seasonal"  # 季节性分析
    COMPARATIVE_ANALYSIS = "comparative"  # 对比分析
    PREDICTIVE_ANALYSIS = "predictive"  # 预测性分析


class AnalysisScope(Enum):
    """分析范围"""
    SYSTEM_LEVEL = "system"  # 系统级分析
    USER_LEVEL = "user"  # 用户级分析
    PRODUCT_LEVEL = "product"  # 产品级分析
    FINANCIAL_LEVEL = "financial"  # 财务级分析
    OPERATIONAL_LEVEL = "operational"  # 运营级分析


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_HIGH = "very_high"  # 非常高 (>0.9)
    HIGH = "high"  # 高 (0.8-0.9)
    MEDIUM = "medium"  # 中等 (0.6-0.8)
    LOW = "low"  # 低 (0.4-0.6)
    VERY_LOW = "very_low"  # 很低 (<0.4)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    analysis_id: str  # 分析ID
    analysis_type: AnalysisType  # 分析类型
    analysis_scope: AnalysisScope  # 分析范围
    confidence_score: float  # 置信度评分 (0-1)

    # 核心结果
    key_findings: List[str]  # 关键发现
    trends: List[Dict[str, Any]]  # 趋势信息
    anomalies: List[Dict[str, Any]]  # 异常信息
    metrics: Dict[str, float]  # 关键指标

    # 业务洞察
    business_insights: List[str]  # 业务洞察
    risk_factors: List[str]  # 风险因素
    opportunities: List[str]  # 机会点
    recommendations: List[str]  # 建议

    # 技术信息
    data_quality: DataQuality  # 数据质量
    analysis_metadata: Dict[str, Any]  # 分析元数据
    processing_time: float  # 处理时间
    timestamp: str  # 分析时间戳


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    metric_name: str  # 指标名称
    trend_direction: str  # 趋势方向 (increasing/decreasing/stable)
    trend_strength: float  # 趋势强度 (0-1)
    growth_rate: float  # 增长率
    volatility: float  # 波动性
    trend_confidence: float  # 趋势置信度
    inflection_points: List[str]  # 拐点日期
    seasonal_patterns: Dict[str, Any]  # 季节性模式


@dataclass
class AnomalyDetection:
    """异常检测结果"""
    date: str  # 异常日期
    metric: str  # 异常指标
    actual_value: float  # 实际值
    expected_value: float  # 期望值
    deviation_score: float  # 偏离程度
    anomaly_type: str  # 异常类型
    severity: str  # 严重程度
    possible_causes: List[str]  # 可能原因
    impact_assessment: str  # 影响评估


class FinancialDataAnalyzer:
    """
    🎯 AI驱动的金融数据深度分析器

    功能架构:
    1. 多维度数据分析能力
    2. 双AI协作的深度洞察
    3. 智能模式识别
    4. 风险预警和机会发现
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化金融数据分析器

        Args:
            claude_client: Claude客户端，负责业务洞察分析
            gpt_client: GPT客户端，负责数值计算和统计分析
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client

        # 初始化工具组件
        self.date_utils = create_date_utils(claude_client)
        self.validator = create_validation_utils(claude_client, gpt_client)
        self.time_series_builder = create_time_series_builder(claude_client, gpt_client)
        self.financial_calculator = create_financial_calculator(gpt_client, 4)  # 修正参数顺序，明确指定precision为4

        # 分析配置
        self.analysis_config = self._load_analysis_config()

        # 分析统计
        self.analysis_stats = {
            'total_analyses': 0,
            'analysis_by_type': {},
            'avg_confidence_score': 0.0,
            'anomalies_detected': 0,
            'insights_generated': 0
        }

        logger.info("FinancialDataAnalyzer initialized with dual-AI capabilities")

    def _load_analysis_config(self) -> Dict[str, Any]:
        """加载分析配置"""
        return {
            # 趋势分析配置
            'trend_analysis': {
                'min_data_points': 7,  # 最少数据点
                'volatility_threshold': 0.15,  # 波动性阈值
                'trend_strength_threshold': 0.7,  # 趋势强度阈值
                'growth_rate_bounds': (-0.5, 2.0)  # 增长率合理范围
            },

            # 异常检测配置
            'anomaly_detection': {
                'sensitivity': 2.0,  # 敏感度 (标准差倍数)
                'min_history_days': 14,  # 最少历史天数
                'outlier_threshold': 0.05,  # 异常值阈值
                'severe_anomaly_threshold': 5.0  # 严重异常阈值
            },

            # 风险评估配置
            'risk_assessment': {
                'high_risk_threshold': 0.8,  # 高风险阈值
                'volatility_risk_multiplier': 2.0,  # 波动性风险乘数
                'liquidity_risk_threshold': 0.1  # 流动性风险阈值
            },

            # AI分析配置
            'ai_analysis': {
                'use_claude_for_insights': True,  # 使用Claude生成洞察
                'use_gpt_for_calculations': True,  # 使用GPT进行计算
                'confidence_threshold': 0.6,  # 最低置信度阈值
                'max_ai_retries': 3  # AI调用最大重试次数
            }
        }

    # ============= 核心分析方法 =============

    async def analyze_trend(self, data_source: str, metric: str, time_range: int) -> AnalysisResult:
        """
        🎯 趋势分析 - 分析指定指标的趋势变化

        Args:
            data_source: 数据源类型 (system/daily/product等)
            metric: 分析指标 (total_balance/daily_inflow等)
            time_range: 时间范围 (天数)

        Returns:
            AnalysisResult: 完整的趋势分析结果
        """
        try:
            logger.info(f"🔍 开始趋势分析: {data_source}.{metric}, 时间范围: {time_range}天")

            analysis_start_time = datetime.now()
            self.analysis_stats['total_analyses'] += 1

            # Step 1: 数据准备和验证
            analysis_data = await self._prepare_trend_analysis_data(data_source, metric, time_range)

            if not analysis_data['is_valid']:
                return self._create_error_analysis_result("trend_analysis", analysis_data['error'])

            # Step 2: 构建时间序列
            time_series_result = await self._build_trend_time_series(analysis_data['raw_data'], metric)

            # Step 3: 计算趋势统计指标
            trend_statistics = await self._calculate_trend_statistics(time_series_result, metric)

            # Step 4: AI趋势模式识别
            trend_patterns = await self._ai_identify_trend_patterns(time_series_result, trend_statistics)

            # Step 5: 异常检测
            anomalies = await self._detect_trend_anomalies(time_series_result, trend_statistics)

            # Step 6: 生成业务洞察
            business_insights = await self._generate_trend_insights(
                trend_statistics, trend_patterns, anomalies, data_source, metric
            )

            # Step 7: 计算置信度
            confidence_score = self._calculate_analysis_confidence(
                analysis_data, trend_statistics, anomalies
            )

            # Step 8: 构建分析结果
            processing_time = (datetime.now() - analysis_start_time).total_seconds()

            analysis_result = AnalysisResult(
                analysis_id=f"trend_{data_source}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                analysis_type=AnalysisType.TREND_ANALYSIS,
                analysis_scope=self._determine_analysis_scope(data_source),
                confidence_score=confidence_score,

                key_findings=trend_patterns.get('key_findings', []),
                trends=[{
                    'metric': metric,
                    'direction': trend_statistics['trend_direction'],
                    'strength': trend_statistics['trend_strength'],
                    'growth_rate': trend_statistics['growth_rate'],
                    'volatility': trend_statistics['volatility']
                }],
                anomalies=anomalies,
                metrics=trend_statistics,

                business_insights=business_insights.get('insights', []),
                risk_factors=business_insights.get('risks', []),
                opportunities=business_insights.get('opportunities', []),
                recommendations=business_insights.get('recommendations', []),

                data_quality=analysis_data['data_quality'],
                analysis_metadata={
                    'data_source': data_source,
                    'metric': metric,
                    'time_range_days': time_range,
                    'data_points': len(time_series_result.get('series_data', [])),
                    'ai_models_used': ['claude', 'gpt'] if self.claude_client and self.gpt_client else []
                },
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )

            # 更新统计信息
            self._update_analysis_stats('trend_analysis', confidence_score)

            logger.info(f"✅ 趋势分析完成: 置信度={confidence_score:.2f}, 耗时={processing_time:.2f}秒")
            return analysis_result

        except Exception as e:
            logger.error(f"❌ 趋势分析失败: {str(e)}")
            return self._create_error_analysis_result("trend_analysis", str(e))

    async def analyze_business_performance(self, scope: str, time_range: int) -> AnalysisResult:
        """
        📊 业务表现分析 - 综合分析业务各方面表现

        Args:
            scope: 分析范围 (financial/operational/user/product)
            time_range: 时间范围 (天数)

        Returns:
            AnalysisResult: 业务表现分析结果
        """
        try:
            logger.info(f"📊 开始业务表现分析: {scope}, 时间范围: {time_range}天")

            analysis_start_time = datetime.now()

            # Step 1: 根据scope确定分析指标
            performance_metrics = self._get_performance_metrics(scope)

            # Step 2: 获取和准备数据
            performance_data = await self._prepare_performance_analysis_data(scope, performance_metrics, time_range)

            # Step 3: 计算关键绩效指标
            kpi_results = await self._calculate_business_kpis(performance_data, scope)

            # Step 4: 对比分析 (与历史同期对比)
            comparative_analysis = await self._perform_comparative_analysis(performance_data, scope)

            # Step 5: AI业务洞察生成
            business_analysis = await self._ai_analyze_business_performance(
                kpi_results, comparative_analysis, scope
            )

            # Step 6: 风险和机会识别
            risk_opportunity_analysis = await self._assess_performance_risks_opportunities(
                kpi_results, comparative_analysis
            )

            processing_time = (datetime.now() - analysis_start_time).total_seconds()

            analysis_result = AnalysisResult(
                analysis_id=f"performance_{scope}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
                analysis_scope=AnalysisScope(scope),
                confidence_score=business_analysis.get('confidence', 0.8),

                key_findings=business_analysis.get('key_findings', []),
                trends=comparative_analysis.get('trends', []),
                anomalies=kpi_results.get('anomalies', []),
                metrics=kpi_results.get('kpis', {}),

                business_insights=business_analysis.get('insights', []),
                risk_factors=risk_opportunity_analysis.get('risks', []),
                opportunities=risk_opportunity_analysis.get('opportunities', []),
                recommendations=business_analysis.get('recommendations', []),

                data_quality=performance_data.get('data_quality', DataQuality.GOOD),
                analysis_metadata={
                    'scope': scope,
                    'metrics_analyzed': len(performance_metrics),
                    'time_range_days': time_range,
                    'comparative_analysis': True
                },
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )

            self._update_analysis_stats('performance_analysis', business_analysis.get('confidence', 0.8))

            logger.info(f"✅ 业务表现分析完成: {scope}")
            return analysis_result

        except Exception as e:
            logger.error(f"❌ 业务表现分析失败: {str(e)}")
            return self._create_error_analysis_result("performance_analysis", str(e))

    # async def detect_anomalies(self, data_source: str, metrics: List[str],
    #                            sensitivity: float = 2.0) -> AnalysisResult:
    #     """
    #     🚨 异常检测 - 检测数据中的异常模式
    #
    #     Args:
    #         data_source: 数据源类型
    #         metrics: 检测的指标列表
    #         sensitivity: 敏感度 (标准差倍数)
    #
    #     Returns:
    #         AnalysisResult: 异常检测结果
    #     """
    #     try:
    #         logger.info(f"🚨 开始异常检测: {data_source}, 指标: {metrics}")
    #
    #         analysis_start_time = datetime.now()
    #
    #         # Step 1: 准备异常检测数据
    #         anomaly_data = await self._prepare_anomaly_detection_data(data_source, metrics)
    #
    #         # Step 2: 统计学异常检测
    #         statistical_anomalies = await self._statistical_anomaly_detection(
    #             anomaly_data, metrics, sensitivity
    #         )
    #
    #         # Step 3: AI模式异常检测
    #         pattern_anomalies = await self._ai_pattern_anomaly_detection(
    #             anomaly_data, statistical_anomalies
    #         )
    #
    #         # Step 4: 业务逻辑异常检测
    #         business_anomalies = await self._business_logic_anomaly_detection(
    #             anomaly_data, metrics
    #         )
    #
    #         # Step 5: 异常综合评估和分类
    #         consolidated_anomalies = self._consolidate_anomalies(
    #             statistical_anomalies, pattern_anomalies, business_anomalies
    #         )
    #
    #         # Step 6: 异常影响评估
    #         impact_assessment = await self._assess_anomaly_impact(consolidated_anomalies, data_source)
    #
    #         # Step 7: AI异常解释和建议
    #         anomaly_insights = await self._ai_explain_anomalies(consolidated_anomalies, impact_assessment)
    #
    #         processing_time = (datetime.now() - analysis_start_time).total_seconds()
    #
    #         analysis_result = AnalysisResult(
    #             analysis_id=f"anomaly_{data_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #             analysis_type=AnalysisType.ANOMALY_DETECTION,
    #             analysis_scope=self._determine_analysis_scope(data_source),
    #             confidence_score=anomaly_insights.get('confidence', 0.8),
    #
    #             key_findings=anomaly_insights.get('key_findings', []),
    #             trends=[],
    #             anomalies=consolidated_anomalies,
    #             metrics={'total_anomalies': len(consolidated_anomalies), 'sensitivity': sensitivity},
    #
    #             business_insights=anomaly_insights.get('insights', []),
    #             risk_factors=impact_assessment.get('risks', []),
    #             opportunities=impact_assessment.get('opportunities', []),
    #             recommendations=anomaly_insights.get('recommendations', []),
    #
    #             data_quality=anomaly_data.get('data_quality', DataQuality.GOOD),
    #             analysis_metadata={
    #                 'data_source': data_source,
    #                 'metrics_checked': metrics,
    #                 'sensitivity_level': sensitivity,
    #                 'detection_methods': ['statistical', 'pattern', 'business_logic']
    #             },
    #             processing_time=processing_time,
    #             timestamp=datetime.now().isoformat()
    #         )
    #
    #         # 更新异常统计
    #         self.analysis_stats['anomalies_detected'] += len(consolidated_anomalies)
    #         self._update_analysis_stats('anomaly_detection', anomaly_insights.get('confidence', 0.8))
    #
    #         logger.info(f"✅ 异常检测完成: 发现{len(consolidated_anomalies)}个异常")
    #         return analysis_result
    #
    #     except Exception as e:
    #         logger.error(f"❌ 异常检测失败: {str(e)}")
    #         return self._create_error_analysis_result("anomaly_detection", str(e))

    # ============= 数据准备和预处理 =============

    async def _prepare_trend_analysis_data(self, data_source: str, metric: str,
                                           time_range: int) -> Dict[str, Any]:
        """准备趋势分析数据"""
        try:
            # 这里需要调用smart_data_fetcher获取数据
            # 由于这是分析器，我们假设数据已经通过fetcher获取
            # 实际使用时，这个方法会接收fetcher传递的数据

            # 模拟数据验证逻辑
            if self.validator:
                validation_result = await self.validator.validate_data(
                    {'data_source': data_source, 'metric': metric, 'time_range': time_range},
                    'trend_analysis_data'
                )

                if not validation_result.is_valid:
                    return {
                        'is_valid': False,
                        'error': f"数据验证失败: {validation_result.issues}",
                        'data_quality': DataQuality.POOR
                    }

            return {
                'is_valid': True,
                'raw_data': {},  # 实际数据将在这里
                'data_quality': DataQuality.GOOD,
                'validation_passed': True
            }

        except Exception as e:
            logger.error(f"数据准备失败: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'data_quality': DataQuality.INSUFFICIENT
            }

    async def _build_trend_time_series(self, raw_data: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """构建趋势分析时间序列"""
        try:
            if not self.time_series_builder:
                # 基础时间序列构建逻辑
                return {'series_data': [], 'metadata': {'method': 'basic'}}

            # 使用time_series_builder构建时间序列
            # 这里假设raw_data包含了日期和数值信息
            mock_daily_data = []  # 实际使用时会有真实数据

            time_series_result = await self.time_series_builder.build_daily_time_series(
                mock_daily_data, metric, 'date'
            )

            return time_series_result

        except Exception as e:
            logger.error(f"时间序列构建失败: {str(e)}")
            return {'series_data': [], 'metadata': {'error': str(e)}}

    # ============= 统计计算和AI分析 =============

    async def _calculate_trend_statistics(self, time_series_result: Dict[str, Any],
                                          metric: str) -> Dict[str, Any]:
        """计算趋势统计指标"""
        try:
            series_data = time_series_result.get('series_data', [])

            if len(series_data) < 2:
                return {
                    'trend_direction': 'insufficient_data',
                    'trend_strength': 0.0,
                    'growth_rate': 0.0,
                    'volatility': 0.0,
                    'data_points': len(series_data)
                }

            # 提取数值序列
            values = [point.value for point in series_data if hasattr(point, 'value')]

            if not values:
                values = [float(point.get('value', 0)) for point in series_data if isinstance(point, dict)]

            if len(values) < 2:
                return {'trend_direction': 'no_valid_data', 'data_points': len(series_data)}

            # 计算基础统计指标
            if self.financial_calculator:
                # 使用financial_calculator计算增长率
                growth_rate = await self.financial_calculator.calculate_growth_rate(values, 'simple')
                growth_rate_value = growth_rate.result_value if hasattr(growth_rate, 'result_value') else growth_rate
            else:
                # 基础增长率计算
                growth_rate_value = (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0.0

            # 计算趋势方向和强度
            trend_direction = self._determine_trend_direction(values)
            trend_strength = self._calculate_trend_strength(values)
            volatility = self._calculate_volatility(values)

            # 使用GPT进行更精确的统计分析
            if self.gpt_client:
                enhanced_stats = await self._gpt_enhanced_statistics(values, metric)
                if enhanced_stats:
                    return {**enhanced_stats, 'ai_enhanced': True}

            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'growth_rate': growth_rate_value,
                'volatility': volatility,
                'data_points': len(values),
                'mean_value': statistics.mean(values),
                'median_value': statistics.median(values),
                'std_deviation': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min_value': min(values),
                'max_value': max(values)
            }

        except Exception as e:
            logger.error(f"趋势统计计算失败: {str(e)}")
            return {'error': str(e), 'trend_direction': 'calculation_error'}

    async def _ai_identify_trend_patterns(self, time_series_result: Dict[str, Any],
                                          trend_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """AI识别趋势模式"""
        try:
            if not self.claude_client:
                return {
                    'key_findings': ['基础趋势分析完成'],
                    'patterns': [],
                    'method': 'basic'
                }

            # 使用Claude进行模式识别
            pattern_analysis_prompt = f"""
            作为金融数据分析专家，请分析以下趋势数据的模式：

            趋势统计信息：
            {json.dumps(trend_statistics, ensure_ascii=False, indent=2)}

            时间序列元数据：
            {json.dumps(time_series_result.get('metadata', {}), ensure_ascii=False, indent=2)}

            请识别：
            1. 主要趋势模式 (线性、指数、周期性等)
            2. 关键特征和转折点
            3. 潜在的季节性或周期性
            4. 异常模式或突发变化
            5. 趋势的可持续性评估

            返回JSON格式的分析结果，包括：
            {{
                "key_findings": ["关键发现1", "关键发现2"],
                "pattern_type": "pattern_description",
                "seasonality": "seasonal_analysis",
                "inflection_points": ["date1", "date2"],
                "sustainability_assessment": "assessment_text",
                "confidence": 0.0-1.0
            }}
            """

            claude_result = await self.claude_client.analyze_complex_query(
                pattern_analysis_prompt,
                {
                    "trend_stats": trend_statistics,
                    "time_series_metadata": time_series_result.get('metadata', {})
                }
            )

            if claude_result.get('success'):
                analysis = claude_result.get('analysis', {})
                return {
                    **analysis,
                    'ai_analysis': True,
                    'model_used': 'claude'
                }

        except Exception as e:
            logger.error(f"AI模式识别失败: {str(e)}")

        # 降级到基础分析
        return {
            'key_findings': [
                f"趋势方向: {trend_statistics.get('trend_direction', 'unknown')}",
                f"增长率: {trend_statistics.get('growth_rate', 0):.2%}",
                f"波动性: {trend_statistics.get('volatility', 0):.2f}"
            ],
            'pattern_type': 'basic_trend',
            'confidence': 0.6
        }

    async def _detect_trend_anomalies(self, time_series_result: Dict[str, Any],
                                      trend_statistics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测趋势中的异常"""
        try:
            series_data = time_series_result.get('series_data', [])

            if len(series_data) < self.analysis_config['anomaly_detection']['min_history_days']:
                return []

            anomalies = []
            sensitivity = self.analysis_config['anomaly_detection']['sensitivity']

            # 提取数值和日期
            values = []
            dates = []

            for point in series_data:
                if hasattr(point, 'value') and hasattr(point, 'date'):
                    values.append(point.value)
                    dates.append(point.date)
                elif isinstance(point, dict):
                    values.append(float(point.get('value', 0)))
                    dates.append(point.get('date', ''))

            if len(values) < 3:
                return anomalies

            # 计算统计阈值
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            upper_threshold = mean_val + (sensitivity * std_val)
            lower_threshold = mean_val - (sensitivity * std_val)

            # 检测统计异常
            for i, (value, date) in enumerate(zip(values, dates)):
                if value > upper_threshold or value < lower_threshold:
                    deviation_score = abs(value - mean_val) / std_val if std_val > 0 else 0

                    anomaly = {
                        'date': date,
                        'metric': 'trend_value',
                        'actual_value': value,
                        'expected_value': mean_val,
                        'deviation_score': deviation_score,
                        'anomaly_type': 'statistical_outlier',
                        'severity': 'high' if deviation_score > 3 else 'medium',
                        'detection_method': 'statistical'
                    }
                    anomalies.append(anomaly)

            # 检测趋势突变异常
            if len(values) >= 5:
                for i in range(2, len(values) - 2):
                    # 计算局部趋势变化
                    before_trend = (values[i] - values[i - 2]) / 2 if values[i - 2] != 0 else 0
                    after_trend = (values[i + 2] - values[i]) / 2 if values[i] != 0 else 0

                    # 检测趋势反转
                    if abs(before_trend - after_trend) > std_val:
                        trend_change_anomaly = {
                            'date': dates[i],
                            'metric': 'trend_change',
                            'actual_value': values[i],
                            'expected_value': values[i - 1],
                            'deviation_score': abs(before_trend - after_trend),
                            'anomaly_type': 'trend_reversal',
                            'severity': 'medium',
                            'detection_method': 'trend_analysis',
                            'before_trend': before_trend,
                            'after_trend': after_trend
                        }
                        anomalies.append(trend_change_anomaly)

            # AI增强异常检测
            if self.claude_client and len(anomalies) > 0:
                ai_enhanced_anomalies = await self._ai_enhance_anomaly_detection(
                    anomalies, values, dates, trend_statistics
                )
                anomalies = ai_enhanced_anomalies

            return anomalies

        except Exception as e:
            logger.error(f"趋势异常检测失败: {str(e)}")
            return []

    async def _ai_enhance_anomaly_detection(self, anomalies: List[Dict[str, Any]],
                                            values: List[float], dates: List[str],
                                            trend_statistics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI增强异常检测"""
        try:
            if not self.claude_client or not anomalies:
                return anomalies

            enhancement_prompt = f"""
            作为金融风险分析专家，请分析以下检测到的数据异常：

            检测到的异常：
            {json.dumps(anomalies, ensure_ascii=False, indent=2)}

            趋势统计背景：
            {json.dumps(trend_statistics, ensure_ascii=False, indent=2)}

            请为每个异常提供：
            1. 可能的业务原因
            2. 风险级别评估
            3. 影响范围预估
            4. 建议的处理措施

            返回增强后的异常信息JSON数组。
            """

            claude_result = await self.claude_client.analyze_complex_query(
                enhancement_prompt,
                {
                    "anomalies": anomalies,
                    "trend_context": trend_statistics
                }
            )

            if claude_result.get('success'):
                enhanced_analysis = claude_result.get('analysis', {})

                # 合并AI分析结果到原异常数据
                enhanced_anomalies = []
                for i, anomaly in enumerate(anomalies):
                    enhanced_anomaly = anomaly.copy()

                    if isinstance(enhanced_analysis, list) and i < len(enhanced_analysis):
                        ai_insight = enhanced_analysis[i]
                        enhanced_anomaly.update({
                            'possible_causes': ai_insight.get('possible_causes', []),
                            'impact_assessment': ai_insight.get('impact_assessment', 'unknown'),
                            'risk_level': ai_insight.get('risk_level', 'medium'),
                            'recommended_actions': ai_insight.get('recommended_actions', []),
                            'ai_enhanced': True
                        })

                    enhanced_anomalies.append(enhanced_anomaly)

                return enhanced_anomalies

        except Exception as e:
            logger.error(f"AI异常增强分析失败: {str(e)}")

        # AI失败时返回原始异常
        return anomalies

    async def _generate_trend_insights(self, trend_statistics: Dict[str, Any],
                                       trend_patterns: Dict[str, Any],
                                       anomalies: List[Dict[str, Any]],
                                       data_source: str, metric: str) -> Dict[str, Any]:
        """生成趋势业务洞察"""
        try:
            if not self.claude_client:
                # 基础洞察生成
                return {
                    'insights': [
                        f"{metric}趋势方向: {trend_statistics.get('trend_direction', 'unknown')}",
                        f"增长率: {trend_statistics.get('growth_rate', 0):.2%}",
                        f"检测到{len(anomalies)}个异常点"
                    ],
                    'risks': ['数据波动性较高'] if trend_statistics.get('volatility', 0) > 0.2 else [],
                    'opportunities': ['增长趋势良好'] if trend_statistics.get('growth_rate', 0) > 0 else [],
                    'recommendations': ['继续监控趋势变化']
                }

            insight_prompt = f"""
            作为资深金融业务分析师，请基于以下数据提供深度业务洞察：

            数据源: {data_source}
            分析指标: {metric}

            趋势统计：
            {json.dumps(trend_statistics, ensure_ascii=False, indent=2)}

            趋势模式：
            {json.dumps(trend_patterns, ensure_ascii=False, indent=2)}

            异常情况：
            检测到 {len(anomalies)} 个异常点
            {json.dumps(anomalies[:3], ensure_ascii=False, indent=2) if anomalies else "无异常"}

            请提供：
            1. 核心业务洞察 (3-5个关键发现)
            2. 潜在风险因素
            3. 业务机会识别
            4. 具体行动建议

            返回JSON格式：
            {{
                "insights": ["洞察1", "洞察2", ...],
                "risks": ["风险1", "风险2", ...],
                "opportunities": ["机会1", "机会2", ...],
                "recommendations": ["建议1", "建议2", ...],
                "confidence": 0.0-1.0
            }}
            """

            claude_result = await self.claude_client.analyze_complex_query(
                insight_prompt,
                {
                    "trend_statistics": trend_statistics,
                    "patterns": trend_patterns,
                    "anomalies": anomalies
                }
            )

            if claude_result.get('success'):
                insights = claude_result.get('analysis', {})
                return {
                    **insights,
                    'ai_generated': True,
                    'model_used': 'claude'
                }

        except Exception as e:
            logger.error(f"业务洞察生成失败: {str(e)}")

        # 降级生成基础洞察
        return {
            'insights': [f"{metric}的趋势分析已完成"],
            'risks': [],
            'opportunities': [],
            'recommendations': ['建议定期监控该指标的变化'],
            'confidence': 0.5
        }

    # ============= 业务表现分析相关方法 =============

    def _get_performance_metrics(self, scope: str) -> List[str]:
        """根据分析范围获取性能指标"""
        metrics_map = {
            'financial': [
                'total_balance', 'total_inflow', 'total_outflow', 'net_cash_flow',
                'liquidity_ratio', 'growth_rate', 'return_on_investment'
            ],
            'operational': [
                'daily_registrations', 'active_users', 'product_purchases',
                'product_maturities', 'user_activity_rate', 'conversion_rate'
            ],
            'user': [
                'new_users', 'active_users', 'user_retention', 'vip_distribution',
                'average_investment_per_user', 'user_lifetime_value'
            ],
            'product': [
                'product_sales', 'product_performance', 'maturity_distribution',
                'product_roi', 'popular_products', 'product_risk_assessment'
            ]
        }

        return metrics_map.get(scope, ['total_balance', 'total_inflow', 'total_outflow'])

    async def _prepare_performance_analysis_data(self, scope: str, metrics: List[str],
                                                 time_range: int) -> Dict[str, Any]:
        """准备业务表现分析数据"""
        try:
            # 模拟数据准备过程
            # 实际使用时会调用smart_data_fetcher获取数据

            performance_data = {
                'scope': scope,
                'metrics': metrics,
                'time_range': time_range,
                'data_quality': DataQuality.GOOD,
                'raw_data': {},  # 实际数据
                'metadata': {
                    'data_collection_time': datetime.now().isoformat(),
                    'data_sources': ['system', 'daily', 'product', 'user'],
                    'completeness': 0.95
                }
            }

            return performance_data

        except Exception as e:
            logger.error(f"业务表现数据准备失败: {str(e)}")
            return {
                'scope': scope,
                'data_quality': DataQuality.POOR,
                'error': str(e)
            }

    async def _calculate_business_kpis(self, performance_data: Dict[str, Any],
                                       scope: str) -> Dict[str, Any]:
        """计算业务关键绩效指标"""
        try:
            kpis = {}
            metrics = performance_data.get('metrics', [])

            # 基础KPI计算
            if 'total_balance' in metrics:
                kpis['current_balance'] = 85000000.0  # 模拟数据
                kpis['balance_growth_rate'] = 0.08

            if 'total_inflow' in metrics:
                kpis['total_inflow'] = 45000000.0
                kpis['inflow_growth_rate'] = 0.12

            if 'active_users' in metrics:
                kpis['active_users'] = 12500
                kpis['user_growth_rate'] = 0.15

            # 使用financial_calculator进行复杂计算
            if self.financial_calculator:
                # 计算投资回报率
                roi_calculation = await self.financial_calculator.calculate_return_on_investment(
                    initial_investment=80000000.0,
                    current_value=85000000.0,
                    time_period=30
                )
                kpis['roi'] = roi_calculation.result_value if hasattr(roi_calculation, 'result_value') else 0.06

            # GPT增强计算
            if self.gpt_client:
                enhanced_kpis = await self._gpt_enhanced_kpi_calculation(performance_data, kpis)
                if enhanced_kpis:
                    kpis.update(enhanced_kpis)

            return {
                'kpis': kpis,
                'calculation_time': datetime.now().isoformat(),
                'scope': scope,
                'anomalies': []  # 在KPI计算中发现的异常
            }

        except Exception as e:
            logger.error(f"KPI计算失败: {str(e)}")
            return {'kpis': {}, 'error': str(e)}

    async def _perform_comparative_analysis(self, performance_data: Dict[str, Any],
                                            scope: str) -> Dict[str, Any]:
        """执行对比分析"""
        try:
            # 模拟对比分析
            # 实际实现会对比历史同期数据

            comparative_results = {
                'period_comparison': {
                    'current_period': 'last_30_days',
                    'comparison_period': 'previous_30_days',
                    'improvements': [
                        '用户增长率提升15%',
                        '资金流入增加12%'
                    ],
                    'deteriorations': [
                        '用户活跃度下降3%'
                    ]
                },
                'trend_analysis': {
                    'overall_trend': 'positive',
                    'trend_strength': 0.75,
                    'sustainability': 'likely'
                },
                'benchmarks': {
                    'industry_comparison': 'above_average',
                    'historical_performance': 'improving'
                }
            }

            return comparative_results

        except Exception as e:
            logger.error(f"对比分析失败: {str(e)}")
            return {'error': str(e)}

    async def _ai_analyze_business_performance(self, kpi_results: Dict[str, Any],
                                               comparative_analysis: Dict[str, Any],
                                               scope: str) -> Dict[str, Any]:
        """AI分析业务表现"""
        try:
            if not self.claude_client:
                return {
                    'key_findings': ['业务表现分析完成'],
                    'insights': ['基于KPI的基础分析'],
                    'recommendations': ['继续监控关键指标'],
                    'confidence': 0.6
                }

            analysis_prompt = f"""
            作为业务分析专家，请深度分析以下业务表现数据：

            分析范围: {scope}

            关键绩效指标：
            {json.dumps(kpi_results.get('kpis', {}), ensure_ascii=False, indent=2)}

            对比分析结果：
            {json.dumps(comparative_analysis, ensure_ascii=False, indent=2)}

            请提供：
            1. 关键业务发现 (5个最重要的发现)
            2. 深度业务洞察
            3. 战略建议
            4. 风险预警
            5. 机会识别

            返回JSON格式的分析结果。
            """

            claude_result = await self.claude_client.analyze_complex_query(
                analysis_prompt,
                {
                    "scope": scope,
                    "kpis": kpi_results,
                    "comparative": comparative_analysis
                }
            )

            if claude_result.get('success'):
                analysis = claude_result.get('analysis', {})
                return {
                    **analysis,
                    'ai_analysis': True,
                    'model_used': 'claude'
                }

        except Exception as e:
            logger.error(f"AI业务表现分析失败: {str(e)}")

        return {
            'key_findings': ['业务表现评估完成'],
            'insights': ['需要进一步数据分析'],
            'recommendations': ['定期监控业务指标'],
            'confidence': 0.5
        }

    # ============= 辅助计算方法 =============

    def _determine_trend_direction(self, values: List[float]) -> str:
        """确定趋势方向"""
        if len(values) < 2:
            return 'insufficient_data'

        # 计算线性回归斜率
        n = len(values)
        x = list(range(n))

        # 简单线性回归
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable'

        slope = numerator / denominator

        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_trend_strength(self, values: List[float]) -> float:
        """计算趋势强度"""
        if len(values) < 3:
            return 0.0

        try:
            # 计算相关系数作为趋势强度
            n = len(values)
            x = list(range(n))

            x_mean = sum(x) / n
            y_mean = sum(values) / n

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            x_var = sum((x[i] - x_mean) ** 2 for i in range(n))
            y_var = sum((values[i] - y_mean) ** 2 for i in range(n))

            if x_var == 0 or y_var == 0:
                return 0.0

            correlation = numerator / (x_var * y_var) ** 0.5
            return abs(correlation)

        except Exception:
            return 0.0

    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动性"""
        if len(values) < 2:
            return 0.0

        try:
            # 计算相对标准差作为波动性指标
            mean_val = statistics.mean(values)
            if mean_val == 0:
                return 0.0

            std_val = statistics.stdev(values)
            return std_val / abs(mean_val)  # 变异系数

        except Exception:
            return 0.0

    async def _gpt_enhanced_statistics(self, values: List[float], metric: str) -> Optional[Dict[str, Any]]:
        """GPT增强统计分析"""
        try:
            if not self.gpt_client:
                return None

            stats_prompt = f"""
            对以下数据序列进行高级统计分析：

            指标: {metric}
            数据点: {len(values)}
            数据序列: {values}

            请计算：
            1. 基础统计量 (均值、中位数、标准差等)
            2. 趋势分析 (线性趋势、增长率等)
            3. 异常值检测
            4. 周期性检测
            5. 预测性指标

            返回JSON格式的详细统计结果。
            """

            gpt_result = await self.gpt_client.process_direct_query(
                stats_prompt,
                {"values": values, "metric": metric}
            )

            if gpt_result.get('success'):
                # 解析GPT返回的统计结果
                response_text = gpt_result.get('response', '')

                # 简化解析，实际会有更复杂的JSON解析逻辑
                enhanced_stats = {
                    'ai_enhanced': True,
                    'model_used': 'gpt',
                    'detailed_analysis': response_text
                }

                return enhanced_stats

        except Exception as e:
            logger.error(f"GPT统计增强失败: {str(e)}")
            return None

    # ============= 工具方法 =============

    def _determine_analysis_scope(self, data_source: str) -> AnalysisScope:
        """根据数据源确定分析范围"""
        scope_mapping = {
            'system': AnalysisScope.SYSTEM_LEVEL,
            'daily': AnalysisScope.OPERATIONAL_LEVEL,
            'product': AnalysisScope.PRODUCT_LEVEL,
            'user': AnalysisScope.USER_LEVEL,
            'financial': AnalysisScope.FINANCIAL_LEVEL
        }
        return scope_mapping.get(data_source, AnalysisScope.SYSTEM_LEVEL)

    def _calculate_analysis_confidence(self, analysis_data: Dict[str, Any],
                                       trend_statistics: Dict[str, Any],
                                       anomalies: List[Dict[str, Any]]) -> float:
        """计算分析置信度"""
        confidence_factors = []

        # 数据质量因子
        data_quality = analysis_data.get('data_quality', DataQuality.GOOD)
        if data_quality == DataQuality.EXCELLENT:
            confidence_factors.append(0.95)
        elif data_quality == DataQuality.GOOD:
            confidence_factors.append(0.85)
        elif data_quality == DataQuality.ACCEPTABLE:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # 数据点数量因子
        data_points = trend_statistics.get('data_points', 0)
        if data_points >= 30:
            confidence_factors.append(0.9)
        elif data_points >= 14:
            confidence_factors.append(0.8)
        elif data_points >= 7:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # 趋势强度因子
        trend_strength = trend_statistics.get('trend_strength', 0)
        confidence_factors.append(max(0.5, min(0.95, trend_strength)))

        # 异常影响因子
        if len(anomalies) == 0:
            confidence_factors.append(0.9)
        elif len(anomalies) <= 2:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        return sum(confidence_factors) / len(confidence_factors)

    def _update_analysis_stats(self, analysis_type: str, confidence: float):
        """更新分析统计信息"""
        if analysis_type not in self.analysis_stats['analysis_by_type']:
            self.analysis_stats['analysis_by_type'][analysis_type] = 0

        self.analysis_stats['analysis_by_type'][analysis_type] += 1

        # 更新平均置信度
        total_analyses = self.analysis_stats['total_analyses']
        current_avg = self.analysis_stats['avg_confidence_score']
        new_avg = (current_avg * (total_analyses - 1) + confidence) / total_analyses
        self.analysis_stats['avg_confidence_score'] = new_avg

    def _create_error_analysis_result(self, analysis_type: str, error: str) -> AnalysisResult:
        """创建错误分析结果"""
        return AnalysisResult(
            analysis_id=f"error_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            analysis_type=AnalysisType(analysis_type),
            analysis_scope=AnalysisScope.SYSTEM_LEVEL,
            confidence_score=0.0,

            key_findings=[f"分析失败: {error}"],
            trends=[],
            anomalies=[],
            metrics={},

            business_insights=[],
            risk_factors=[f"分析错误: {error}"],
            opportunities=[],
            recommendations=["请检查数据源和分析参数"],

            data_quality=DataQuality.INSUFFICIENT,
            analysis_metadata={"error": error},
            processing_time=0.0,
            timestamp=datetime.now().isoformat()
        )

    # ============= 外部接口方法 =============

    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return self.analysis_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "ai_clients": {
                "claude_available": self.claude_client is not None,
                "gpt_available": self.gpt_client is not None
            },
            "tools": {
                "date_utils": self.date_utils is not None,
                "validator": self.validator is not None,
                "time_series_builder": self.time_series_builder is not None,
                "financial_calculator": self.financial_calculator is not None
            },
            "analysis_stats": self.analysis_stats,
            "timestamp": datetime.now().isoformat()
        }


# ============= 工厂函数 =============

def create_financial_data_analyzer(claude_client=None, gpt_client=None) -> FinancialDataAnalyzer:
    """
    创建金融数据分析器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        FinancialDataAnalyzer: 金融数据分析器实例
    """
    return FinancialDataAnalyzer(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建分析器
    analyzer = create_financial_data_analyzer()

    print("=== 金融数据分析器测试 ===")

    # 1. 趋势分析
    trend_result = await analyzer.analyze_trend(
        data_source="system",
        metric="total_balance",
        time_range=30
    )

    print(f"趋势分析结果:")
    print(f"- 分析ID: {trend_result.analysis_id}")
    print(f"- 置信度: {trend_result.confidence_score:.2f}")
    print(f"- 关键发现: {trend_result.key_findings}")
    print(f"- 处理时间: {trend_result.processing_time:.2f}秒")

    # 2. 业务表现分析
    performance_result = await analyzer.analyze_business_performance(
        scope="financial",
        time_range=30
    )

    print(f"\n业务表现分析结果:")
    print(f"- 分析范围: {performance_result.analysis_scope.value}")
    print(f"- 业务洞察: {performance_result.business_insights}")
    print(f"- 风险因素: {performance_result.risk_factors}")

    # 3. 异常检测
    anomaly_result = await analyzer.detect_anomalies(
        data_source="daily",
        metrics=["daily_inflow", "daily_outflow"],
        sensitivity=2.0
    )

    print(f"\n异常检测结果:")
    print(f"- 发现异常: {len(anomaly_result.anomalies)}个")
    print(f"- 建议措施: {anomaly_result.recommendations}")

    # 4. 健康检查
    health_status = await analyzer.health_check()
    print(f"\n系统健康状态: {health_status['status']}")

    # 5. 统计信息
    stats = analyzer.get_analysis_stats()
    print(f"总分析次数: {stats['total_analyses']}")
    print(f"平均置信度: {stats['avg_confidence_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())