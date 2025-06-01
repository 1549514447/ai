"""
🧮 统一数据计算器 - 合并版本
合并了StatisticalCalculator和FinancialCalculator的所有功能
作为系统唯一的计算引擎

合并策略:
1. 保留StatisticalCalculator的框架和API设计
2. 集成FinancialCalculator的高精度金融计算
3. 统一数据结构，消除冲突
4. 其他所有类都调用这个统一计算器
"""

import logging
import statistics
import math
import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


class CalculationType(Enum):
    """统一的计算类型枚举"""
    # 基础统计
    BASIC_STATISTICS = "basic_statistics"
    TREND_ANALYSIS = "trend_analysis"
    GROWTH_CALCULATION = "growth_calculation"
    COMPARISON_ANALYSIS = "comparison_analysis"  # 🆕 添加对比分析

    # 金融计算 (来自FinancialCalculator)
    COMPOUND_INTEREST = "compound_interest"
    REINVESTMENT_ANALYSIS = "reinvestment_analysis"
    FINANCIAL_RATIOS = "financial_ratios"
    CASH_FLOW_ANALYSIS = "cash_flow_analysis"
    ROI_CALCULATION = "roi_calculation"

    # 预测计算
    TREND_PREDICTION = "trend_prediction"
    CASH_RUNWAY = "cash_runway"
    SCENARIO_SIMULATION = "scenario_simulation"

    # 业务分析
    USER_ANALYTICS = "user_analytics"
    VIP_DISTRIBUTION = "vip_distribution"
    PRODUCT_PERFORMANCE = "product_performance"
    EXPIRY_ANALYSIS = "expiry_analysis"


@dataclass
class UnifiedCalculationResult:
    """统一的计算结果格式 - 解决数据结构冲突"""
    calculation_type: str
    success: bool
    primary_result: Union[float, Dict[str, Any]]  # 主要结果
    detailed_results: Dict[str, Any]  # 详细结果
    metadata: Dict[str, Any]
    confidence: float = 1.0
    processing_time: float = 0.0
    warnings: List[str] = None
    calculation_steps: List[Dict[str, Any]] = None  # 来自FinancialCalculator的计算步骤

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.calculation_steps is None:
            self.calculation_steps = []


class UnifiedCalculator:
    """
    🧮 统一数据计算器

    合并了原来两个计算器的所有功能：
    - StatisticalCalculator: 业务数据分析和统计
    - FinancialCalculator: 高精度金融计算
    """

    def __init__(self, gpt_client=None, precision: int = 6):
        """
        初始化统一计算器

        Args:
            gpt_client: GPT客户端，用于AI增强计算
            precision: 金融计算精度 (小数位数)
        """
        self.gpt_client = gpt_client
        self.precision = precision

        # 计算配置
        self.config = {
            'precision': precision,
            'ai_calculation_timeout': 30,
            'min_data_points_for_trend': 3,
            'confidence_threshold': 0.8,
            'outlier_threshold': 2.5,  # 异常值阈值
        }

        # 金融数据合理性范围 (来自原ValidationUtils)
        self.financial_ranges = {
            "balance_min": 0,
            "balance_max": 1000000000,  # 10亿
            "user_count_min": 0,
            "user_count_max": 10000000,  # 1千万
            "daily_amount_min": 0,
            "daily_amount_max": 50000000,  # 5千万/天
        }

        # 计算统计
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'ai_calculations': 0,
            'calculation_types': {},
            'average_processing_time': 0.0
        }

        logger.info("UnifiedCalculator initialized - All-in-one calculation engine")

    # ============= 统一计算入口 =============

    async def calculate(self, calculation_type: str, data: Dict[str, Any],
                        params: Optional[Dict[str, Any]] = None) -> UnifiedCalculationResult:
        """
        🎯 统一计算入口 - 自动路由到对应的计算方法

        Args:
            calculation_type: 计算类型
            data: 输入数据
            params: 计算参数

        Returns:
            UnifiedCalculationResult: 统一的计算结果
        """
        start_time = datetime.now()
        params = params or {}

        try:
            self.stats['total_calculations'] += 1

            # 🎯 智能路由到具体计算方法
            if calculation_type == CalculationType.BASIC_STATISTICS.value:
                result = await self._calculate_basic_statistics(data, params)
            elif calculation_type == CalculationType.TREND_ANALYSIS.value:
                result = await self._calculate_trend_analysis(data, params)
            elif calculation_type == CalculationType.COMPARISON_ANALYSIS.value:  # 🆕 添加对比分析路由
                result = await self._calculate_comparison_analysis(data, params)
            elif calculation_type == CalculationType.COMPOUND_INTEREST.value:
                result = await self._calculate_compound_interest(data, params)
            elif calculation_type == CalculationType.REINVESTMENT_ANALYSIS.value:
                result = await self._calculate_reinvestment_analysis(data, params)
            elif calculation_type == CalculationType.CASH_FLOW_ANALYSIS.value:
                result = self._calculate_cash_flow_analysis(data, params)
            elif calculation_type == CalculationType.ROI_CALCULATION.value:
                result = self._calculate_roi_metrics(data, params)
            elif calculation_type == CalculationType.FINANCIAL_RATIOS.value:
                result = self._calculate_financial_ratios(data, params)
            elif calculation_type == CalculationType.TREND_PREDICTION.value:
                result = await self._calculate_trend_prediction(data, params)
            elif calculation_type == CalculationType.CASH_RUNWAY.value:
                result = self._calculate_cash_runway(data, params)
            elif calculation_type == CalculationType.SCENARIO_SIMULATION.value:
                result = self._calculate_scenario_simulation(data, params)
            elif calculation_type == CalculationType.USER_ANALYTICS.value:
                result = self._calculate_user_analytics(data, params)
            elif calculation_type == CalculationType.VIP_DISTRIBUTION.value:
                result = self._calculate_vip_distribution(data, params)
            elif calculation_type == CalculationType.PRODUCT_PERFORMANCE.value:
                result = self._calculate_product_performance(data, params)
            elif calculation_type == CalculationType.EXPIRY_ANALYSIS.value:
                result = self._calculate_expiry_analysis(data, params)
            else:
                raise ValueError(f"Unsupported calculation type: {calculation_type}")

            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            # 更新统计
            self._update_stats(calculation_type, processing_time, True)

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(calculation_type, processing_time, False)

            logger.error(f"Calculation failed for {calculation_type}: {str(e)}")

            return UnifiedCalculationResult(
                calculation_type=calculation_type,
                success=False,
                primary_result=0.0,
                detailed_results={'error': str(e)},
                metadata={'error_details': str(e)},
                confidence=0.0,
                processing_time=processing_time,
                warnings=[f"计算失败: {str(e)}"]
            )

    # 🆕 添加对比分析计算方法
    async def _calculate_comparison_analysis(self, data: Dict[str, Any],
                                             params: Dict[str, Any]) -> UnifiedCalculationResult:
        """🆕 对比分析计算 - 统一计算器版本"""
        try:
            logger.info("🆕 执行对比分析计算")

            # 从提取的数据中获取对比信息
            extracted_data = data.get('extracted_metrics', {})
            comparison_analysis = data.get('comparison_analysis', {})

            calculation_results = {}
            insights = []

            # 如果Claude提取器已经做了对比分析，直接使用
            if comparison_analysis:
                logger.info("使用Claude提取器的对比分析结果")
                calculation_results = comparison_analysis

                # 生成洞察
                for metric, analysis in comparison_analysis.items():
                    if isinstance(analysis, dict):
                        current_val = analysis.get('current_value', 0)
                        baseline_val = analysis.get('baseline_value', 0)
                        change_rate = analysis.get('percentage_change', 0)
                        direction = analysis.get('change_direction', '持平')

                        insight = f"{metric}{direction}{abs(change_rate):.1%}，从{baseline_val:,.2f}变为{current_val:,.2f}"
                        insights.append(insight)

            # 如果没有现成的对比分析，尝试从原始数据构建
            elif extracted_data:
                logger.info("从原始数据构建对比分析")

                # 查找本周和上周的数据
                current_week_total = {}
                last_week_total = {}

                # 🔍 调试：打印所有键
                logger.info(f"🔍 [DEBUG] extracted_data的所有键: {list(extracted_data.keys())}")

                for key, value in extracted_data.items():
                    if 'current_week' in key.lower():
                        logger.info(f"🔍 [DEBUG] 发现本周数据: {key}")
                        # 累加本周数据
                        if isinstance(value, dict):
                            for metric, amount in value.items():
                                if isinstance(amount, (int, float)):
                                    current_week_total[metric] = current_week_total.get(metric, 0) + amount
                    elif 'last_week' in key.lower():
                        logger.info(f"🔍 [DEBUG] 发现上周数据: {key}")
                        # 累加上周数据
                        if isinstance(value, dict):
                            for metric, amount in value.items():
                                if isinstance(amount, (int, float)):
                                    last_week_total[metric] = last_week_total.get(metric, 0) + amount

                logger.info(f"🔍 [DEBUG] 本周汇总: {current_week_total}")
                logger.info(f"🔍 [DEBUG] 上周汇总: {last_week_total}")

                # 计算对比
                for metric in ['入金', '出金', '净流入', '注册人数']:
                    if metric in current_week_total and metric in last_week_total:
                        current_val = current_week_total[metric]
                        last_val = last_week_total[metric]

                        if last_val != 0:
                            change_rate = (current_val - last_val) / last_val
                            calculation_results[metric] = {
                                'current_value': current_val,
                                'baseline_value': last_val,
                                'absolute_change': current_val - last_val,
                                'percentage_change': change_rate,
                                'change_direction': '增长' if change_rate > 0 else '下降' if change_rate < 0 else '持平'
                            }

                            insights.append(f"{metric}{'增长' if change_rate > 0 else '下降'}{abs(change_rate):.1%}")

            # 🔍 特殊处理：如果仍然没有结果，查找数据结构
            if not calculation_results:
                logger.warning("🔍 [DEBUG] 常规方法未找到对比数据，尝试深度查找...")

                # 尝试从results中查找
                if 'results' in data:
                    results = data['results']
                    logger.info(f"🔍 [DEBUG] results中的键: {list(results.keys())}")

                    # 聚合current_week和last_week的数据
                    current_data = {}
                    last_data = {}

                    for result_key, result_value in results.items():
                        if 'current_week' in result_key and result_value.get('success'):
                            result_data = result_value.get('data', {})
                            for metric in ['入金', '出金', '注册人数']:
                                if metric in result_data:
                                    current_data[metric] = current_data.get(metric, 0) + float(result_data[metric])

                        elif 'last_week' in result_key and result_value.get('success'):
                            result_data = result_value.get('data', {})
                            for metric in ['入金', '出金', '注册人数']:
                                if metric in result_data:
                                    last_data[metric] = last_data.get(metric, 0) + float(result_data[metric])

                    logger.info(f"🔍 [DEBUG] 从results聚合 - 本周: {current_data}, 上周: {last_data}")

                    # 计算对比
                    for metric in ['入金', '出金', '注册人数']:
                        if metric in current_data and metric in last_data:
                            current_val = current_data[metric]
                            last_val = last_data[metric]

                            if last_val != 0:
                                change_rate = (current_val - last_val) / last_val
                                calculation_results[metric] = {
                                    'current_value': current_val,
                                    'baseline_value': last_val,
                                    'absolute_change': current_val - last_val,
                                    'percentage_change': change_rate,
                                    'change_direction': '增长' if change_rate > 0 else '下降' if change_rate < 0 else '持平'
                                }

                                insights.append(
                                    f"{metric}{'增长' if change_rate > 0 else '下降'}{abs(change_rate):.1%}")

                    # 计算净流入
                    if '入金' in calculation_results and '出金' in calculation_results:
                        current_net = calculation_results['入金']['current_value'] - calculation_results['出金'][
                            'current_value']
                        last_net = calculation_results['入金']['baseline_value'] - calculation_results['出金'][
                            'baseline_value']

                        if last_net != 0:
                            net_change_rate = (current_net - last_net) / last_net
                            calculation_results['净流入'] = {
                                'current_value': current_net,
                                'baseline_value': last_net,
                                'absolute_change': current_net - last_net,
                                'percentage_change': net_change_rate,
                                'change_direction': '增长' if net_change_rate > 0 else '下降' if net_change_rate < 0 else '持平'
                            }

                            insights.append(
                                f"净流入{'增长' if net_change_rate > 0 else '下降'}{abs(net_change_rate):.1%}")

            if not calculation_results:
                # 最终降级处理：提供基础统计信息
                logger.warning("无法进行详细对比分析，提供基础统计")
                calculation_results = {
                    'summary': '对比分析数据不足，建议检查数据获取结果',
                    'data_available': len(extracted_data),
                    'available_keys': list(extracted_data.keys())[:5] if extracted_data else []
                }
                insights = ['数据获取可能不完整，请检查API调用结果']

            return UnifiedCalculationResult(
                calculation_type=CalculationType.COMPARISON_ANALYSIS.value,
                success=True,
                primary_result=len(calculation_results),
                detailed_results=calculation_results,
                metadata={
                    'comparison_metrics_count': len(calculation_results),
                    'data_source': 'claude_extractor' if comparison_analysis else 'unified_calculator_processing',
                    'insights': insights
                },
                confidence=0.9 if comparison_analysis else (0.8 if len(calculation_results) > 1 else 0.6)
            )

        except Exception as e:
            logger.error(f"对比分析计算失败: {str(e)}")
            return UnifiedCalculationResult(
                calculation_type=CalculationType.COMPARISON_ANALYSIS.value,
                success=False,
                primary_result=0.0,
                detailed_results={'error': str(e)},
                metadata={'error_details': str(e)},
                confidence=0.0,
                warnings=[f"对比分析计算失败: {str(e)}"]
            )
    # ============= 高精度金融计算 (来自FinancialCalculator) =============

    async def _calculate_compound_interest(self, data: Dict[str, Any],
                                           params: Dict[str, Any]) -> UnifiedCalculationResult:
        """高精度复利计算"""
        try:
            # 从params获取参数
            principal = float(params.get('principal', 10000))
            rate = float(params.get('rate', 0.05))
            periods = int(params.get('periods', 12))
            frequency = int(params.get('frequency', 1))

            # 🔢 使用Decimal确保高精度
            p = Decimal(str(principal))
            r = Decimal(str(rate))
            n = Decimal(str(frequency))
            t = Decimal(str(periods))

            # 复利公式: A = P(1 + r/n)^(nt)
            rate_per_period = r / n
            exponent = n * t
            compound_factor = (1 + rate_per_period) ** float(exponent)

            final_amount = float(p * Decimal(str(compound_factor)))
            interest_earned = final_amount - principal

            # 📋 构建计算步骤 (保留原FinancialCalculator的详细步骤)
            calculation_steps = [
                {
                    "step": 1,
                    "description": "计算每期利率",
                    "formula": "r/n",
                    "calculation": f"{rate}/{frequency} = {float(rate_per_period):.6f}",
                    "result": float(rate_per_period)
                },
                {
                    "step": 2,
                    "description": "计算复利因子",
                    "formula": "(1 + r/n)^(nt)",
                    "calculation": f"(1 + {float(rate_per_period):.6f})^{float(exponent)} = {compound_factor:.6f}",
                    "result": compound_factor
                },
                {
                    "step": 3,
                    "description": "计算最终金额",
                    "formula": "P × 复利因子",
                    "calculation": f"{principal} × {compound_factor:.6f} = {final_amount:.2f}",
                    "result": final_amount
                }
            ]

            detailed_results = {
                'final_amount': final_amount,
                'interest_earned': interest_earned,
                'compound_factor': compound_factor,
                'effective_annual_rate': (final_amount / principal) ** (1 / periods) - 1 if periods > 0 else 0,
                'calculation_parameters': {
                    'principal': principal,
                    'rate': rate,
                    'periods': periods,
                    'frequency': frequency
                }
            }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.COMPOUND_INTEREST.value,
                success=True,
                primary_result=final_amount,
                detailed_results=detailed_results,
                metadata={
                    'calculation_method': 'high_precision_decimal',
                    'precision_digits': self.precision
                },
                confidence=0.98,  # 数学计算高置信度
                calculation_steps=calculation_steps
            )

        except Exception as e:
            logger.error(f"Compound interest calculation failed: {e}")
            return self._create_error_result(CalculationType.COMPOUND_INTEREST.value, str(e))

    async def _calculate_reinvestment_analysis(self, data: Dict[str, Any],
                                               params: Dict[str, Any]) -> UnifiedCalculationResult:
        """复投分析计算"""
        try:
            base_amount = float(params.get('base_amount', 100000))
            reinvest_rate = float(params.get('reinvestment_rate', 0.5))
            periods = int(params.get('periods', 12))
            monthly_return = float(params.get('monthly_return_rate', 0.02))

            # 复投建模
            current_balance = base_amount
            monthly_details = []
            total_withdrawn = 0

            for month in range(periods):
                monthly_gain = current_balance * monthly_return
                total_available = current_balance + monthly_gain

                reinvested = total_available * reinvest_rate
                withdrawn = total_available * (1 - reinvest_rate)
                total_withdrawn += withdrawn

                current_balance = reinvested

                monthly_details.append({
                    'month': month + 1,
                    'starting_balance': current_balance,
                    'monthly_gain': monthly_gain,
                    'reinvested': reinvested,
                    'withdrawn': withdrawn,
                    'ending_balance': current_balance
                })

            # 计算关键指标
            final_balance = current_balance
            total_return = total_withdrawn + final_balance - base_amount
            roi = total_return / base_amount if base_amount > 0 else 0

            detailed_results = {
                'final_balance': final_balance,
                'total_withdrawn': total_withdrawn,
                'total_return': total_return,
                'roi_percentage': roi * 100,
                'monthly_breakdown': monthly_details,
                'risk_assessment': self._assess_reinvestment_risk(reinvest_rate, final_balance)
            }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.REINVESTMENT_ANALYSIS.value,
                success=True,
                primary_result=total_return,
                detailed_results=detailed_results,
                metadata={
                    'periods_analyzed': periods,
                    'reinvestment_rate': reinvest_rate
                },
                confidence=0.85
            )

        except Exception as e:
            return self._create_error_result(CalculationType.REINVESTMENT_ANALYSIS.value, str(e))

    def _calculate_financial_ratios(self, data: Dict[str, Any],
                                    params: Dict[str, Any]) -> UnifiedCalculationResult:
        """金融比率计算 (来自FinancialCalculator)"""
        try:
            # 从系统数据中提取财务数据
            financial_data = {}
            if 'system_data' in data:
                system_data = data['system_data']
                financial_data = {
                    "总余额": float(system_data.get("总余额", 0)),
                    "总入金": float(system_data.get("总入金", 0)),
                    "总出金": float(system_data.get("总出金", 0)),
                    "总投资金额": float(system_data.get("总投资金额", 0)),
                    "总奖励发放": float(system_data.get("总奖励发放", 0))
                }

            ratios = {}

            # 获取基础数据
            total_balance = financial_data.get("总余额", 0)
            total_inflow = financial_data.get("总入金", 0)
            total_outflow = financial_data.get("总出金", 0)
            total_investment = financial_data.get("总投资金额", 0)
            total_rewards = financial_data.get("总奖励发放", 0)

            # 计算各种金融比率
            if total_outflow > 0:
                ratios["liquidity_ratio"] = total_balance / total_outflow

            if total_balance > 0:
                ratios["fund_utilization_ratio"] = total_investment / total_balance

            if total_investment > 0:
                ratios["return_on_investment"] = total_rewards / total_investment

            net_flow = total_inflow - total_outflow
            if total_inflow > 0:
                ratios["net_inflow_ratio"] = net_flow / total_inflow
                ratios["cashout_ratio"] = total_outflow / total_inflow
                ratios["fund_growth_multiple"] = total_balance / total_inflow

            return UnifiedCalculationResult(
                calculation_type=CalculationType.FINANCIAL_RATIOS.value,
                success=True,
                primary_result=len(ratios),
                detailed_results=ratios,
                metadata={
                    'ratios_calculated': list(ratios.keys()),
                    'base_data': financial_data
                },
                confidence=0.9
            )

        except Exception as e:
            return self._create_error_result(CalculationType.FINANCIAL_RATIOS.value, str(e))

    def _assess_reinvestment_risk(self, reinvest_rate: float, final_balance: float) -> Dict[str, Any]:
        """评估复投风险"""
        if reinvest_rate > 0.8:
            risk_level = "高风险"
            liquidity_concern = "流动性不足"
        elif reinvest_rate > 0.5:
            risk_level = "中等风险"
            liquidity_concern = "流动性适中"
        else:
            risk_level = "低风险"
            liquidity_concern = "流动性充足"

        return {
            'risk_level': risk_level,
            'liquidity_assessment': liquidity_concern,
            'sustainability_score': min(100, max(0, (1 - abs(reinvest_rate - 0.6)) * 100))
        }

    # ============= 业务数据分析 (来自StatisticalCalculator) =============

    async def _calculate_basic_statistics(self, data: Dict[str, Any],
                                          params: Dict[str, Any]) -> UnifiedCalculationResult:
        """基础统计计算"""
        try:
            # 从API数据中提取数值序列
            numeric_data = self._extract_numeric_series(data)
            results = {}

            for metric_name, values in numeric_data.items():
                if not values or len(values) < 1:
                    continue

                # 基础统计指标
                metric_stats = {
                    'count': len(values),
                    'mean': round(statistics.mean(values), self.config['precision']),
                    'median': round(statistics.median(values), self.config['precision']),
                    'min': round(min(values), self.config['precision']),
                    'max': round(max(values), self.config['precision']),
                    'sum': round(sum(values), self.config['precision']),
                }

                # 方差和标准差
                if len(values) >= 2:
                    metric_stats.update({
                        'std_dev': round(statistics.stdev(values), self.config['precision']),
                        'variance': round(statistics.variance(values), self.config['precision']),
                    })

                    if metric_stats['mean'] != 0:
                        metric_stats['coefficient_of_variation'] = round(
                            metric_stats['std_dev'] / metric_stats['mean'],
                            self.config['precision']
                        )

                # 增长率
                if len(values) >= 2:
                    growth_rate = (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0
                    metric_stats['total_growth_rate'] = round(growth_rate, self.config['precision'])

                results[metric_name] = metric_stats

            return UnifiedCalculationResult(
                calculation_type=CalculationType.BASIC_STATISTICS.value,
                success=len(results) > 0,
                primary_result=len(results),
                detailed_results=results,
                metadata={
                    'metrics_calculated': list(results.keys()),
                    'total_metrics': len(results)
                },
                confidence=0.95 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.BASIC_STATISTICS.value, str(e))

    def _calculate_cash_flow_analysis(self, data: Dict[str, Any],
                                      params: Dict[str, Any]) -> UnifiedCalculationResult:
        """现金流分析计算"""
        try:
            results = {}

            # 从系统概览数据提取
            if 'system_data' in data:
                system_data = data['system_data']
                total_inflow = float(system_data.get('总入金', 0))
                total_outflow = float(system_data.get('总出金', 0))
                current_balance = float(system_data.get('总余额', 0))

                net_flow = total_inflow - total_outflow
                outflow_ratio = total_outflow / total_inflow if total_inflow > 0 else 0
                balance_ratio = current_balance / total_inflow if total_inflow > 0 else 0

                results['overall_cashflow'] = {
                    'total_inflow': total_inflow,
                    'total_outflow': total_outflow,
                    'net_flow': net_flow,
                    'outflow_ratio': round(outflow_ratio, self.config['precision']),
                    'balance_ratio': round(balance_ratio, self.config['precision']),
                    'current_balance': current_balance
                }

            # 从每日数据计算趋势
            if 'daily_data' in data:
                daily_data = data['daily_data']
                if isinstance(daily_data, list) and daily_data:
                    daily_inflows = [float(d.get('入金', 0)) for d in daily_data]
                    daily_outflows = [float(d.get('出金', 0)) for d in daily_data]
                    daily_net_flows = [daily_inflows[i] - daily_outflows[i] for i in range(len(daily_inflows))]

                    if daily_net_flows:
                        avg_daily_inflow = statistics.mean(daily_inflows)
                        avg_daily_outflow = statistics.mean(daily_outflows)
                        avg_daily_net = statistics.mean(daily_net_flows)

                        results['daily_cashflow_trends'] = {
                            'avg_daily_inflow': round(avg_daily_inflow, self.config['precision']),
                            'avg_daily_outflow': round(avg_daily_outflow, self.config['precision']),
                            'avg_daily_net': round(avg_daily_net, self.config['precision']),
                            'analysis_period_days': len(daily_data)
                        }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.CASH_FLOW_ANALYSIS.value,
                success=len(results) > 0,
                primary_result=results.get('overall_cashflow', {}).get('net_flow', 0),
                detailed_results=results,
                metadata={'analysis_components': list(results.keys())},
                confidence=0.85
            )

        except Exception as e:
            return self._create_error_result(CalculationType.CASH_FLOW_ANALYSIS.value, str(e))

    async def _calculate_trend_analysis(self, data: Dict[str, Any],
                                        params: Dict[str, Any]) -> UnifiedCalculationResult:
        """趋势分析计算"""
        try:
            # 如果有GPT客户端，使用AI增强分析
            if self.gpt_client and params.get('use_ai', True):
                self.stats['ai_calculations'] += 1
                return await self._ai_enhanced_trend_analysis(data, params)

            # 基础趋势分析
            numeric_data = self._extract_numeric_series(data)
            results = {}

            for metric_name, values in numeric_data.items():
                if len(values) < self.config['min_data_points_for_trend']:
                    continue

                # 简单线性回归
                x = list(range(len(values)))
                y = values
                n = len(values)

                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(xi * xi for xi in x)

                # 计算斜率和截距
                if (n * sum_x2 - sum_x * sum_x) != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n

                    # R²相关系数
                    y_mean = statistics.mean(y)
                    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
                    ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                    # 趋势描述
                    trend_direction = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
                    trend_strength = abs(slope) / max(abs(max(y)), abs(min(y))) if max(y) != min(y) else 0

                    results[metric_name] = {
                        'slope': round(slope, self.config['precision']),
                        'intercept': round(intercept, self.config['precision']),
                        'r_squared': round(r_squared, self.config['precision']),
                        'trend_direction': trend_direction,
                        'trend_strength': round(trend_strength, self.config['precision']),
                        'data_points': n,
                        'confidence': min(r_squared + 0.1, 1.0)
                    }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.TREND_ANALYSIS.value,
                success=len(results) > 0,
                primary_result=len(results),
                detailed_results=results,
                metadata={
                    'analysis_method': 'linear_regression',
                    'metrics_analyzed': list(results.keys())
                },
                confidence=0.8 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.TREND_ANALYSIS.value, str(e))

    # ============= 预测计算 =============

    async def _calculate_trend_prediction(self, data: Dict[str, Any],
                                          params: Dict[str, Any]) -> UnifiedCalculationResult:
        """趋势预测计算"""
        try:
            prediction_days = params.get('prediction_days', 30)

            # 先进行趋势分析
            trend_result = await self._calculate_trend_analysis(data, params)

            if not trend_result.success:
                return self._create_error_result(CalculationType.TREND_PREDICTION.value, '趋势分析失败，无法进行预测')

            results = {}

            for metric_name, trend_data in trend_result.detailed_results.items():
                if 'error' in trend_data:
                    continue

                slope = trend_data.get('slope', 0)
                intercept = trend_data.get('intercept', 0)
                r_squared = trend_data.get('r_squared', 0)
                data_points = trend_data.get('data_points', 0)

                # 基于线性趋势预测
                predicted_values = []
                for future_x in range(data_points, data_points + prediction_days):
                    predicted_value = slope * future_x + intercept
                    predicted_values.append(max(0, predicted_value))

                # 预测置信度
                prediction_confidence = min(r_squared * (data_points / 10), 0.95)

                results[metric_name] = {
                    'prediction_method': 'linear_extrapolation',
                    'prediction_days': prediction_days,
                    'predicted_values': [round(v, self.config['precision']) for v in predicted_values],
                    'final_predicted_value': round(predicted_values[-1], self.config['precision']),
                    'trend_slope': slope,
                    'prediction_confidence': round(prediction_confidence, self.config['precision']),
                    'based_on_data_points': data_points
                }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.TREND_PREDICTION.value,
                success=len(results) > 0,
                primary_result=len(results),
                detailed_results=results,
                metadata={
                    'prediction_horizon_days': prediction_days,
                    'metrics_predicted': list(results.keys())
                },
                confidence=0.7
            )

        except Exception as e:
            return self._create_error_result(CalculationType.TREND_PREDICTION.value, str(e))

    def _calculate_cash_runway(self, data: Dict[str, Any],
                               params: Dict[str, Any]) -> UnifiedCalculationResult:
        """现金跑道计算"""
        try:
            results = {}

            # 获取当前余额和日均支出
            current_balance = 0
            daily_outflow = 0

            if 'system_data' in data:
                current_balance = float(data['system_data'].get('总余额', 0))

            if 'daily_data' in data:
                daily_data = data['daily_data']
                if isinstance(daily_data, list) and daily_data:
                    outflows = [float(d.get('出金', 0)) for d in daily_data]
                    daily_outflow = statistics.mean(outflows) if outflows else 0

            # 从参数获取（优先级更高）
            if 'current_balance' in params:
                current_balance = float(params['current_balance'])
            if 'daily_outflow' in params:
                daily_outflow = float(params['daily_outflow'])

            if current_balance <= 0:
                results['error'] = '当前余额为零或负数'
            elif daily_outflow <= 0:
                results['runway_analysis'] = {
                    'current_balance': current_balance,
                    'daily_outflow': daily_outflow,
                    'runway_days': float('inf'),
                    'runway_months': float('inf'),
                    'risk_level': 'low',
                    'note': '无现金流出，资金可持续'
                }
            else:
                runway_days = current_balance / daily_outflow
                runway_months = runway_days / 30.44

                # 风险评级
                if runway_days < 30:
                    risk_level = 'critical'
                elif runway_days < 90:
                    risk_level = 'high'
                elif runway_days < 180:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'

                depletion_date = (datetime.now() + timedelta(days=runway_days)).strftime('%Y-%m-%d')

                results['runway_analysis'] = {
                    'current_balance': current_balance,
                    'daily_outflow': daily_outflow,
                    'runway_days': round(runway_days, 1),
                    'runway_months': round(runway_months, 1),
                    'risk_level': risk_level,
                    'estimated_depletion_date': depletion_date,
                    'monthly_burn_rate': round(daily_outflow * 30.44, self.config['precision'])
                }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.CASH_RUNWAY.value,
                success='error' not in results,
                primary_result=results.get('runway_analysis', {}).get('runway_days', 0),
                detailed_results=results,
                metadata={'calculation_date': datetime.now().isoformat()},
                confidence=0.9 if 'error' not in results else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.CASH_RUNWAY.value, str(e))

    # ============= 业务分析计算 =============

    def _calculate_user_analytics(self, data: Dict[str, Any],
                                  params: Dict[str, Any]) -> UnifiedCalculationResult:
        """用户分析计算"""
        try:
            results = {}

            # 用户概览分析
            if 'system_data' in data:
                system_data = data['system_data']
                user_stats = system_data.get('用户统计', {})

                total_users = int(user_stats.get('总用户数', 0))
                active_users = int(user_stats.get('活跃用户数', 0))

                if total_users > 0:
                    activity_rate = active_users / total_users
                    engagement_score = self._calculate_engagement_score(active_users, total_users)

                    results['user_overview'] = {
                        'total_users': total_users,
                        'active_users': active_users,
                        'inactive_users': total_users - active_users,
                        'activity_rate': round(activity_rate, self.config['precision']),
                        'engagement_score': engagement_score
                    }

            # 用户投资行为分析
            if 'user_data' in data:
                user_data = data['user_data']
                if isinstance(user_data, list) and user_data:
                    investments = [float(u.get('总投入', 0)) for u in user_data if u.get('总投入')]

                    if investments:
                        results['investment_behavior'] = {
                            'analyzed_users': len(investments),
                            'avg_investment': statistics.mean(investments),
                            'median_investment': statistics.median(investments),
                            'total_investment': sum(investments),
                            'investment_std_dev': statistics.stdev(investments) if len(investments) > 1 else 0,
                            'min_investment': min(investments),
                            'max_investment': max(investments)
                        }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.USER_ANALYTICS.value,
                success=len(results) > 0,
                primary_result=results.get('user_overview', {}).get('engagement_score', 0),
                detailed_results=results,
                metadata={'analysis_components': list(results.keys())},
                confidence=0.85
            )

        except Exception as e:
            return self._create_error_result(CalculationType.USER_ANALYTICS.value, str(e))

    def _calculate_engagement_score(self, active_users: int, total_users: int) -> float:
        """计算用户参与度得分"""
        if total_users == 0:
            return 0.0

        base_score = (active_users / total_users) * 100

        # 根据活跃用户规模调整
        if active_users > 1000:
            scale_bonus = min(10, active_users / 1000)
        else:
            scale_bonus = 0

        return min(100, base_score + scale_bonus)

    # ============= 其他业务分析方法 =============

    def _calculate_vip_distribution(self, data: Dict[str, Any],
                                    params: Dict[str, Any]) -> UnifiedCalculationResult:
        """VIP分布分析"""
        try:
            results = {}

            if 'user_daily_data' in data:
                user_daily_data = data['user_daily_data']

                if isinstance(user_daily_data, list) and user_daily_data:
                    # 获取最新的VIP分布数据
                    latest_data = user_daily_data[-1]

                    vip_distribution = {}
                    total_users = 0

                    for vip_level in range(11):  # VIP0 到 VIP10
                        vip_count = int(latest_data.get(f'vip{vip_level}的人数', 0))
                        vip_distribution[f'vip{vip_level}'] = vip_count
                        total_users += vip_count

                    # 计算比例
                    vip_percentages = {}
                    for vip_level, count in vip_distribution.items():
                        percentage = (count / total_users * 100) if total_users > 0 else 0
                        vip_percentages[vip_level] = round(percentage, 2)

                    # VIP集中度分析
                    high_value_users = sum(vip_distribution[f'vip{i}'] for i in range(5, 11))  # VIP5-VIP10
                    mid_value_users = sum(vip_distribution[f'vip{i}'] for i in range(2, 5))  # VIP2-VIP4
                    low_value_users = sum(vip_distribution[f'vip{i}'] for i in range(0, 2))  # VIP0-VIP1

                    results['vip_distribution'] = {
                        'analysis_date': latest_data.get('日期', 'unknown'),
                        'total_users': total_users,
                        'vip_counts': vip_distribution,
                        'vip_percentages': vip_percentages,
                        'user_segments': {
                            'high_value_users': high_value_users,
                            'mid_value_users': mid_value_users,
                            'low_value_users': low_value_users,
                            'high_value_percentage': round(high_value_users / total_users * 100,
                                                           2) if total_users > 0 else 0
                        }
                    }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.VIP_DISTRIBUTION.value,
                success=len(results) > 0,
                primary_result=results.get('vip_distribution', {}).get('total_users', 0),
                detailed_results=results,
                metadata={'vip_levels_analyzed': 11},
                confidence=0.9 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.VIP_DISTRIBUTION.value, str(e))

    def _calculate_product_performance(self, data: Dict[str, Any],
                                       params: Dict[str, Any]) -> UnifiedCalculationResult:
        """产品表现分析"""
        try:
            results = {}

            if 'product_data' in data:
                product_data = data['product_data']
                product_list = product_data.get('产品列表', [])

                if product_list:
                    # 分析每个产品
                    product_analysis = []

                    for product in product_list:
                        try:
                            # 提取产品指标
                            product_price = float(product.get('产品价格', 0))
                            total_purchases = int(product.get('总购买次数', 0))
                            total_interest = float(product.get('总利息', 0))
                            daily_rate = float(product.get('每日利率', 0)) / 100
                            period_days = int(product.get('期限天数', 0))
                            current_holdings = int(product.get('持有情况', {}).get('当前持有数', 0))

                            # 计算产品表现指标
                            total_revenue = total_purchases * product_price
                            avg_roi_per_product = (daily_rate * period_days) if daily_rate > 0 else 0
                            utilization_rate = current_holdings / total_purchases if total_purchases > 0 else 0

                            product_analysis.append({
                                'product_name': product.get('产品名称', 'Unknown'),
                                'product_id': product.get('产品编号', 0),
                                'category': product.get('产品分类', 'Unknown'),
                                'price': product_price,
                                'total_purchases': total_purchases,
                                'total_revenue': round(total_revenue, self.config['precision']),
                                'total_interest_paid': total_interest,
                                'expected_roi': round(avg_roi_per_product, self.config['precision']),
                                'current_utilization_rate': round(utilization_rate, self.config['precision']),
                                'daily_rate': daily_rate,
                                'period_days': period_days
                            })

                        except Exception as e:
                            logger.warning(f"Failed to analyze product {product.get('产品名称', 'Unknown')}: {e}")

                    # 整体产品组合分析
                    if product_analysis:
                        total_revenue = sum(p['total_revenue'] for p in product_analysis)
                        total_interest = sum(p['total_interest_paid'] for p in product_analysis)
                        avg_utilization = statistics.mean([p['current_utilization_rate'] for p in product_analysis])

                        # 按表现排序
                        by_revenue = sorted(product_analysis, key=lambda x: x['total_revenue'], reverse=True)
                        by_purchases = sorted(product_analysis, key=lambda x: x['total_purchases'], reverse=True)
                        by_roi = sorted(product_analysis, key=lambda x: x['expected_roi'], reverse=True)

                        results['product_performance'] = {
                            'total_products': len(product_analysis),
                            'portfolio_total_revenue': round(total_revenue, self.config['precision']),
                            'portfolio_total_interest': round(total_interest, self.config['precision']),
                            'average_utilization_rate': round(avg_utilization, self.config['precision']),
                            'top_products_by_revenue': by_revenue[:5],
                            'top_products_by_popularity': by_purchases[:5],
                            'highest_roi_products': by_roi[:5],
                            'detailed_analysis': product_analysis
                        }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.PRODUCT_PERFORMANCE.value,
                success=len(results) > 0,
                primary_result=results.get('product_performance', {}).get('total_products', 0),
                detailed_results=results,
                metadata={
                    'products_analyzed': len(results.get('product_performance', {}).get('detailed_analysis', []))},
                confidence=0.85 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.PRODUCT_PERFORMANCE.value, str(e))

    def _calculate_expiry_analysis(self, data: Dict[str, Any],
                                   params: Dict[str, Any]) -> UnifiedCalculationResult:
        """到期分析计算"""
        try:
            results = {}

            # 今日到期分析
            if 'system_data' in data:
                system_data = data['system_data']
                expiry_overview = system_data.get('到期概览', {})

                today_expiry_count = int(expiry_overview.get('今日到期产品数', 0))
                today_expiry_amount = float(expiry_overview.get('今日到期金额', 0))
                week_expiry_count = int(expiry_overview.get('本周到期产品数', 0))
                week_expiry_amount = float(expiry_overview.get('本周到期金额', 0))

                results['expiry_overview'] = {
                    'today': {
                        'count': today_expiry_count,
                        'amount': today_expiry_amount,
                        'avg_amount_per_product': round(today_expiry_amount / today_expiry_count,
                                                        self.config['precision']) if today_expiry_count > 0 else 0
                    },
                    'this_week': {
                        'count': week_expiry_count,
                        'amount': week_expiry_amount,
                        'avg_amount_per_product': round(week_expiry_amount / week_expiry_count,
                                                        self.config['precision']) if week_expiry_count > 0 else 0,
                        'daily_average_count': round(week_expiry_count / 7, 1),
                        'daily_average_amount': round(week_expiry_amount / 7, self.config['precision'])
                    }
                }

            # 具体产品到期分析
            if 'expiry_data' in data:
                expiry_data = data['expiry_data']
                product_list = expiry_data.get('产品列表', [])

                if product_list:
                    product_expiry_analysis = []

                    for product in product_list:
                        expiry_count = int(product.get('到期数量', 0))
                        expiry_amount = float(product.get('到期金额', 0))

                        if expiry_count > 0:
                            product_expiry_analysis.append({
                                'product_name': product.get('产品名称', 'Unknown'),
                                'product_id': product.get('产品编号', 0),
                                'expiry_count': expiry_count,
                                'expiry_amount': expiry_amount,
                                'avg_expiry_value': round(expiry_amount / expiry_count, self.config['precision'])
                            })

                    if product_expiry_analysis:
                        # 按到期金额排序
                        by_amount = sorted(product_expiry_analysis, key=lambda x: x['expiry_amount'], reverse=True)
                        by_count = sorted(product_expiry_analysis, key=lambda x: x['expiry_count'], reverse=True)

                        total_expiry_amount = sum(p['expiry_amount'] for p in product_expiry_analysis)
                        total_expiry_count = sum(p['expiry_count'] for p in product_expiry_analysis)

                        results['product_expiry_analysis'] = {
                            'products_with_expiry': len(product_expiry_analysis),
                            'total_expiry_amount': round(total_expiry_amount, self.config['precision']),
                            'total_expiry_count': total_expiry_count,
                            'products_by_expiry_amount': by_amount,
                            'products_by_expiry_count': by_count,
                            'detailed_analysis': product_expiry_analysis
                        }

            # 到期风险评估
            if 'expiry_overview' in results:
                week_amount = results['expiry_overview']['this_week']['amount']
                current_balance = 0

                if 'system_data' in data:
                    current_balance = float(data['system_data'].get('总余额', 0))

                if current_balance > 0:
                    expiry_impact_ratio = week_amount / current_balance

                    if expiry_impact_ratio > 0.3:
                        risk_level = 'high'
                    elif expiry_impact_ratio > 0.15:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'

                    results['expiry_risk_assessment'] = {
                        'current_balance': current_balance,
                        'week_expiry_amount': week_amount,
                        'expiry_impact_ratio': round(expiry_impact_ratio, self.config['precision']),
                        'risk_level': risk_level,
                        'impact_percentage': f"{round(expiry_impact_ratio * 100, 2)}%"
                    }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.EXPIRY_ANALYSIS.value,
                success=len(results) > 0,
                primary_result=results.get('expiry_overview', {}).get('today', {}).get('amount', 0),
                detailed_results=results,
                metadata={'analysis_components': list(results.keys())},
                confidence=0.9 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.EXPIRY_ANALYSIS.value, str(e))

    def _calculate_scenario_simulation(self, data: Dict[str, Any],
                                       params: Dict[str, Any]) -> UnifiedCalculationResult:
        """场景模拟计算"""
        try:
            results = {}

            # 获取基础数据
            base_daily_outflow = params.get('daily_outflow', 0)
            current_balance = params.get('current_balance', 0)
            simulation_days = params.get('simulation_days', 30)

            # 从数据中提取基础值（如果参数未提供）
            if not base_daily_outflow and 'daily_data' in data:
                daily_data = data['daily_data']
                if isinstance(daily_data, list) and daily_data:
                    outflows = [float(d.get('出金', 0)) for d in daily_data]
                    base_daily_outflow = statistics.mean(outflows) if outflows else 0

            if not current_balance and 'system_data' in data:
                current_balance = float(data['system_data'].get('总余额', 0))

            # 复投率场景模拟
            reinvestment_rates = params.get('reinvestment_rates', [0.0, 0.3, 0.5, 0.7, 1.0])

            scenarios = {}
            for rate in reinvestment_rates:
                net_outflow = base_daily_outflow * (1 - rate)
                reinvested_amount = base_daily_outflow * rate

                # 计算不同时间跨度的影响
                period_savings = reinvested_amount * simulation_days
                monthly_savings = reinvested_amount * 30.44
                annual_savings = monthly_savings * 12

                # 现金跑道变化
                new_runway = current_balance / net_outflow if net_outflow > 0 else float('inf')

                scenarios[f'reinvestment_{int(rate * 100)}%'] = {
                    'reinvestment_rate': rate,
                    'daily_net_outflow': round(net_outflow, self.config['precision']),
                    'daily_reinvestment': round(reinvested_amount, self.config['precision']),
                    'period_savings': round(period_savings, self.config['precision']),
                    'monthly_savings': round(monthly_savings, self.config['precision']),
                    'annual_savings': round(annual_savings, self.config['precision']),
                    'cash_runway_days': round(new_runway, 1) if new_runway != float('inf') else 'infinite',
                    'balance_after_period': round(current_balance - net_outflow * simulation_days,
                                                  self.config['precision'])
                }

            results['reinvestment_scenarios'] = scenarios

            # 推荐最优复投率
            optimal_rate = self._calculate_optimal_reinvestment_rate(scenarios, current_balance)
            results['recommendations'] = optimal_rate

            # 🧮 如果有历史数据，进行更复杂的建模
            if 'daily_data' in data:
                historical_modeling = self._advanced_scenario_modeling(data, params)
                results['advanced_modeling'] = historical_modeling

            return UnifiedCalculationResult(
                calculation_type=CalculationType.SCENARIO_SIMULATION.value,
                success=len(results) > 0,
                primary_result=optimal_rate.get('recommended_rate', 0.5),
                detailed_results=results,
                metadata={
                    'simulation_parameters': {
                        'base_daily_outflow': base_daily_outflow,
                        'current_balance': current_balance,
                        'simulation_days': simulation_days
                    },
                    'scenarios_count': len(reinvestment_rates)
                },
                confidence=0.85 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.SCENARIO_SIMULATION.value, str(e))

    def _calculate_optimal_reinvestment_rate(self, scenarios: Dict[str, Any], current_balance: float) -> Dict[str, Any]:
        """计算最优复投率"""
        try:
            # 简化的最优化逻辑：平衡流动性和增长
            best_rate = 0.5  # 默认50%
            best_score = 0

            for scenario_name, scenario_data in scenarios.items():
                if isinstance(scenario_data, dict) and 'reinvestment_rate' in scenario_data:
                    rate = scenario_data['reinvestment_rate']
                    runway = scenario_data.get('cash_runway_days', 0)

                    # 评分函数：平衡现金跑道和复投收益
                    if isinstance(runway, (int, float)):
                        liquidity_score = min(runway / 180, 1.0)  # 最多180天满分
                        growth_score = rate  # 复投率越高增长越好
                        combined_score = 0.6 * liquidity_score + 0.4 * growth_score

                        if combined_score > best_score:
                            best_score = combined_score
                            best_rate = rate

            return {
                'recommended_rate': best_rate,
                'recommended_percentage': f"{int(best_rate * 100)}%",
                'optimization_score': round(best_score, self.config['precision']),
                'reasoning': f"平衡流动性和增长的最优选择",
                'alternative_rates': [0.3, 0.7] if best_rate == 0.5 else [best_rate - 0.2, best_rate + 0.2],
                'factors_considered': ['现金跑道', '增长潜力', '风险控制']
            }

        except Exception as e:
            logger.error(f"Optimal reinvestment calculation failed: {e}")
            return {
                'recommended_rate': 0.5,
                'reasoning': '使用默认平衡策略',
                'error': str(e)
            }

    def _advanced_scenario_modeling(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """高级场景建模 - 基于历史数据的复杂模拟"""
        try:
            daily_data = data.get('daily_data', [])
            if not daily_data or len(daily_data) < 7:
                return {"note": "历史数据不足，无法进行高级建模"}

            # 提取历史趋势
            inflows = [float(d.get('入金', 0)) for d in daily_data]
            outflows = [float(d.get('出金', 0)) for d in daily_data]
            registrations = [float(d.get('注册人数', 0)) for d in daily_data]

            # 计算趋势指标
            if len(inflows) >= 3:
                # 入金趋势
                inflow_trend = (inflows[-1] - inflows[0]) / len(inflows)
                outflow_trend = (outflows[-1] - outflows[0]) / len(outflows)

                # 波动性分析
                inflow_volatility = statistics.stdev(inflows) if len(inflows) > 1 else 0
                outflow_volatility = statistics.stdev(outflows) if len(outflows) > 1 else 0

                # 基于趋势的场景预测
                optimistic_scenario = {
                    "name": "乐观场景",
                    "inflow_growth": inflow_trend * 1.5,
                    "outflow_stability": outflow_trend * 0.8,
                    "probability": 0.25
                }

                realistic_scenario = {
                    "name": "现实场景",
                    "inflow_growth": inflow_trend,
                    "outflow_stability": outflow_trend,
                    "probability": 0.5
                }

                pessimistic_scenario = {
                    "name": "悲观场景",
                    "inflow_growth": inflow_trend * 0.5,
                    "outflow_stability": outflow_trend * 1.2,
                    "probability": 0.25
                }

                return {
                    "trend_analysis": {
                        "inflow_trend": round(inflow_trend, 2),
                        "outflow_trend": round(outflow_trend, 2),
                        "inflow_volatility": round(inflow_volatility, 2),
                        "outflow_volatility": round(outflow_volatility, 2)
                    },
                    "scenario_projections": [optimistic_scenario, realistic_scenario, pessimistic_scenario],
                    "risk_factors": {
                        "volatility_risk": "高" if (inflow_volatility + outflow_volatility) > 100000 else "中" if (
                                                                                                                              inflow_volatility + outflow_volatility) > 50000 else "低",
                        "trend_sustainability": "上升" if inflow_trend > outflow_trend else "下降" if inflow_trend < outflow_trend else "稳定"
                    }
                }

        except Exception as e:
            logger.error(f"Advanced scenario modeling failed: {e}")
            return {"error": str(e)}

    # ============= AI增强计算 =============

    async def _ai_enhanced_trend_analysis(self, data: Dict[str, Any],
                                          params: Dict[str, Any]) -> UnifiedCalculationResult:
        """AI增强趋势分析"""
        if not self.gpt_client:
            return await self._calculate_trend_analysis(data, params)

        try:
            # 准备数据摘要给AI
            numeric_data = self._extract_numeric_series(data)
            data_summary = {}

            for metric_name, values in numeric_data.items():
                if len(values) >= 3:  # 至少3个数据点
                    data_summary[metric_name] = {
                        'values': values[-10:],  # 最近10个数据点
                        'count': len(values),
                        'latest': values[-1],
                        'trend_direction': 'up' if values[-1] > values[0] else 'down' if values[-1] < values[
                            0] else 'stable'
                    }

            if not data_summary:
                return await self._calculate_trend_analysis(data, params)

            # AI分析提示
            prompt = f"""
            作为数据分析专家，请分析以下财务指标的趋势模式：

            数据摘要:
            {json.dumps(data_summary, ensure_ascii=False, indent=2)}

            请为每个指标提供：
            1. 趋势强度 (0-1)
            2. 趋势可持续性评估
            3. 潜在转折点识别
            4. 置信度评估
            5. 关键驱动因素分析

            返回JSON格式，包含每个指标的详细分析。
            """

            # 调用AI分析
            ai_response = await self._call_ai_analysis(prompt, "gpt")

            if ai_response.get('success'):
                try:
                    ai_analysis = json.loads(ai_response.get('response', '{}'))

                    # 合并AI分析和基础计算
                    basic_result = await self._calculate_trend_analysis(data, params)

                    # 增强基础结果
                    enhanced_results = basic_result.detailed_results.copy()
                    for metric_name in enhanced_results:
                        if metric_name in ai_analysis:
                            enhanced_results[metric_name].update({
                                'ai_trend_strength': ai_analysis[metric_name].get('trend_strength', 0.5),
                                'ai_sustainability': ai_analysis[metric_name].get('sustainability', 'unknown'),
                                'ai_confidence': ai_analysis[metric_name].get('confidence', 0.7),
                                'ai_insights': ai_analysis[metric_name].get('insights', [])
                            })

                    return UnifiedCalculationResult(
                        calculation_type=CalculationType.TREND_ANALYSIS.value,
                        success=True,
                        primary_result=len(enhanced_results),
                        detailed_results=enhanced_results,
                        metadata={
                            'analysis_method': 'ai_enhanced_regression',
                            'ai_enhanced': True,
                            'metrics_analyzed': list(enhanced_results.keys())
                        },
                        confidence=min(basic_result.confidence + 0.1, 1.0)
                    )

                except json.JSONDecodeError:
                    logger.warning("AI响应JSON解析失败，使用基础分析")
                    return await self._calculate_trend_analysis(data, params)
            else:
                return await self._calculate_trend_analysis(data, params)

        except Exception as e:
            logger.error(f"AI enhanced trend analysis failed: {e}")
            return await self._calculate_trend_analysis(data, params)

    async def _call_ai_analysis(self, prompt: str, client_type: str = "gpt") -> Dict[str, Any]:
        """调用AI进行分析"""
        try:
            if client_type == "gpt" and self.gpt_client:
                # 实际的GPT调用逻辑
                if hasattr(self.gpt_client, 'chat') and hasattr(self.gpt_client.chat, 'completions'):
                    response = await asyncio.to_thread(
                        self.gpt_client.chat.completions.create,
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000
                    )
                    return {
                        'success': True,
                        'response': response.choices[0].message.content
                    }
                else:
                    return {'success': False, 'error': 'GPT client method not available'}
            else:
                return {'success': False, 'error': 'AI client not available'}

        except Exception as e:
            logger.error(f"AI analysis call failed: {e}")
            return {'success': False, 'error': str(e)}

    # ============= 增强的金融比率计算 =============

    def _calculate_roi_metrics(self, data: Dict[str, Any], params: Dict[str, Any]) -> UnifiedCalculationResult:
        """投资回报率计算"""
        try:
            results = {}

            # 系统整体ROI
            if 'system_data' in data:
                system_data = data['system_data']
                total_investment = float(system_data.get('总投资金额', 0))
                total_rewards = float(system_data.get('总奖励发放', 0))
                current_balance = float(system_data.get('总余额', 0))

                if total_investment > 0:
                    # 各种ROI计算
                    reward_roi = total_rewards / total_investment
                    balance_roi = current_balance / total_investment
                    total_roi = (total_rewards + current_balance) / total_investment

                    results['system_roi'] = {
                        'total_investment': total_investment,
                        'total_rewards': total_rewards,
                        'current_balance': current_balance,
                        'reward_roi': round(reward_roi, self.config['precision']),
                        'balance_roi': round(balance_roi, self.config['precision']),
                        'total_roi': round(total_roi, self.config['precision']),
                        'roi_percentage': f"{round(total_roi * 100, 2)}%"
                    }

            # 用户ROI分析
            if 'user_data' in data:
                user_data = data['user_data']
                if isinstance(user_data, list):
                    user_investments = [float(u.get('总投入', 0)) for u in user_data if u.get('总投入')]
                    user_rewards = [float(u.get('累计获得奖励金额', 0)) for u in user_data if u.get('累计获得奖励金额')]
                    user_rois = [float(u.get('投报比', 0)) for u in user_data if u.get('投报比')]

                    if user_rois:
                        results['user_roi_analysis'] = {
                            'user_count': len(user_rois),
                            'avg_roi': round(statistics.mean(user_rois), self.config['precision']),
                            'median_roi': round(statistics.median(user_rois), self.config['precision']),
                            'min_roi': round(min(user_rois), self.config['precision']),
                            'max_roi': round(max(user_rois), self.config['precision']),
                            'roi_std_dev': round(statistics.stdev(user_rois), self.config['precision']) if len(
                                user_rois) > 1 else 0
                        }

            return UnifiedCalculationResult(
                calculation_type=CalculationType.ROI_CALCULATION.value,
                success=len(results) > 0,
                primary_result=results.get('system_roi', {}).get('total_roi', 0),
                detailed_results=results,
                metadata={'calculation_components': list(results.keys())},
                confidence=0.9 if len(results) > 0 else 0.0
            )

        except Exception as e:
            return self._create_error_result(CalculationType.ROI_CALCULATION.value, str(e))

    # ============= 增强的格式化工具方法 =============

    @staticmethod
    def format_currency(amount: float, precision: int = 2) -> str:
        """格式化货币显示"""
        if amount >= 100000000:  # 1亿
            return f"¥{amount / 100000000:.{precision}f}亿"
        elif amount >= 10000:  # 1万
            return f"¥{amount / 10000:.{precision}f}万"
        else:
            return f"¥{amount:.{precision}f}"

    @staticmethod
    def format_percentage(rate: float, precision: int = 2) -> str:
        """格式化百分比显示"""
        return f"{rate * 100:.{precision}f}%"

    def get_calculation_summary(self, results: List[UnifiedCalculationResult]) -> Dict[str, Any]:
        """获取计算汇总信息"""
        if not results:
            return {}

        successful_calcs = [r for r in results if r.success]

        return {
            "total_calculations": len(results),
            "successful_calculations": len(successful_calcs),
            "success_rate": len(successful_calcs) / len(results),
            "average_confidence": sum(r.confidence for r in successful_calcs) / len(
                successful_calcs) if successful_calcs else 0,
            "calculation_types": list(set(r.calculation_type for r in results)),
            "total_warnings": sum(len(r.warnings) for r in results)
        }

    def validate_calculation_request(self, calculation_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证计算请求"""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }

        try:
            # 检查计算类型
            if calculation_type not in self.get_supported_calculations():
                validation['issues'].append(f"不支持的计算类型: {calculation_type}")
                validation['is_valid'] = False

            # 检查数据完整性
            if not data or not isinstance(data, dict):
                validation['issues'].append("数据不能为空且必须是字典格式")
                validation['is_valid'] = False

            # 检查特定计算类型的数据要求
            if calculation_type == CalculationType.TREND_ANALYSIS.value:
                numeric_data = self._extract_numeric_series(data)
                if not numeric_data:
                    validation['issues'].append("趋势分析需要数值时间序列数据")
                    validation['is_valid'] = False
                else:
                    min_points = self.config['min_data_points_for_trend']
                    insufficient_series = [name for name, values in numeric_data.items()
                                           if len(values) < min_points]
                    if insufficient_series:
                        validation['warnings'].append(
                            f"以下序列数据点不足({min_points}个): {insufficient_series}"
                        )

            elif calculation_type == CalculationType.CASH_RUNWAY.value:
                has_balance = any('余额' in str(key) for key in data.keys()) or 'system_data' in data
                has_outflow = 'daily_data' in data or 'daily_outflow' in data

                if not (has_balance and has_outflow):
                    validation['issues'].append("现金跑道计算需要余额和支出数据")
                    validation['is_valid'] = False

        except Exception as e:
            validation['issues'].append(f"验证过程出错: {str(e)}")
            validation['is_valid'] = False

        return validation

    # ============= 辅助方法 =============

    def _extract_numeric_series(self, data: Dict[str, Any]) -> Dict[str, List[float]]:
        """从API数据中提取数值序列"""
        numeric_data = {}

        try:
            # 从每日数据提取时间序列
            if 'daily_data' in data and isinstance(data['daily_data'], list):
                daily_data = data['daily_data']
                numeric_fields = ['入金', '出金', '注册人数', '持仓人数']

                for field in numeric_fields:
                    values = []
                    for day_data in daily_data:
                        try:
                            value = float(day_data.get(field, 0))
                            values.append(value)
                        except (ValueError, TypeError):
                            continue

                    if values:
                        numeric_data[field] = values

                # 计算净流入
                if '入金' in numeric_data and '出金' in numeric_data:
                    net_flows = [numeric_data['入金'][i] - numeric_data['出金'][i]
                                 for i in range(min(len(numeric_data['入金']), len(numeric_data['出金'])))]
                    if net_flows:
                        numeric_data['净流入'] = net_flows

        except Exception as e:
            logger.error(f"数值序列提取失败: {e}")

        return numeric_data

    def _create_error_result(self, calc_type: str, error_msg: str) -> UnifiedCalculationResult:
        """创建错误结果"""
        return UnifiedCalculationResult(
            calculation_type=calc_type,
            success=False,
            primary_result=0.0,
            detailed_results={'error': error_msg},
            metadata={'error_details': error_msg},
            confidence=0.0,
            warnings=[f"计算错误: {error_msg}"]
        )

    def _update_stats(self, calculation_type: str, processing_time: float, success: bool):
        """更新计算统计"""
        if success:
            self.stats['successful_calculations'] += 1

        # 更新按类型统计
        if calculation_type not in self.stats['calculation_types']:
            self.stats['calculation_types'][calculation_type] = 0
        self.stats['calculation_types'][calculation_type] += 1

        # 更新平均处理时间
        total = self.stats['total_calculations']
        current_avg = self.stats['average_processing_time']
        new_avg = (current_avg * (total - 1) + processing_time) / total
        self.stats['average_processing_time'] = round(new_avg, 4)

    # ============= 外部接口 =============

    def get_supported_calculations(self) -> List[str]:
        """获取支持的计算类型"""
        return [calc_type.value for calc_type in CalculationType]

    def get_calculation_stats(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        stats = self.stats.copy()
        total = stats['total_calculations']
        if total > 0:
            stats['success_rate'] = stats['successful_calculations'] / total
            stats['ai_usage_rate'] = stats['ai_calculations'] / total
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy',
            'component_name': 'UnifiedCalculator',
            'ai_client_available': self.gpt_client is not None,
            'supported_calculations': len(self.get_supported_calculations()),
            'total_calculations_performed': self.stats['total_calculations'],
            'success_rate': self.stats['successful_calculations'] / max(1, self.stats['total_calculations']),
            'average_processing_time': self.stats['average_processing_time'],
            'timestamp': datetime.now().isoformat()
        }


# ============= 工厂函数 =============

def create_unified_calculator(gpt_client=None, precision: int = 6) -> UnifiedCalculator:
    """
    创建统一计算器实例

    Args:
        gpt_client: GPT客户端实例
        precision: 计算精度

    Returns:
        UnifiedCalculator: 统一计算器实例
    """
    return UnifiedCalculator(gpt_client, precision)


# ============= 向后兼容的别名 =============

# 为了保持向后兼容，提供别名
StatisticalCalculator = UnifiedCalculator
FinancialCalculator = UnifiedCalculator


def create_statistical_calculator(gpt_client=None) -> UnifiedCalculator:
    """向后兼容的工厂函数"""
    return create_unified_calculator(gpt_client)


def create_financial_calculator(gpt_client=None, precision: int = 4) -> UnifiedCalculator:
    """向后兼容的工厂函数"""
    return create_unified_calculator(gpt_client, precision)