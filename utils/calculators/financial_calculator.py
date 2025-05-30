# utils/calculators/financial_calculator.py
"""
💰 AI辅助的金融计算引擎
专为金融AI分析系统设计，提供高精度的金融计算和业务逻辑处理

核心特点:
- 高精度金融计算
- 复投/提现场景模拟
- 增长率和趋势分析
- 风险指标计算
- AI验证的计算逻辑
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import math
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CalculationType(Enum):
    """计算类型枚举"""
    SIMPLE = "simple"  # 简单计算
    COMPOUND = "compound"  # 复利计算
    REINVESTMENT = "reinvestment"  # 复投计算
    PREDICTION = "prediction"  # 预测计算
    RISK_ANALYSIS = "risk"  # 风险分析


@dataclass
class CalculationResult:
    """计算结果数据类"""
    result_value: float
    calculation_type: CalculationType
    calculation_steps: List[Dict[str, Any]]
    confidence_score: float  # 计算置信度
    warnings: List[str]  # 计算警告
    metadata: Dict[str, Any]  # 额外信息


@dataclass
class ReinvestmentScenario:
    """复投场景数据类"""
    reinvestment_rate: float  # 复投率
    cashout_rate: float  # 提现率
    compound_frequency: int  # 复利频率(天)
    scenario_name: str  # 场景名称


@dataclass
class GrowthAnalysis:
    """增长分析结果"""
    growth_rate: float  # 增长率
    trend_direction: str  # 趋势方向
    volatility: float  # 波动性
    confidence_level: float  # 置信水平
    supporting_data: List[float]  # 支撑数据


class FinancialCalculator:
    """
    💰 AI辅助的金融计算引擎

    功能特点:
    1. 高精度金融计算 (使用Decimal避免浮点误差)
    2. 复杂的复投/提现场景模拟
    3. 智能的增长率和趋势分析
    4. AI验证的计算逻辑
    """

    def __init__(self, gpt_client=None, precision: int = 4):
        """
        初始化金融计算器

        Args:
            gpt_client: GPT客户端，用于复杂计算验证
            precision: 计算精度 (小数位数)
        """
        self.gpt_client = gpt_client
        self.precision = precision
        self.decimal_context = Decimal('0.' + '0' * (precision - 1) + '1')

        logger.info(f"FinancialCalculator initialized with precision: {precision}")

    # ============= 核心金融计算方法 =============

    def calculate_compound_interest(self, principal: float, rate: float,
                                    periods: int, frequency: int = 1) -> CalculationResult:
        """
        计算复利收益

        Args:
            principal: 本金
            rate: 年利率 (小数形式，如0.05表示5%)
            periods: 期数
            frequency: 复利频率 (每年几次，默认1)

        Returns:
            CalculationResult: 计算结果
        """
        try:
            logger.info(f"🧮 计算复利: 本金={principal}, 利率={rate}, 期数={periods}")

            # 使用Decimal确保精度
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

            # 构建计算步骤
            steps = [
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
                },
                {
                    "step": 4,
                    "description": "计算利息收益",
                    "formula": "最终金额 - 本金",
                    "calculation": f"{final_amount:.2f} - {principal} = {interest_earned:.2f}",
                    "result": interest_earned
                }
            ]

            return CalculationResult(
                result_value=final_amount,
                calculation_type=CalculationType.COMPOUND,
                calculation_steps=steps,
                confidence_score=0.95,  # 复利计算置信度高
                warnings=[],
                metadata={
                    "principal": principal,
                    "rate": rate,
                    "periods": periods,
                    "frequency": frequency,
                    "interest_earned": interest_earned,
                    "effective_rate": (final_amount / principal - 1) * 100
                }
            )

        except Exception as e:
            logger.error(f"❌ 复利计算失败: {str(e)}")
            return self._create_error_result(CalculationType.COMPOUND, str(e))

    def calculate_reinvestment_impact(self, initial_amount: float,
                                      reinvestment_rate: float,
                                      periods: int,
                                      period_return_rate: float = 0.0) -> CalculationResult:
        """
        计算复投影响

        Args:
            initial_amount: 初始金额
            reinvestment_rate: 复投率 (0.0-1.0)
            periods: 期数
            period_return_rate: 每期收益率

        Returns:
            CalculationResult: 复投计算结果
        """
        try:
            logger.info(f"💹 计算复投影响: 金额={initial_amount}, 复投率={reinvestment_rate}")

            if not (0 <= reinvestment_rate <= 1):
                raise ValueError(f"复投率必须在0-1之间: {reinvestment_rate}")

            cashout_rate = 1 - reinvestment_rate
            total_reinvested = 0
            total_cashout = 0
            period_details = []

            current_principal = initial_amount

            for period in range(1, periods + 1):
                # 计算当期收益
                period_return = current_principal * period_return_rate
                period_total = current_principal + period_return

                # 计算复投和提现金额
                reinvest_amount = period_total * reinvestment_rate
                cashout_amount = period_total * cashout_rate

                total_reinvested += reinvest_amount
                total_cashout += cashout_amount

                # 更新下期本金 (只有复投的部分)
                current_principal = reinvest_amount

                period_details.append({
                    "period": period,
                    "period_principal": current_principal,
                    "period_return": period_return,
                    "total_before_allocation": period_total,
                    "reinvest_amount": reinvest_amount,
                    "cashout_amount": cashout_amount,
                    "cumulative_reinvested": total_reinvested,
                    "cumulative_cashout": total_cashout
                })

            # 计算最终结果
            final_balance = total_reinvested  # 最后留在系统中的资金
            total_extracted = total_cashout  # 总提现金额

            # 构建计算步骤
            steps = [
                {
                    "step": 1,
                    "description": "设定复投参数",
                    "details": {
                        "initial_amount": initial_amount,
                        "reinvestment_rate": f"{reinvestment_rate * 100:.1f}%",
                        "cashout_rate": f"{cashout_rate * 100:.1f}%",
                        "periods": periods
                    }
                },
                {
                    "step": 2,
                    "description": "逐期计算",
                    "period_breakdown": period_details
                },
                {
                    "step": 3,
                    "description": "汇总结果",
                    "summary": {
                        "total_reinvested": round(total_reinvested, 2),
                        "total_cashout": round(total_cashout, 2),
                        "final_balance": round(final_balance, 2)
                    }
                }
            ]

            # 生成警告
            warnings = []
            if reinvestment_rate < 0.2:
                warnings.append("复投率较低，可能影响长期增长潜力")
            if reinvestment_rate > 0.8:
                warnings.append("复投率较高，需要注意流动性风险")

            return CalculationResult(
                result_value=final_balance,
                calculation_type=CalculationType.REINVESTMENT,
                calculation_steps=steps,
                confidence_score=0.9,
                warnings=warnings,
                metadata={
                    "reinvestment_scenario": {
                        "reinvestment_rate": reinvestment_rate,
                        "cashout_rate": cashout_rate,
                        "total_periods": periods
                    },
                    "financial_impact": {
                        "total_reinvested": total_reinvested,
                        "total_extracted": total_cashout,
                        "final_system_balance": final_balance,
                        "extraction_ratio": total_cashout / initial_amount if initial_amount > 0 else 0
                    }
                }
            )

        except Exception as e:
            logger.error(f"❌ 复投计算失败: {str(e)}")
            return self._create_error_result(CalculationType.REINVESTMENT, str(e))

    def calculate_growth_rate(self, values: List[float], method: str = "compound") -> GrowthAnalysis:
        """
        计算增长率

        Args:
            values: 数值序列 (按时间顺序)
            method: 计算方法 ("simple", "compound", "average")

        Returns:
            GrowthAnalysis: 增长分析结果
        """
        try:
            logger.info(f"📈 计算增长率: {len(values)}个数据点, 方法={method}")

            if len(values) < 2:
                raise ValueError("至少需要2个数据点来计算增长率")

            # 过滤无效数据
            valid_values = [v for v in values if v is not None and v > 0]

            if len(valid_values) < 2:
                raise ValueError("有效数据点不足")

            growth_rate = 0
            trend_direction = "stable"

            if method == "simple":
                # 简单增长率: (最后值 - 第一值) / 第一值
                growth_rate = (valid_values[-1] - valid_values[0]) / valid_values[0]

            elif method == "compound":
                # 复合增长率: (最后值/第一值)^(1/期数) - 1
                periods = len(valid_values) - 1
                growth_rate = (valid_values[-1] / valid_values[0]) ** (1 / periods) - 1

            elif method == "average":
                # 平均增长率: 每期增长率的平均值
                period_rates = []
                for i in range(1, len(valid_values)):
                    if valid_values[i - 1] > 0:
                        rate = (valid_values[i] - valid_values[i - 1]) / valid_values[i - 1]
                        period_rates.append(rate)
                growth_rate = sum(period_rates) / len(period_rates) if period_rates else 0

            # 判断趋势方向
            if growth_rate > 0.05:  # 5%以上认为是增长
                trend_direction = "increasing"
            elif growth_rate < -0.05:  # -5%以下认为是下降
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            # 计算波动性 (标准差)
            if len(valid_values) > 1:
                mean_value = sum(valid_values) / len(valid_values)
                variance = sum((v - mean_value) ** 2 for v in valid_values) / len(valid_values)
                volatility = math.sqrt(variance) / mean_value if mean_value > 0 else 0
            else:
                volatility = 0

            # 计算置信水平
            confidence_level = min(0.95, max(0.5, 1 - volatility))

            return GrowthAnalysis(
                growth_rate=growth_rate,
                trend_direction=trend_direction,
                volatility=volatility,
                confidence_level=confidence_level,
                supporting_data=valid_values
            )

        except Exception as e:
            logger.error(f"❌ 增长率计算失败: {str(e)}")
            return GrowthAnalysis(
                growth_rate=0.0,
                trend_direction="unknown",
                volatility=0.0,
                confidence_level=0.0,
                supporting_data=[]
            )

    def calculate_financial_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """
        计算金融比率

        Args:
            financial_data: 财务数据字典

        Returns:
            Dict[str, float]: 金融比率结果
        """
        try:
            logger.info("📊 计算金融比率")

            ratios = {}

            # 获取基础数据
            total_balance = financial_data.get("总余额", 0)
            total_inflow = financial_data.get("总入金", 0)
            total_outflow = financial_data.get("总出金", 0)
            total_investment = financial_data.get("总投资金额", 0)
            total_rewards = financial_data.get("总奖励发放", 0)

            # 1. 流动性比率
            if total_outflow > 0:
                ratios["liquidity_ratio"] = total_balance / total_outflow

            # 2. 资金利用率
            if total_balance > 0:
                ratios["fund_utilization_ratio"] = total_investment / total_balance

            # 3. 收益率
            if total_investment > 0:
                ratios["return_on_investment"] = total_rewards / total_investment

            # 4. 净流入比率
            net_flow = total_inflow - total_outflow
            if total_inflow > 0:
                ratios["net_inflow_ratio"] = net_flow / total_inflow

            # 5. 提现比率
            if total_inflow > 0:
                ratios["cashout_ratio"] = total_outflow / total_inflow

            # 6. 资金增长倍数
            if total_inflow > 0:
                ratios["fund_growth_multiple"] = total_balance / total_inflow

            logger.info(f"✅ 计算出{len(ratios)}个金融比率")
            return ratios

        except Exception as e:
            logger.error(f"❌ 金融比率计算失败: {str(e)}")
            return {}

    # ============= 复杂场景计算 =============

    async def simulate_reinvestment_scenarios(self, base_data: Dict[str, Any],
                                              scenarios: List[ReinvestmentScenario],
                                              time_horizon: int = 30) -> Dict[str, Any]:
        """
        模拟多种复投场景

        Args:
            base_data: 基础数据
            scenarios: 复投场景列表
            time_horizon: 模拟时间范围(天)

        Returns:
            Dict[str, Any]: 场景模拟结果
        """
        try:
            logger.info(f"🎭 模拟{len(scenarios)}种复投场景，时间范围{time_horizon}天")

            scenario_results = {}

            for scenario in scenarios:
                logger.info(f"📊 模拟场景: {scenario.scenario_name}")

                # 使用AI验证场景合理性
                if self.gpt_client:
                    validation = await self._ai_validate_scenario(scenario, base_data)
                    if not validation.get("is_valid", True):
                        logger.warning(f"⚠️ 场景验证失败: {validation.get('reason', 'Unknown')}")
                        continue

                # 执行场景计算
                scenario_result = self._simulate_single_scenario(
                    base_data, scenario, time_horizon
                )

                scenario_results[scenario.scenario_name] = {
                    "scenario_parameters": {
                        "reinvestment_rate": scenario.reinvestment_rate,
                        "cashout_rate": scenario.cashout_rate,
                        "compound_frequency": scenario.compound_frequency
                    },
                    "simulation_results": scenario_result,
                    "risk_assessment": self._assess_scenario_risk(scenario_result),
                    "performance_metrics": self._calculate_scenario_metrics(scenario_result)
                }

            # 生成场景对比分析
            comparison_analysis = self._compare_scenarios(scenario_results)

            return {
                "scenarios": scenario_results,
                "comparison": comparison_analysis,
                "simulation_metadata": {
                    "time_horizon": time_horizon,
                    "total_scenarios": len(scenarios),
                    "simulation_date": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"❌ 场景模拟失败: {str(e)}")
            return {"error": str(e)}

    async def _ai_validate_scenario(self, scenario: ReinvestmentScenario,
                                    base_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI验证场景参数的合理性"""

        if not self.gpt_client:
            return {"is_valid": True, "confidence": 0.5}

        validation_prompt = f"""
验证以下复投场景的合理性和风险：

场景参数:
- 复投率: {scenario.reinvestment_rate * 100:.1f}%
- 提现率: {scenario.cashout_rate * 100:.1f}%
- 复利频率: {scenario.compound_frequency}天

基础数据:
{base_data}

请分析:
1. 参数是否合理
2. 是否存在明显风险
3. 是否符合业务逻辑

返回JSON格式:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "risk_level": "low/medium/high",
    "warnings": ["warning1", "warning2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""

        try:
            result = await self.gpt_client.precise_calculation({
                "calculation_type": "scenario_validation",
                "parameters": scenario.__dict__,
                "base_data": base_data
            }, {"prompt": validation_prompt})

            if result.get("success"):
                return result.get("calculation", {})
            else:
                return {"is_valid": True, "confidence": 0.5}

        except Exception as e:
            logger.error(f"AI场景验证失败: {str(e)}")
            return {"is_valid": True, "confidence": 0.5}

    def _simulate_single_scenario(self, base_data: Dict[str, Any],
                                  scenario: ReinvestmentScenario,
                                  time_horizon: int) -> Dict[str, Any]:
        """模拟单个场景"""

        initial_balance = base_data.get("总余额", 0)
        daily_return_rate = 0.001  # 假设日收益率0.1%

        daily_results = []
        current_balance = initial_balance
        total_reinvested = 0
        total_cashout = 0

        for day in range(1, time_horizon + 1):
            # 计算当日收益
            daily_return = current_balance * daily_return_rate
            total_with_return = current_balance + daily_return

            # 按复投率分配
            reinvest_amount = total_with_return * scenario.reinvestment_rate
            cashout_amount = total_with_return * scenario.cashout_rate

            # 更新累计数据
            total_reinvested += reinvest_amount
            total_cashout += cashout_amount
            current_balance = reinvest_amount  # 下期本金

            daily_results.append({
                "day": day,
                "starting_balance": current_balance,
                "daily_return": daily_return,
                "reinvest_amount": reinvest_amount,
                "cashout_amount": cashout_amount,
                "ending_balance": current_balance
            })

        return {
            "daily_breakdown": daily_results,
            "final_balance": current_balance,
            "total_reinvested": total_reinvested,
            "total_cashout": total_cashout,
            "roi": (current_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
        }

    def _assess_scenario_risk(self, scenario_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估场景风险"""

        final_balance = scenario_result.get("final_balance", 0)
        total_cashout = scenario_result.get("total_cashout", 0)

        # 计算风险指标
        liquidity_risk = "low"
        if final_balance < total_cashout * 0.1:
            liquidity_risk = "high"
        elif final_balance < total_cashout * 0.3:
            liquidity_risk = "medium"

        sustainability_risk = "low"
        roi = scenario_result.get("roi", 0)
        if roi < -0.2:
            sustainability_risk = "high"
        elif roi < 0:
            sustainability_risk = "medium"

        return {
            "liquidity_risk": liquidity_risk,
            "sustainability_risk": sustainability_risk,
            "overall_risk": "high" if liquidity_risk == "high" or sustainability_risk == "high" else "medium" if liquidity_risk == "medium" or sustainability_risk == "medium" else "low"
        }

    def _calculate_scenario_metrics(self, scenario_result: Dict[str, Any]) -> Dict[str, float]:
        """计算场景表现指标"""

        final_balance = scenario_result.get("final_balance", 0)
        total_cashout = scenario_result.get("total_cashout", 0)
        roi = scenario_result.get("roi", 0)

        return {
            "final_balance": final_balance,
            "total_extracted": total_cashout,
            "roi_percentage": roi * 100,
            "cash_extraction_ratio": total_cashout / (final_balance + total_cashout) if (
                                                                                                    final_balance + total_cashout) > 0 else 0,
            "sustainability_score": max(0, min(100, (roi + 1) * 50))  # 0-100分
        }

    def _compare_scenarios(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """对比不同场景"""

        if not scenario_results:
            return {}

        metrics = {}
        for name, result in scenario_results.items():
            perf = result.get("performance_metrics", {})
            metrics[name] = perf

        # 找出最佳和最差场景
        best_roi = max(metrics.items(), key=lambda x: x[1].get("roi_percentage", 0))
        best_balance = max(metrics.items(), key=lambda x: x[1].get("final_balance", 0))
        best_extraction = max(metrics.items(), key=lambda x: x[1].get("total_extracted", 0))

        return {
            "best_roi_scenario": {"name": best_roi[0], "roi": best_roi[1].get("roi_percentage", 0)},
            "best_balance_scenario": {"name": best_balance[0], "balance": best_balance[1].get("final_balance", 0)},
            "best_extraction_scenario": {"name": best_extraction[0],
                                         "extracted": best_extraction[1].get("total_extracted", 0)},
            "scenario_count": len(scenario_results)
        }

    # ============= 预测计算 =============

    def predict_future_value(self, current_value: float, growth_rate: float,
                             periods: int, volatility: float = 0.0) -> CalculationResult:
        """
        预测未来价值

        Args:
            current_value: 当前价值
            growth_rate: 增长率
            periods: 预测期数
            volatility: 波动性 (可选)

        Returns:
            CalculationResult: 预测结果
        """
        try:
            logger.info(f"🔮 预测未来价值: 当前={current_value}, 增长率={growth_rate}, 期数={periods}")

            # 基础预测 (指数增长)
            predicted_value = current_value * (1 + growth_rate) ** periods

            # 考虑波动性的置信区间
            confidence_interval = {}
            if volatility > 0:
                # 计算95%置信区间
                std_dev = predicted_value * volatility
                confidence_interval = {
                    "lower_bound": predicted_value - 1.96 * std_dev,
                    "upper_bound": predicted_value + 1.96 * std_dev,
                    "confidence_level": 0.95
                }

            steps = [
                {
                    "step": 1,
                    "description": "应用增长率",
                    "formula": "Future Value = Present Value × (1 + growth_rate)^periods",
                    "calculation": f"{current_value} × (1 + {growth_rate})^{periods} = {predicted_value:.2f}",
                    "result": predicted_value
                }
            ]

            if confidence_interval:
                steps.append({
                    "step": 2,
                    "description": "计算置信区间",
                    "details": confidence_interval
                })

            # 评估预测置信度
            confidence_score = max(0.3, min(0.95, 1 - volatility))

            warnings = []
            if volatility > 0.3:
                warnings.append("高波动性可能影响预测准确性")
            if periods > 30:
                warnings.append("长期预测的不确定性较高")

            return CalculationResult(
                result_value=predicted_value,
                calculation_type=CalculationType.PREDICTION,
                calculation_steps=steps,
                confidence_score=confidence_score,
                warnings=warnings,
                metadata={
                    "prediction_parameters": {
                        "current_value": current_value,
                        "growth_rate": growth_rate,
                        "periods": periods,
                        "volatility": volatility
                    },
                    "confidence_interval": confidence_interval,
                    "growth_multiple": predicted_value / current_value if current_value > 0 else 0
                }
            )

        except Exception as e:
            logger.error(f"❌ 未来价值预测失败: {str(e)}")
            return self._create_error_result(CalculationType.PREDICTION, str(e))

    # ============= 辅助方法 =============

    def _create_error_result(self, calc_type: CalculationType, error_msg: str) -> CalculationResult:
        """创建错误结果"""
        return CalculationResult(
            result_value=0.0,
            calculation_type=calc_type,
            calculation_steps=[],
            confidence_score=0.0,
            warnings=[f"计算错误: {error_msg}"],
            metadata={"error": error_msg}
        )

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

    def get_calculation_summary(self, results: List[CalculationResult]) -> Dict[str, Any]:
        """获取计算汇总信息"""

        if not results:
            return {}

        successful_calcs = [r for r in results if r.confidence_score > 0]

        return {
            "total_calculations": len(results),
            "successful_calculations": len(successful_calcs),
            "success_rate": len(successful_calcs) / len(results),
            "average_confidence": sum(r.confidence_score for r in successful_calcs) / len(
                successful_calcs) if successful_calcs else 0,
            "calculation_types": list(set(r.calculation_type.value for r in results)),
            "total_warnings": sum(len(r.warnings) for r in results)
        }


# ============= 工厂函数 =============

def create_financial_calculator(gpt_client=None, precision: int = 4) -> FinancialCalculator:
    """
    创建金融计算器实例

    Args:
        gpt_client: GPT客户端实例
        precision: 计算精度

    Returns:
        FinancialCalculator: 金融计算器实例
    """
    return FinancialCalculator(gpt_client, precision)


# ============= 预定义场景 =============

COMMON_REINVESTMENT_SCENARIOS = [
    ReinvestmentScenario(0.5, 0.5, 1, "平衡型 (50%-50%)"),
    ReinvestmentScenario(0.3, 0.7, 1, "保守型 (30%-70%)"),
    ReinvestmentScenario(0.7, 0.3, 1, "积极型 (70%-30%)"),
    ReinvestmentScenario(0.25, 0.75, 1, "超保守型 (25%-75%)"),
    ReinvestmentScenario(0.8, 0.2, 1, "超积极型 (80%-20%)")
]


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建金融计算器
    calculator = create_financial_calculator()

    print("=== 金融计算器功能测试 ===")

    # 1. 复利计算
    compound_result = calculator.calculate_compound_interest(10000, 0.05, 5)
    print(f"复利计算: ¥{compound_result.result_value:.2f}")
    print(f"置信度: {compound_result.confidence_score:.2f}")

    # 2. 复投计算
    reinvest_result = calculator.calculate_reinvestment_impact(100000, 0.3, 10, 0.01)
    print(f"复投计算: 最终余额¥{reinvest_result.result_value:.2f}")

    # 3. 增长率分析
    test_values = [100, 110, 105, 120, 118, 135, 142]
    growth_analysis = calculator.calculate_growth_rate(test_values, "compound")
    print(f"增长率: {growth_analysis.growth_rate * 100:.2f}%")
    print(f"趋势: {growth_analysis.trend_direction}")

    # 4. 金融比率
    financial_data = {
        "总余额": 8223695,
        "总入金": 17227143,
        "总出金": 9003448,
        "总投资金额": 30772686,
        "总奖励发放": 1850502
    }
    ratios = calculator.calculate_financial_ratios(financial_data)
    print(f"计算出{len(ratios)}个金融比率")

    # 5. 预测计算
    prediction = calculator.predict_future_value(1000000, 0.02, 30, 0.1)
    print(f"30期后预测值: ¥{prediction.result_value:.2f}")

    # 6. 场景模拟
    scenarios = COMMON_REINVESTMENT_SCENARIOS[:3]  # 测试前3个场景
    simulation = await calculator.simulate_reinvestment_scenarios(
        financial_data, scenarios, 30
    )
    print(f"模拟了{simulation.get('simulation_metadata', {}).get('total_scenarios', 0)}个场景")


if __name__ == "__main__":
    asyncio.run(main())