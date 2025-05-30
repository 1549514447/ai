# core/analyzers/insight_generator.py
"""
💡 AI驱动的业务洞察生成器
基于真实API数据生成可执行的业务洞察和建议

核心特点:
- 基于8个真实API的数据洞察生成
- Claude专精业务理解和策略建议
- GPT-4o专精数据解读和计算验证
- 面向实际业务场景的可执行建议
- 智能风险预警和机会识别
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """洞察类型"""
    FINANCIAL_HEALTH = "financial_health"  # 财务健康状况
    CASH_FLOW_ANALYSIS = "cash_flow_analysis"  # 资金流动分析
    USER_GROWTH_INSIGHT = "user_growth_insight"  # 用户增长洞察
    PRODUCT_PERFORMANCE = "product_performance"  # 产品表现洞察
    RISK_WARNING = "risk_warning"  # 风险预警
    OPPORTUNITY_IDENTIFICATION = "opportunity"  # 机会识别
    OPERATIONAL_EFFICIENCY = "operational_efficiency"  # 运营效率
    EXPIRY_MANAGEMENT = "expiry_management"  # 到期管理


class InsightPriority(Enum):
    """洞察优先级"""
    CRITICAL = "critical"  # 紧急，需要立即行动
    HIGH = "high"  # 重要，需要尽快处理
    MEDIUM = "medium"  # 中等，建议关注
    LOW = "low"  # 一般，可以监控


class ActionType(Enum):
    """行动类型"""
    IMMEDIATE_ACTION = "immediate"  # 立即行动
    SHORT_TERM_PLAN = "short_term"  # 短期计划 (1-7天)
    MEDIUM_TERM_PLAN = "medium_term"  # 中期计划 (1-4周)
    LONG_TERM_STRATEGY = "long_term"  # 长期战略 (1个月+)
    MONITORING = "monitoring"  # 持续监控


@dataclass
class BusinessInsight:
    """业务洞察数据类"""
    insight_id: str  # 洞察ID
    insight_type: InsightType  # 洞察类型
    priority: InsightPriority  # 优先级

    # 核心内容
    title: str  # 洞察标题
    summary: str  # 洞察摘要
    detailed_analysis: str  # 详细分析

    # 支撑数据
    key_metrics: Dict[str, Any]  # 关键指标
    supporting_data: Dict[str, Any]  # 支撑数据
    confidence_score: float  # 置信度

    # 行动建议
    recommended_actions: List[Dict[str, Any]]  # 推荐行动
    expected_impact: str  # 预期影响
    implementation_difficulty: str  # 实施难度

    # 元数据
    data_sources: List[str]  # 数据来源API
    analysis_timestamp: str  # 分析时间
    applicable_timeframe: str  # 适用时间框架


@dataclass
class RecommendedAction:
    """推荐行动数据类"""
    action_id: str  # 行动ID
    action_type: ActionType  # 行动类型
    title: str  # 行动标题
    description: str  # 行动描述
    priority: InsightPriority  # 优先级

    # 执行信息
    timeline: str  # 执行时间线
    responsible_party: str  # 负责方
    required_resources: List[str]  # 需要的资源
    success_metrics: List[str]  # 成功指标

    # 影响评估
    expected_outcome: str  # 预期结果
    potential_risks: List[str]  # 潜在风险
    estimated_roi: Optional[float]  # 预估ROI


class InsightGenerator:
    """
    💡 AI驱动的业务洞察生成器

    专注于将真实API数据转化为可执行的业务洞察
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化洞察生成器

        Args:
            claude_client: Claude客户端，负责业务洞察生成
            gpt_client: GPT客户端，负责数据验证和计算
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client

        # 业务规则配置 (基于实际业务场景)
        self.business_rules = self._load_business_rules()

        # 洞察生成统计
        self.insight_stats = {
            'total_insights_generated': 0,
            'insights_by_type': {},
            'avg_confidence_score': 0.0,
            'actionable_insights': 0,
            'critical_insights': 0
        }

        logger.info("InsightGenerator initialized for real business scenarios")

    def _load_business_rules(self) -> Dict[str, Any]:
        """加载基于实际业务的规则配置"""
        return {
            # 基于 /api/sta/system 的财务健康规则
            'financial_health': {
                'healthy_growth_rate': 0.05,  # 月增长率5%以上为健康
                'risk_outflow_ratio': 0.8,  # 出金/入金比例超过80%为风险
                'liquidity_warning_days': 30,  # 资金支撑天数低于30天预警
                'balance_fluctuation_threshold': 0.15  # 余额波动超过15%需关注
            },

            # 基于 /api/sta/day 的用户增长规则
            'user_growth': {
                'healthy_daily_growth': 50,  # 日新增用户50+为健康
                'retention_warning_threshold': 0.6,  # 持仓人数/注册人数低于60%预警
                'activity_decline_threshold': 0.1,  # 活跃度下降10%需关注
                'conversion_target': 0.8  # 注册转化率目标80%
            },

            # 基于 /api/sta/product 的产品表现规则
            'product_performance': {
                'low_utilization_threshold': 0.3,  # 持有率低于30%为低利用
                'popular_product_threshold': 100,  # 购买次数100+为热门
                'expiry_concentration_risk': 0.4,  # 单日到期超过40%为集中风险
                'new_product_ramp_days': 7  # 新产品爬坡期7天
            },

            # 基于到期相关API的现金流规则
            'cash_flow_management': {
                'daily_expiry_limit': 5000000,  # 日到期金额500万以上需重点关注
                'expiry_preparation_days': 3,  # 到期前3天开始准备资金
                'reinvestment_rate_target': 0.6,  # 目标复投率60%
                'cash_reserve_ratio': 0.15  # 现金储备比例15%
            }
        }

    # ============= 核心洞察生成方法 =============

    async def generate_comprehensive_insights(self, analysis_results: List[Any],
                                              user_context: Optional[Dict[str, Any]] = None,
                                              focus_areas: List[str] = None) -> Tuple[
        List[BusinessInsight], Dict[str, Any]]:
        """
        🎯 生成综合业务洞察

        Args:
            analysis_results: 来自financial_data_analyzer的分析结果
            user_context: 用户上下文信息
            focus_areas: 重点关注领域

        Returns:
            Tuple[洞察列表, 元数据]
        """
        try:
            logger.info("💡 开始生成综合业务洞察")

            generation_start_time = datetime.now()

            # Step 1: 数据预处理和验证
            processed_data = await self._preprocess_analysis_results(analysis_results)

            # Step 2: 识别关键业务模式
            business_patterns = await self._identify_business_patterns(processed_data)

            # Step 3: 生成分类洞察
            insights = []

            # 财务健康洞察
            financial_insights = await self._generate_financial_health_insights(processed_data)
            insights.extend(financial_insights)

            # 用户增长洞察
            user_insights = await self._generate_user_growth_insights(processed_data)
            insights.extend(user_insights)

            # 产品表现洞察
            product_insights = await self._generate_product_performance_insights(processed_data)
            insights.extend(product_insights)

            # 到期管理洞察
            expiry_insights = await self._generate_expiry_management_insights(processed_data)
            insights.extend(expiry_insights)

            # 风险预警洞察
            risk_insights = await self._generate_risk_warning_insights(processed_data)
            insights.extend(risk_insights)

            # Step 4: 优先级排序和去重
            prioritized_insights = self._prioritize_and_deduplicate_insights(insights)

            # Step 5: 生成可执行行动建议
            for insight in prioritized_insights:
                insight.recommended_actions = await self._generate_actionable_recommendations(insight, processed_data)

            # Step 6: 质量验证和置信度调整
            validated_insights = await self._validate_insights_quality(prioritized_insights)

            # 生成元数据
            generation_time = (datetime.now() - generation_start_time).total_seconds()
            metadata = {
                'generation_time': generation_time,
                'total_insights': len(validated_insights),
                'insights_by_priority': self._count_insights_by_priority(validated_insights),
                'data_sources_used': self._extract_data_sources(processed_data),
                'confidence_distribution': self._calculate_confidence_distribution(validated_insights),
                'actionable_insights_ratio': sum(1 for i in validated_insights if i.recommended_actions) / len(
                    validated_insights) if validated_insights else 0
            }

            # 更新统计信息
            self._update_insight_stats(validated_insights)

            logger.info(f"✅ 生成{len(validated_insights)}条业务洞察，耗时{generation_time:.2f}秒")

            return validated_insights, metadata

        except Exception as e:
            logger.error(f"❌ 业务洞察生成失败: {str(e)}")
            return [], {'error': str(e)}

    # ============= 财务健康洞察生成 =============

    async def _generate_financial_health_insights(self, processed_data: Dict[str, Any]) -> List[BusinessInsight]:
        """生成财务健康洞察"""
        insights = []

        try:
            # 从系统数据提取财务指标
            system_data = processed_data.get('system_data', {})
            daily_data = processed_data.get('daily_trends', {})

            total_balance = float(system_data.get('总余额', 0))
            total_inflow = float(system_data.get('总入金', 0))
            total_outflow = float(system_data.get('总出金', 0))

            # 计算关键财务比率
            outflow_ratio = total_outflow / total_inflow if total_inflow > 0 else 0
            net_flow = total_inflow - total_outflow

            # 基于业务规则生成洞察
            rules = self.business_rules['financial_health']

            # 洞察1: 资金流动健康度
            if outflow_ratio > rules['risk_outflow_ratio']:
                insight = await self._create_financial_risk_insight(outflow_ratio, net_flow, system_data)
                insights.append(insight)
            elif outflow_ratio < 0.5:
                insight = await self._create_financial_opportunity_insight(outflow_ratio, net_flow, system_data)
                insights.append(insight)

            # 洞察2: 余额增长趋势
            if daily_data.get('balance_growth_rate'):
                growth_rate = daily_data['balance_growth_rate']
                if growth_rate < rules['healthy_growth_rate']:
                    insight = await self._create_growth_concern_insight(growth_rate, daily_data)
                    insights.append(insight)

            # 洞察3: 流动性风险评估
            daily_outflow_avg = daily_data.get('avg_daily_outflow', 0)
            if daily_outflow_avg > 0:
                sustainability_days = total_balance / daily_outflow_avg
                if sustainability_days < rules['liquidity_warning_days']:
                    insight = await self._create_liquidity_warning_insight(sustainability_days, total_balance,
                                                                           daily_outflow_avg)
                    insights.append(insight)

        except Exception as e:
            logger.error(f"财务健康洞察生成失败: {str(e)}")

        return insights

    async def _create_financial_risk_insight(self, outflow_ratio: float, net_flow: float,
                                             system_data: Dict) -> BusinessInsight:
        """创建财务风险洞察"""

        # 使用Claude生成专业分析
        if self.claude_client:
            analysis_prompt = f"""
            作为金融风险分析专家，请分析以下财务状况：

            关键指标：
            - 出金/入金比例: {outflow_ratio:.2%}
            - 净现金流: ¥{net_flow:,.0f}
            - 总余额: ¥{system_data.get('总余额', 0):,.0f}

            出金比例超过80%，存在现金流风险。请提供：
            1. 风险程度评估
            2. 可能的原因分析
            3. 具体的风险控制建议

            要求简洁专业，重点突出可执行的建议。
            """

            claude_result = await self.claude_client.analyze_complex_query(analysis_prompt, system_data)

            if claude_result.get('success'):
                analysis_text = claude_result.get('analysis', {}).get('detailed_analysis', '')
                recommendations = claude_result.get('analysis', {}).get('recommendations', [])
            else:
                analysis_text = f"出金比例{outflow_ratio:.1%}超过正常水平，需要加强现金流管理。"
                recommendations = ["控制大额出金审批", "提高复投激励", "加强流动性监控"]
        else:
            analysis_text = f"当前出金比例为{outflow_ratio:.1%}，超过健康阈值80%，存在现金流压力。"
            recommendations = ["加强出金审核", "优化复投策略"]

        return BusinessInsight(
            insight_id=f"financial_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.RISK_WARNING,
            priority=InsightPriority.HIGH if outflow_ratio > 0.9 else InsightPriority.MEDIUM,

            title="现金流风险预警",
            summary=f"出金比例{outflow_ratio:.1%}偏高，存在流动性风险",
            detailed_analysis=analysis_text,

            key_metrics={
                'outflow_ratio': outflow_ratio,
                'net_cash_flow': net_flow,
                'total_balance': system_data.get('总余额', 0),
                'risk_level': 'high' if outflow_ratio > 0.9 else 'medium'
            },
            supporting_data=system_data,
            confidence_score=0.85,

            recommended_actions=[],  # 将在后续步骤中生成
            expected_impact="降低现金流风险，提升资金安全性",
            implementation_difficulty="中等",

            data_sources=['/api/sta/system'],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="立即执行，持续监控"
        )

    # ============= 用户增长洞察生成 =============

    async def _generate_user_growth_insights(self, processed_data: Dict[str, Any]) -> List[BusinessInsight]:
        """生成用户增长洞察"""
        insights = []

        try:
            daily_data = processed_data.get('daily_trends', {})
            user_data = processed_data.get('user_statistics', {})

            # 获取用户增长数据
            avg_daily_registrations = daily_data.get('avg_daily_registrations', 0)
            avg_active_users = daily_data.get('avg_active_users', 0)
            user_growth_rate = daily_data.get('user_growth_rate', 0)

            rules = self.business_rules['user_growth']

            # 洞察1: 用户增长速度分析
            if avg_daily_registrations < rules['healthy_daily_growth']:
                insight = await self._create_user_growth_concern_insight(avg_daily_registrations, user_growth_rate,
                                                                         daily_data)
                insights.append(insight)
            elif user_growth_rate > 0.2:  # 20%以上为高增长
                insight = await self._create_user_growth_opportunity_insight(avg_daily_registrations, user_growth_rate,
                                                                             daily_data)
                insights.append(insight)

            # 洞察2: 用户活跃度分析
            if avg_active_users > 0 and avg_daily_registrations > 0:
                activation_rate = avg_active_users / avg_daily_registrations
                if activation_rate < rules['retention_warning_threshold']:
                    insight = await self._create_user_activation_insight(activation_rate, user_data)
                    insights.append(insight)

        except Exception as e:
            logger.error(f"用户增长洞察生成失败: {str(e)}")

        return insights

    async def _create_user_growth_concern_insight(self, avg_registrations: float, growth_rate: float,
                                                  daily_data: Dict) -> BusinessInsight:
        """创建用户增长关注洞察"""

        analysis_text = f"日均新增用户{avg_registrations:.0f}人，低于健康水平50人，增长率{growth_rate:.1%}需要提升。"

        if self.claude_client:
            analysis_prompt = f"""
            分析用户增长现状并提供改进建议：

            当前数据：
            - 日均新增用户: {avg_registrations:.0f}人
            - 用户增长率: {growth_rate:.1%}
            - 目标增长: 50人/天

            请提供：
            1. 增长缓慢的可能原因
            2. 具体的获客策略建议
            3. 短期可执行的行动方案
            """

            claude_result = await self.claude_client.analyze_complex_query(analysis_prompt, daily_data)
            if claude_result.get('success'):
                analysis_text = claude_result.get('analysis', {}).get('detailed_analysis', analysis_text)

        return BusinessInsight(
            insight_id=f"user_growth_concern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.USER_GROWTH_INSIGHT,
            priority=InsightPriority.MEDIUM,

            title="用户增长速度待优化",
            summary=f"日均新增{avg_registrations:.0f}人，建议加强获客措施",
            detailed_analysis=analysis_text,

            key_metrics={
                'avg_daily_registrations': avg_registrations,
                'growth_rate': growth_rate,
                'target_registrations': 50,
                'gap': 50 - avg_registrations
            },
            supporting_data=daily_data,
            confidence_score=0.8,

            recommended_actions=[],
            expected_impact="提升用户获取效率，扩大用户基数",
            implementation_difficulty="中等",

            data_sources=['/api/sta/day', '/api/sta/user_daily'],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="1-2周内制定并执行获客计划"
        )

    # ============= 产品表现洞察生成 =============

    async def _generate_product_performance_insights(self, processed_data: Dict[str, Any]) -> List[BusinessInsight]:
        """生成产品表现洞察"""
        insights = []

        try:
            product_data = processed_data.get('product_data', {})

            if not product_data:
                return insights

            product_list = product_data.get('产品列表', [])

            # 分析产品利用率
            low_utilization_products = []
            high_performance_products = []

            for product in product_list:
                purchase_count = product.get('总购买次数', 0)
                current_holdings = product.get('当前持有数', 0)

                # 计算利用率指标
                if purchase_count > 0:
                    utilization_rate = current_holdings / purchase_count

                    if utilization_rate < self.business_rules['product_performance']['low_utilization_threshold']:
                        low_utilization_products.append(product)
                    elif purchase_count > self.business_rules['product_performance']['popular_product_threshold']:
                        high_performance_products.append(product)

            # 生成低利用率产品洞察
            if low_utilization_products:
                insight = await self._create_product_utilization_insight(low_utilization_products)
                insights.append(insight)

            # 生成高表现产品洞察
            if high_performance_products:
                insight = await self._create_product_opportunity_insight(high_performance_products)
                insights.append(insight)

        except Exception as e:
            logger.error(f"产品表现洞察生成失败: {str(e)}")

        return insights

    async def _create_product_utilization_insight(self, low_utilization_products: List[Dict]) -> BusinessInsight:
        """创建产品利用率洞察"""

        product_names = [p.get('产品名称', 'Unknown') for p in low_utilization_products]

        analysis_text = f"发现{len(low_utilization_products)}个产品利用率偏低，包括：{', '.join(product_names[:3])}等。"

        if self.claude_client:
            analysis_prompt = f"""
            分析产品利用率偏低的原因并提供优化建议：

            低利用率产品数量: {len(low_utilization_products)}
            产品示例: {product_names[:3]}

            请分析：
            1. 可能的原因（价格、期限、收益等）
            2. 优化建议（调整策略、营销推广等）
            3. 具体执行方案
            """

            claude_result = await self.claude_client.analyze_complex_query(analysis_prompt,
                                                                           {'products': low_utilization_products})
            if claude_result.get('success'):
                analysis_text = claude_result.get('analysis', {}).get('detailed_analysis', analysis_text)

        return BusinessInsight(
            insight_id=f"product_utilization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.PRODUCT_PERFORMANCE,
            priority=InsightPriority.MEDIUM,

            title="部分产品利用率待提升",
            summary=f"{len(low_utilization_products)}个产品需要优化推广策略",
            detailed_analysis=analysis_text,

            key_metrics={
                'low_utilization_count': len(low_utilization_products),
                'affected_products': product_names,
                'avg_utilization_rate': sum(
                    p.get('当前持有数', 0) / max(p.get('总购买次数', 1), 1) for p in low_utilization_products) / len(
                    low_utilization_products)
            },
            supporting_data={'products': low_utilization_products},
            confidence_score=0.75,

            recommended_actions=[],
            expected_impact="提升产品销售效率，优化产品组合",
            implementation_difficulty="中等",

            data_sources=['/api/sta/product'],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="2-4周内调整产品策略"
        )

    # ============= 到期管理洞察生成 =============

    async def _generate_expiry_management_insights(self, processed_data: Dict[str, Any]) -> List[BusinessInsight]:
        """生成到期管理洞察"""
        insights = []

        try:
            expiry_data = processed_data.get('expiry_analysis', {})

            if not expiry_data:
                return insights

            # 分析到期集中度风险
            daily_expiry_amounts = expiry_data.get('daily_expiry_amounts', {})
            total_expiry = sum(daily_expiry_amounts.values()) if daily_expiry_amounts else 0

            rules = self.business_rules['cash_flow_management']

            # 检查单日到期风险
            high_expiry_days = []
            for date, amount in daily_expiry_amounts.items():
                if amount > rules['daily_expiry_limit']:
                    high_expiry_days.append({'date': date, 'amount': amount})

            if high_expiry_days:
                insight = await self._create_expiry_concentration_insight(high_expiry_days, total_expiry)
                insights.append(insight)

            # 分析复投率机会
            expiry_trends = expiry_data.get('expiry_trends', {})
            if expiry_trends:
                insight = await self._create_reinvestment_opportunity_insight(expiry_trends)
                insights.append(insight)

        except Exception as e:
            logger.error(f"到期管理洞察生成失败: {str(e)}")

        return insights

    async def _create_expiry_concentration_insight(self, high_expiry_days: List[Dict],
                                                   total_expiry: float) -> BusinessInsight:
        """创建到期集中风险洞察"""

        max_expiry_day = max(high_expiry_days, key=lambda x: x['amount'])
        concentration_ratio = max_expiry_day['amount'] / total_expiry if total_expiry > 0 else 0

        analysis_text = f"发现{len(high_expiry_days)}天存在大额到期，最高单日{max_expiry_day['amount']:,.0f}元，集中度{concentration_ratio:.1%}。"

        if self.claude_client:
            analysis_prompt = f"""
            分析到期集中风险并提供资金管理建议：

            高风险到期天数: {len(high_expiry_days)}天
            最大单日到期: ¥{max_expiry_day['amount']:,.0f}
            到期集中度: {concentration_ratio:.1%}

            请提供：
            1. 资金准备建议
            2. 风险缓解措施
            3. 复投推广策略
            """

            claude_result = await self.claude_client.analyze_complex_query(analysis_prompt,
                                                                           {'expiry_days': high_expiry_days})
            if claude_result.get('success'):
                analysis_text = claude_result.get('analysis', {}).get('detailed_analysis', analysis_text)

        return BusinessInsight(
            insight_id=f"expiry_concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.EXPIRY_MANAGEMENT,
            priority=InsightPriority.HIGH,

            title="到期集中度风险预警",
            summary=f"{len(high_expiry_days)}天存在大额到期，需要提前准备资金",
            detailed_analysis=analysis_text,

            key_metrics={
                'high_expiry_days_count': len(high_expiry_days),
                'max_daily_expiry': max_expiry_day['amount'],
                'concentration_ratio': concentration_ratio,
                'total_expiry_amount': total_expiry
            },
            supporting_data={'high_expiry_days': high_expiry_days},
            confidence_score=0.9,

            recommended_actions=[],
            expected_impact="降低流动性风险，确保到期资金充足",
            implementation_difficulty="中等",

            data_sources=['/api/sta/product_end', '/api/sta/product_end_interval'],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="提前3-5天准备资金，长期优化到期分布"
        )

    async def _create_reinvestment_opportunity_insight(self, expiry_trends: Dict[str, Any]) -> BusinessInsight:
        """创建复投机会洞察"""

        # 分析复投潜力
        estimated_reinvestment_rate = expiry_trends.get('estimated_reinvestment_rate', 0.5)
        potential_reinvestment = expiry_trends.get('potential_reinvestment_amount', 0)

        analysis_text = f"基于到期趋势分析，预估复投率{estimated_reinvestment_rate:.1%}，有机会提升复投转化。"

        if self.claude_client:
            analysis_prompt = f"""
            分析复投机会并提供提升策略：

            当前预估复投率: {estimated_reinvestment_rate:.1%}
            潜在复投金额: ¥{potential_reinvestment:,.0f}
            目标复投率: 60%

            请提供：
            1. 复投率提升的策略建议
            2. 激励措施设计
            3. 具体执行时间点
            """

            claude_result = await self.claude_client.analyze_complex_query(analysis_prompt, expiry_trends)
            if claude_result.get('success'):
                analysis_text = claude_result.get('analysis', {}).get('detailed_analysis', analysis_text)

        return BusinessInsight(
            insight_id=f"reinvestment_opportunity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.OPPORTUNITY_IDENTIFICATION,
            priority=InsightPriority.MEDIUM,

            title="复投率提升机会",
            summary=f"当前复投率{estimated_reinvestment_rate:.1%}，存在提升空间",
            detailed_analysis=analysis_text,

            key_metrics={
                'current_reinvestment_rate': estimated_reinvestment_rate,
                'target_reinvestment_rate': 0.6,
                'potential_increase': 0.6 - estimated_reinvestment_rate,
                'potential_reinvestment_amount': potential_reinvestment
            },
            supporting_data=expiry_trends,
            confidence_score=0.75,

            recommended_actions=[],
            expected_impact="提升资金留存率，减少现金流出压力",
            implementation_difficulty="容易",

            data_sources=['/api/sta/product_end_interval'],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="到期前1周开始推广，持续优化"
        )

    # ============= 风险预警洞察生成 =============

    async def _generate_risk_warning_insights(self, processed_data: Dict[str, Any]) -> List[BusinessInsight]:
        """生成风险预警洞察"""
        insights = []

        try:
            # 综合风险评估
            system_data = processed_data.get('system_data', {})
            daily_trends = processed_data.get('daily_trends', {})
            anomalies = processed_data.get('detected_anomalies', [])

            # 异常数据风险
            if len(anomalies) > 5:  # 异常点过多
                insight = await self._create_data_anomaly_risk_insight(anomalies)
                insights.append(insight)

            # 业务连续性风险
            volatility = daily_trends.get('volatility', 0)
            if volatility > 0.2:  # 波动性过高
                insight = await self._create_business_continuity_risk_insight(volatility, daily_trends)
                insights.append(insight)

            # 流动性风险 (基于系统数据)
            total_balance = float(system_data.get('总余额', 0))
            daily_outflow = daily_trends.get('avg_daily_outflow', 0)

            if daily_outflow > 0:
                liquidity_days = total_balance / daily_outflow
                if liquidity_days < 15:  # 资金支撑不到15天
                    insight = await self._create_liquidity_critical_risk_insight(liquidity_days, total_balance,
                                                                                 daily_outflow)
                    insights.append(insight)

        except Exception as e:
            logger.error(f"风险预警洞察生成失败: {str(e)}")

        return insights

    async def _create_liquidity_critical_risk_insight(self, liquidity_days: float,
                                                      total_balance: float, daily_outflow: float) -> BusinessInsight:
        """创建流动性严重风险洞察"""

        analysis_text = f"按当前出金速度，资金仅能支撑{liquidity_days:.1f}天，存在严重流动性风险。"

        if self.claude_client:
            analysis_prompt = f"""
            紧急流动性风险分析和应对方案：

            当前资金余额: ¥{total_balance:,.0f}
            日均出金: ¥{daily_outflow:,.0f}
            支撑天数: {liquidity_days:.1f}天

            这是紧急情况，请提供：
            1. 立即执行的风险控制措施
            2. 资金筹措建议
            3. 紧急预案启动条件
            """

            claude_result = await self.claude_client.analyze_complex_query(
                analysis_prompt,
                {'balance': total_balance, 'outflow': daily_outflow}
            )
            if claude_result.get('success'):
                analysis_text = claude_result.get('analysis', {}).get('detailed_analysis', analysis_text)

        return BusinessInsight(
            insight_id=f"liquidity_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.RISK_WARNING,
            priority=InsightPriority.CRITICAL,

            title="流动性严重风险预警",
            summary=f"资金仅能支撑{liquidity_days:.1f}天，需要立即行动",
            detailed_analysis=analysis_text,

            key_metrics={
                'liquidity_days': liquidity_days,
                'total_balance': total_balance,
                'daily_outflow': daily_outflow,
                'risk_level': 'critical',
                'urgency': 'immediate'
            },
            supporting_data={'balance': total_balance, 'outflow': daily_outflow},
            confidence_score=0.95,

            recommended_actions=[],
            expected_impact="避免流动性危机，维护业务连续性",
            implementation_difficulty="高",

            data_sources=['/api/sta/system', '/api/sta/day'],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="立即执行，24小时内制定应对方案"
        )

    # ============= 可执行行动建议生成 =============

    async def _generate_actionable_recommendations(self, insight: BusinessInsight,
                                                   processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为洞察生成可执行的行动建议"""
        try:
            actions = []

            # 根据洞察类型生成相应的行动建议
            if insight.insight_type == InsightType.RISK_WARNING:
                actions = await self._generate_risk_mitigation_actions(insight, processed_data)
            elif insight.insight_type == InsightType.OPPORTUNITY_IDENTIFICATION:
                actions = await self._generate_opportunity_capture_actions(insight, processed_data)
            elif insight.insight_type == InsightType.FINANCIAL_HEALTH:
                actions = await self._generate_financial_optimization_actions(insight, processed_data)
            elif insight.insight_type == InsightType.USER_GROWTH_INSIGHT:
                actions = await self._generate_user_growth_actions(insight, processed_data)
            elif insight.insight_type == InsightType.EXPIRY_MANAGEMENT:
                actions = await self._generate_expiry_management_actions(insight, processed_data)
            else:
                actions = await self._generate_generic_actions(insight, processed_data)

            return actions

        except Exception as e:
            logger.error(f"行动建议生成失败: {str(e)}")
            return []

    async def _generate_risk_mitigation_actions(self, insight: BusinessInsight,
                                                processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成风险缓解行动"""
        actions = []

        if insight.priority == InsightPriority.CRITICAL:
            # 紧急风险控制
            actions.append({
                'action_id': f"emergency_control_{datetime.now().strftime('%H%M%S')}",
                'action_type': ActionType.IMMEDIATE_ACTION,
                'title': "启动紧急风险控制",
                'description': "立即限制大额出金，启动风险应急预案",
                'timeline': "立即执行",
                'responsible_party': "风控部门",
                'priority': InsightPriority.CRITICAL,
                'success_metrics': ["出金量控制在安全范围", "流动性比例提升到安全水平"]
            })

            actions.append({
                'action_id': f"emergency_funding_{datetime.now().strftime('%H%M%S')}",
                'action_type': ActionType.IMMEDIATE_ACTION,
                'title': "紧急资金筹措",
                'description': "启动紧急融资渠道，确保短期流动性",
                'timeline': "24小时内",
                'responsible_party': "财务部门",
                'priority': InsightPriority.CRITICAL,
                'success_metrics': ["获得应急资金", "流动性天数延长至30天以上"]
            })

        else:
            # 常规风险管理
            actions.append({
                'action_id': f"risk_monitoring_{datetime.now().strftime('%H%M%S')}",
                'action_type': ActionType.SHORT_TERM_PLAN,
                'title': "加强风险监控",
                'description': "建立日度风险监控机制，设置预警阈值",
                'timeline': "1周内建立",
                'responsible_party': "风控团队",
                'priority': insight.priority,
                'success_metrics': ["风险监控体系建立", "预警机制正常运行"]
            })

        return actions

    async def _generate_opportunity_capture_actions(self, insight: BusinessInsight,
                                                    processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成机会捕获行动"""
        actions = []

        if insight.insight_type == InsightType.OPPORTUNITY_IDENTIFICATION:
            # 复投率提升行动
            if 'reinvestment' in insight.title.lower():
                actions.append({
                    'action_id': f"reinvest_promotion_{datetime.now().strftime('%H%M%S')}",
                    'action_type': ActionType.SHORT_TERM_PLAN,
                    'title': "复投激励活动",
                    'description': "推出复投奖励计划，提高到期资金留存率",
                    'timeline': "2周内设计并实施",
                    'responsible_party': "产品运营部",
                    'priority': insight.priority,
                    'success_metrics': ["复投率提升5%", "资金留存率改善"]
                })

                actions.append({
                    'action_id': f"expiry_communication_{datetime.now().strftime('%H%M%S')}",
                    'action_type': ActionType.MEDIUM_TERM_PLAN,
                    'title': "到期提醒优化",
                    'description': "优化到期提醒机制，增加复投引导",
                    'timeline': "1个月内完成",
                    'responsible_party': "技术开发部",
                    'priority': InsightPriority.MEDIUM,
                    'success_metrics': ["提醒触达率95%", "复投转化率提升"]
                })

        return actions

    async def _generate_expiry_management_actions(self, insight: BusinessInsight,
                                                  processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成到期管理行动"""
        actions = []

        # 资金准备行动
        actions.append({
            'action_id': f"cash_preparation_{datetime.now().strftime('%H%M%S')}",
            'action_type': ActionType.SHORT_TERM_PLAN,
            'title': "大额到期资金准备",
            'description': "提前3天准备大额到期所需资金，确保充足流动性",
            'timeline': "到期前3天",
            'responsible_party': "财务部门",
            'priority': insight.priority,
            'success_metrics': ["资金准备充足", "到期兑付100%"]
        })

        # 到期分布优化
        actions.append({
            'action_id': f"expiry_optimization_{datetime.now().strftime('%H%M%S')}",
            'action_type': ActionType.MEDIUM_TERM_PLAN,
            'title': "优化到期分布",
            'description': "调整产品期限设计，分散到期集中度风险",
            'timeline': "1个月内调整",
            'responsible_party': "产品设计部",
            'priority': InsightPriority.MEDIUM,
            'success_metrics': ["单日到期占比降低至30%以下", "到期分布更均匀"]
        })

        return actions

    # ============= 洞察处理和验证 =============

    async def _preprocess_analysis_results(self, analysis_results: List[Any]) -> Dict[str, Any]:
        """预处理分析结果"""
        processed_data = {
            'system_data': {},
            'daily_trends': {},
            'user_statistics': {},
            'product_data': {},
            'expiry_analysis': {},
            'detected_anomalies': []
        }

        try:
            for result in analysis_results:
                if hasattr(result, 'analysis_type'):
                    if result.analysis_type.value == 'trend_analysis':
                        processed_data['daily_trends'] = {
                            'growth_rate': result.metrics.get('growth_rate', 0),
                            'volatility': result.metrics.get('volatility', 0),
                            'trend_direction': result.metrics.get('trend_direction', 'stable'),
                            'avg_daily_outflow': result.metrics.get('mean_value', 0)
                        }
                    elif result.analysis_type.value == 'anomaly_detection':
                        processed_data['detected_anomalies'] = result.anomalies

                # 处理其他数据源
                if hasattr(result, 'supporting_data'):
                    if '系统' in str(result.supporting_data) or 'system' in str(result.supporting_data):
                        processed_data['system_data'].update(result.supporting_data)

        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")

        return processed_data

    async def _identify_business_patterns(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """识别业务模式"""
        patterns = {
            'growth_pattern': 'stable',
            'risk_level': 'low',
            'user_engagement': 'normal',
            'cash_flow_health': 'good'
        }

        try:
            # 分析增长模式
            daily_trends = processed_data.get('daily_trends', {})
            growth_rate = daily_trends.get('growth_rate', 0)

            if growth_rate > 0.1:
                patterns['growth_pattern'] = 'high_growth'
            elif growth_rate < -0.05:
                patterns['growth_pattern'] = 'declining'

            # 分析风险水平
            volatility = daily_trends.get('volatility', 0)
            if volatility > 0.2:
                patterns['risk_level'] = 'high'
            elif volatility > 0.1:
                patterns['risk_level'] = 'medium'

            # 分析现金流健康度
            system_data = processed_data.get('system_data', {})
            if system_data:
                total_inflow = float(system_data.get('总入金', 0))
                total_outflow = float(system_data.get('总出金', 0))

                if total_inflow > 0:
                    outflow_ratio = total_outflow / total_inflow
                    if outflow_ratio > 0.8:
                        patterns['cash_flow_health'] = 'concerning'
                    elif outflow_ratio > 0.6:
                        patterns['cash_flow_health'] = 'moderate'

        except Exception as e:
            logger.error(f"业务模式识别失败: {str(e)}")

        return patterns

    def _prioritize_and_deduplicate_insights(self, insights: List[BusinessInsight]) -> List[BusinessInsight]:
        """优先级排序和去重"""
        if not insights:
            return []

        # 去重 - 基于洞察类型和关键指标
        unique_insights = {}
        for insight in insights:
            key = f"{insight.insight_type.value}_{insight.title}"
            if key not in unique_insights or insight.priority.value == 'critical':
                unique_insights[key] = insight

        # 按优先级排序
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_insights = sorted(
            unique_insights.values(),
            key=lambda x: (priority_order.get(x.priority.value, 4), -x.confidence_score)
        )

        return sorted_insights

    async def _validate_insights_quality(self, insights: List[BusinessInsight]) -> List[BusinessInsight]:
        """验证洞察质量"""
        validated_insights = []

        for insight in insights:
            # 基础质量检查
            if self._is_insight_valid(insight):
                # 调整置信度
                adjusted_confidence = self._adjust_confidence_score(insight)
                insight.confidence_score = adjusted_confidence

                # 只保留高质量洞察
                if adjusted_confidence >= 0.6:
                    validated_insights.append(insight)

        return validated_insights

    def _is_insight_valid(self, insight: BusinessInsight) -> bool:
        """检查洞察是否有效"""
        return (
                insight.title and
                insight.summary and
                insight.confidence_score > 0.5 and
                insight.key_metrics
        )

    def _adjust_confidence_score(self, insight: BusinessInsight) -> float:
        """调整置信度评分"""
        base_confidence = insight.confidence_score

        # 基于数据源数量调整
        data_source_bonus = min(0.1, len(insight.data_sources) * 0.02)

        # 基于关键指标数量调整
        metrics_bonus = min(0.1, len(insight.key_metrics) * 0.02)

        # 基于AI分析调整
        ai_bonus = 0.05 if self.claude_client else 0

        adjusted_confidence = min(0.95, base_confidence + data_source_bonus + metrics_bonus + ai_bonus)

        return adjusted_confidence

    # ============= 统计和工具方法 =============

    def _count_insights_by_priority(self, insights: List[BusinessInsight]) -> Dict[str, int]:
        """统计各优先级洞察数量"""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for insight in insights:
            priority = insight.priority.value
            if priority in counts:
                counts[priority] += 1

        return counts

    def _extract_data_sources(self, processed_data: Dict[str, Any]) -> List[str]:
        """提取使用的数据源"""
        sources = set()

        # 根据processed_data的内容推断数据源
        if processed_data.get('system_data'):
            sources.add('/api/sta/system')
        if processed_data.get('daily_trends'):
            sources.add('/api/sta/day')
        if processed_data.get('product_data'):
            sources.add('/api/sta/product')
        if processed_data.get('expiry_analysis'):
            sources.add('/api/sta/product_end_interval')

        return list(sources)

    def _calculate_confidence_distribution(self, insights: List[BusinessInsight]) -> Dict[str, float]:
        """计算置信度分布"""
        if not insights:
            return {'high': 0, 'medium': 0, 'low': 0}

        high_confidence = sum(1 for i in insights if i.confidence_score >= 0.8)
        medium_confidence = sum(1 for i in insights if 0.6 <= i.confidence_score < 0.8)
        low_confidence = sum(1 for i in insights if i.confidence_score < 0.6)

        total = len(insights)

        return {
            'high': high_confidence / total,
            'medium': medium_confidence / total,
            'low': low_confidence / total
        }

    def _update_insight_stats(self, insights: List[BusinessInsight]):
        """更新洞察统计信息"""
        self.insight_stats['total_insights_generated'] += len(insights)

        for insight in insights:
            insight_type = insight.insight_type.value
            if insight_type not in self.insight_stats['insights_by_type']:
                self.insight_stats['insights_by_type'][insight_type] = 0
            self.insight_stats['insights_by_type'][insight_type] += 1

            if insight.recommended_actions:
                self.insight_stats['actionable_insights'] += 1

            if insight.priority == InsightPriority.CRITICAL:
                self.insight_stats['critical_insights'] += 1

        # 更新平均置信度
        if insights:
            total_confidence = sum(i.confidence_score for i in insights)
            avg_confidence = total_confidence / len(insights)

            current_total = self.insight_stats['total_insights_generated']
            current_avg = self.insight_stats['avg_confidence_score']

            new_avg = ((current_avg * (current_total - len(insights))) + total_confidence) / current_total
            self.insight_stats['avg_confidence_score'] = new_avg

    # ============= 外部接口方法 =============

    def get_insight_stats(self) -> Dict[str, Any]:
        """获取洞察生成统计信息"""
        return self.insight_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "ai_clients": {
                "claude_available": self.claude_client is not None,
                "gpt_available": self.gpt_client is not None
            },
            "business_rules_loaded": bool(self.business_rules),
            "insight_stats": self.insight_stats,
            "timestamp": datetime.now().isoformat()
        }


# ============= 工厂函数 =============

def create_insight_generator(claude_client=None, gpt_client=None) -> InsightGenerator:
    """
    创建洞察生成器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        InsightGenerator: 洞察生成器实例
    """
    return InsightGenerator(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建洞察生成器
    generator = create_insight_generator()

    print("=== 业务洞察生成器测试 ===")

    # 模拟分析结果数据
    mock_analysis_results = []

    # 生成综合洞察
    insights, metadata = await generator.generate_comprehensive_insights(
        analysis_results=mock_analysis_results,
        user_context=None,
        focus_areas=['financial_health', 'risk_management']
    )

    print(f"生成洞察数量: {len(insights)}")
    print(f"处理时间: {metadata.get('generation_time', 0):.2f}秒")

    # 显示洞察详情
    for i, insight in enumerate(insights[:3], 1):
        print(f"\n=== 洞察 {i} ===")
        print(f"标题: {insight.title}")
        print(f"优先级: {insight.priority.value}")
        print(f"摘要: {insight.summary}")
        print(f"置信度: {insight.confidence_score:.2f}")
        print(f"数据源: {insight.data_sources}")
        print(f"行动建议数: {len(insight.recommended_actions)}")

    # 统计信息
    stats = generator.get_insight_stats()
    print(f"\n=== 统计信息 ===")
    print(f"总生成洞察: {stats['total_insights_generated']}")
    print(f"可执行洞察: {stats['actionable_insights']}")
    print(f"紧急洞察: {stats['critical_insights']}")
    print(f"平均置信度: {stats['avg_confidence_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())