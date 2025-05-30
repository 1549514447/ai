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

from core.analyzers.query_parser import QueryAnalysisResult
from core.models.claude_client import CustomJSONEncoder

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

        # 在 InsightGenerator 类中
    async def _create_product_opportunity_insight(self, high_performance_products: List[
        Dict[str, Any]]) -> BusinessInsight:
        """
        (私有) 创建高表现产品的机会洞察。
        high_performance_products: 表现优异的产品列表，每个产品是包含名称、购买次数等的字典。
        """
        insight_id = f"prod_opportunity_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        title = "明星产品表现强劲，存在增长新机会"

        num_high_perf_products = len(high_performance_products)
        product_names_examples = [p.get('产品名称', '未知产品') for p in high_performance_products[:2]]  # 列举1-2个例子

        summary = f"发现 {num_high_perf_products} 款产品表现突出（例如：{', '.join(product_names_examples)}），购买次数多，用户反馈积极。这些明星产品是重要的增长引擎，并可能带来新的营销和交叉销售机会。"
        detailed_analysis = f"{summary} 建议深入分析这些产品的成功因素，并考虑如何复制成功经验到其他产品线，或围绕这些产品设计新的增值服务。"
        key_metrics = {
            '高表现产品数量': num_high_perf_products,
            '高表现产品示例': product_names_examples,
            '平均购买次数（高表现产品）': sum(p.get('总购买次数', 0) for p in
                                            high_performance_products) / num_high_perf_products if num_high_perf_products else 0
        }
        confidence = 0.85
        priority = InsightPriority.HIGH  # 机会通常是高优先级

        if self.claude_client and high_performance_products:
            prompt = f"""
            金融产品分析显示，以下产品表现优异，用户购买踊跃：
            {json.dumps(high_performance_products[:3], ensure_ascii=False, indent=2, cls=CustomJSONEncoder)} # 最多3个产品示例给AI

            请用中文分析：
            1. 这些“明星产品”的成功可能归因于哪些因素？（例如：收益率、期限、市场定位、推广策略等）
            2. 如何进一步利用这些产品的成功来带动整体业务增长？（例如：加大推广力度、设计关联产品、针对高价值用户进行精准营销、打包销售等）
            3. 基于这些产品的特性，是否存在新的市场细分或用户群可以拓展？

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "对明星产品成功因素和潜在机会的详细中文分析。",
            "growth_strategies_suggested": ["具体的增长策略建议1", "建议2"],
            "target_actions": ["围绕这些产品的具体行动点1", "行动点2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"high_performance_products_sample": high_performance_products[:3]}
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for product opportunity insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for product opportunity insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.OPPORTUNITY_IDENTIFICATION,
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"high_performance_products": high_performance_products},
            confidence_score=confidence,
            recommended_actions=[],
            # 由 _generate_actionable_recommendations (特别是 _generate_opportunity_capture_actions) 填充
            expected_impact="通过聚焦和推广高表现产品，带动整体销售额和用户参与度的提升。",
            implementation_difficulty="中等",
            data_sources=["/api/sta/product"],  # 主要数据来源
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="未来1-3个月重点推广"
        )

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

    async def _create_business_continuity_risk_insight(self, volatility_metric: float,
                                                       daily_trend_data: Dict[str, Any]) -> BusinessInsight:
        """
        (私有) 创建业务连续性风险洞察（例如，关键指标波动过大）。
        volatility_metric: 一个量化波动性的指标值。
        daily_trend_data: 相关的日趋势数据作为上下文。
        """
        insight_id = f"biz_continuity_risk_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        title = "业务关键指标波动异常，关注连续性风险"
        summary = f"监测到近期业务关键指标（例如资金、用户活跃度）出现显著波动（波动性指标: {volatility_metric:.2f}），可能影响业务的稳定性和连续性。"
        detailed_analysis = f"{summary} 高波动性可能预示着市场环境变化、内部运营问题或突发事件影响。建议深入分析波动来源，评估其对核心业务流程的影响，并制定应对预案。"
        key_metrics = {
            '观测到的波动性指标': volatility_metric,
            '风险等级评估': '高' if volatility_metric > 0.25 else '中'  # 示例阈值
        }
        confidence = 0.70 + (0.15 if volatility_metric > 0.25 else 0)
        priority = InsightPriority.HIGH if volatility_metric > 0.25 else InsightPriority.MEDIUM

        if self.claude_client:
            prompt = f"""
            公司业务数据显示关键指标存在较高波动性：
            - 量化的波动性指标值为: {volatility_metric:.3f} (例如，高于0.15或0.2即为显著)
            - 相关日趋势数据摘要: {json.dumps(daily_trend_data, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}

            请用中文分析：
            1. 这种高波动性对业务连续性可能造成的具体风险是什么？（例如：现金流规划困难、用户信任度下降、运营计划打乱等）
            2. 建议从哪些方面调查波动产生的原因？
            3. 为应对此类波动，可以考虑哪些风险管理或业务调整措施？

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "对波动性风险的详细中文分析。",
            "potential_continuity_risks": ["具体连续性风险1", "风险2"],
            "risk_management_suggestions": ["风险管理建议1", "建议2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"volatility_data": daily_trend_data}
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for business continuity risk insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for business continuity risk insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.RISK_WARNING,
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"daily_trends_snapshot": daily_trend_data, "calculated_volatility": volatility_metric},
            confidence_score=confidence,
            recommended_actions=[],  # 由 _generate_actionable_recommendations 填充
            expected_impact="通过识别和管理波动性，增强业务韧性，保障运营稳定。",
            implementation_difficulty="中等",
            data_sources=[f"{api_name}" for api_name in daily_trend_data.get("api_source_names", ["/api/sta/day"])],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="需持续监控，1-2周内制定应对策略"
        )

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

    async def _generate_financial_optimization_actions(self, insight: BusinessInsight,
                                                       processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        (私有) 为财务健康相关的洞察生成具体的财务优化行动建议。
        """
        actions: List[Dict[str, Any]] = []
        action_prefix = f"fin_opt_act_{insight.insight_id[-6:]}"  # 基于洞察ID生成唯一前缀

        if not self.claude_client:
            logger.warning("ClaudeClient not available for generating financial optimization actions.")
            # 提供一些基于规则的通用建议
            if insight.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]:
                actions.append({
                    'action_id': f"{action_prefix}_review_cashflow", 'title': "紧急审视现金流",
                    'description': "立即对当前现金流状况进行全面审查，识别主要出金点和潜在风险。",
                    'action_type': ActionType.IMMEDIATE_ACTION.value, 'priority': InsightPriority.HIGH.value
                })
            actions.append({
                'action_id': f"{action_prefix}_monitor_key_ratios", 'title': "监控关键财务比率",
                'description': "持续监控如流动比率、速动比率、出入金比等关键财务健康指标。",
                'action_type': ActionType.MONITORING.value, 'priority': InsightPriority.MEDIUM.value
            })
            return actions

        # 使用AI生成更具体的建议
        # insight.summary 和 insight.detailed_analysis 已经包含了AI对财务状况的分析
        # insight.key_metrics 包含了相关数据

        prompt_context = {
            "insight_title": insight.title,
            "insight_summary": insight.summary,
            "insight_priority": insight.priority.value,
            "key_financial_metrics": insight.key_metrics,  # 包含关键数据
            "current_financial_analysis": insight.detailed_analysis  # AI之前的分析
        }

        prompt = f"""
        您是一位经验丰富的首席财务官(CFO)。以下是一项关于公司财务状况的业务洞察：
        洞察标题: "{insight.title}"
        洞察摘要: "{insight.summary}"
        优先级: {insight.priority.value}
        关键相关财务指标: {json.dumps(insight.key_metrics, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        系统生成的详细分析:
        ---
        {insight.detailed_analysis}
        ---

        基于以上洞察和分析，请提出 2-3 条具体的、可操作的财务优化行动建议。
        对于每条建议，请明确：
        1.  `action_title`: 建议的简洁标题 (例如：“优化短期债务结构”)
        2.  `action_description`: 建议的详细描述和执行要点。
        3.  `action_type`: 从以下选择：{', '.join([e.value for e in ActionType])} (例如："short_term_plan")
        4.  `timeline_suggestion`: 建议的执行时间框架 (例如：“1周内启动”，“持续进行”)
        5.  `expected_outcome_summary`: 执行此建议预期的主要成果。

        请以JSON数组的格式返回这些建议，每个元素是一个包含上述键的字典。例如：
        [
            {{
                "action_title": "...",
                "action_description": "...",
                "action_type": "...",
                "timeline_suggestion": "...",
                "expected_outcome_summary": "..."
            }}
        ]
        如果洞察表明财务状况良好，建议可以是“维持并监控”或“探索新的投资机会”。
        如果洞察表明存在风险，建议应侧重于风险缓解和控制。
        """
        try:
            ai_response = await self.claude_client.analyze_complex_query(query=prompt, context=prompt_context)
            if ai_response and ai_response.get('success'):
                response_content = ai_response.get('analysis', ai_response.get('response'))
                if isinstance(response_content, str):
                    try:
                        response_content = json.loads(response_content)  # 期望返回JSON列表
                    except json.JSONDecodeError:
                        logger.error(f"AI response for financial actions is not valid JSON: {response_content[:200]}")
                        # 可以尝试从文本中解析，或返回通用建议

                if isinstance(response_content, list):
                    for idx, action_data in enumerate(response_content):
                        if isinstance(action_data, dict):
                            actions.append({
                                'action_id': f"{action_prefix}_{idx}",
                                'action_type': ActionType(
                                    action_data.get('action_type', ActionType.MEDIUM_TERM_PLAN.value)).value,
                                'title': action_data.get('action_title', f"财务优化建议 {idx + 1}"),
                                'description': action_data.get('action_description', "根据AI分析的具体建议。"),
                                'priority': insight.priority.value,  # 继承洞察的优先级或AI重新评估
                                'timeline': action_data.get('timeline_suggestion', "根据实际情况制定"),
                                'responsible_party': "财务部门/管理层",  # 通用负责人
                                'required_resources': ["财务数据分析能力", "决策权"],
                                'success_metrics': [f"{action_data.get('expected_outcome_summary', '财务指标改善')}"],
                                'expected_outcome': action_data.get('expected_outcome_summary', "改善财务健康状况"),
                                'potential_risks': ["市场变化可能影响效果"]
                            })
                        else:
                            logger.warning(f"AI returned non-dict item in actions list: {action_data}")
            else:
                logger.warning(
                    f"AI call for financial optimization actions failed or no success: {ai_response.get('error') if ai_response else 'N/A'}")
        except Exception as e:
            logger.error(f"Error generating financial optimization actions with AI: {e}")

        # 如果AI失败或没有生成具体行动，补充通用建议
        if not actions:
            actions.append({
                'action_id': f"{action_prefix}_default_monitor", 'title': "持续监控财务指标",
                'description': "定期回顾核心财务报表和关键比率，确保财务健康。",
                'action_type': ActionType.MONITORING.value, 'priority': InsightPriority.MEDIUM.value,
                'timeline': "持续进行", 'responsible_party': "财务团队",
                'expected_outcome': "及时发现财务风险和机会"
            })
        return actions
    async def _create_financial_opportunity_insight(self, outflow_ratio: float, net_flow: float,
                                                    system_data: Dict[str, Any]) -> BusinessInsight:
        """
        (私有) 创建积极财务状况下的机会洞察。
        例如：现金流充裕，出金占比低。
        """
        insight_id = f"fin_opportunity_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        title = "财务状况健康，现金流充裕"
        summary = f"当前出金与入金比例为{outflow_ratio:.1%}，净现金流为 ¥{net_flow:,.0f}，显示公司资金状况良好，存在进一步发展的机会。"
        detailed_analysis = summary  # 初始分析与摘要相同，可由AI丰富
        key_metrics = {
            '出金入金比': outflow_ratio,
            '净现金流': net_flow,
            '总余额': float(system_data.get('总余额', 0)),
            '评估': '健康且有机会'
        }
        confidence = 0.85
        priority = InsightPriority.MEDIUM  # 通常机会是中高优先级

        if self.claude_client:
            prompt = f"""
            当前公司财务数据显示：
            - 出金与入金比例: {outflow_ratio:.2%} (低于50%被认为是健康的)
            - 净现金流: ¥{net_flow:,.0f}
            - 总余额: ¥{float(system_data.get('总余额', 0)):,.0f}

            这是一个积极的财务信号。请基于此信息，用中文分析：
            1. 这种良好财务状况可能带来的具体业务机会（例如：新产品投资、市场扩张、股东分红、债务偿还等）。
            2. 如何利用当前充裕的现金流来进一步提升公司价值或降低潜在风险？
            3. 对这种机会的简要评估（例如，机会窗口、潜在回报）。

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "详细的中文分析文本。",
            "identified_opportunities": ["机会点1的描述", "机会点2的描述"],
            "strategic_suggestions": ["利用此机会的策略建议1", "建议2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"financial_data": system_data}
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})  # 假设 'analysis' 包含所需内容
                    if isinstance(analysis_content, str):  # 有时候AI可能直接返回文本
                        try:
                            analysis_content = json.loads(analysis_content)  # 尝试解析
                        except:
                            pass

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        # Opportunity insights might be directly part of the text, or structured
                        # For now, we'll assume the detailed_analysis from AI is comprehensive.
                        # Recommendations will be generated separately.
                        confidence = analysis_content.get('confidence', confidence)  # AI可能给出置信度
                else:
                    logger.warning(
                        f"AI call for financial opportunity insight failed or returned no success: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for financial opportunity insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.OPPORTUNITY_IDENTIFICATION,  # 更具体的类型
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"system_snapshot": system_data},
            confidence_score=confidence,
            recommended_actions=[],  # 将由 _generate_actionable_recommendations 填充
            expected_impact="通过有效利用现有资金优势，可能实现业务增长或风险降低。",
            implementation_difficulty="中等",  # 取决于具体机会
            data_sources=[f"{api_name}" for api_name in system_data.get("api_source_names", ["/api/sta/system"])],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="未来1-3个月"
        )

    async def _create_growth_concern_insight(self, avg_registrations: float, growth_rate: float,
                                             daily_data: Dict[str, Any]) -> BusinessInsight:
        """
        (私有) 创建用户增长缓慢或未达标的关注点洞察。
        """
        insight_id = f"user_growth_concern_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        rules = self.business_rules['user_growth']
        title = "用户增长速度需关注和提升"
        summary = f"日均新增用户约{avg_registrations:.0f}人，增长率{growth_rate:.1%}，可能低于预期（目标: {rules.get('healthy_daily_growth', 50)}人/天）。建议分析原因并采取措施。"
        detailed_analysis = summary
        key_metrics = {
            '日均新增注册': avg_registrations,
            '用户增长率': growth_rate,
            '目标日新增': rules.get('healthy_daily_growth', 50),
            '日新增缺口': rules.get('healthy_daily_growth', 50) - avg_registrations
        }
        confidence = 0.75  # 基于规则的判断，但原因分析需要AI
        priority = InsightPriority.MEDIUM

        if self.claude_client:
            prompt = f"""
            当前用户增长数据显示：
            - 日均新增用户: {avg_registrations:.0f} 人
            - 近期用户增长率: {growth_rate:.1%}
            - 业务期望的日新增用户数约为: {rules.get('healthy_daily_growth', 50)} 人

            该增长数据可能未达业务预期。请用中文分析：
            1. 用户增长缓慢或未达预期的可能原因（例如：市场推广不足、产品吸引力下降、用户体验问题、竞争对手影响等）。
            2. 针对这些可能原因，可以从哪些方面入手调查和分析？
            3. 初步提出1-2个可以考虑的提升用户增长的策略方向。

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "详细的中文分析，包括可能原因和调查方向。",
            "preliminary_strategies": ["初步策略方向1", "初步策略方向2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"daily_user_data": daily_data}  # daily_data 包含 avg_daily_registrations, user_growth_rate 等
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass
                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        # Recommendations will be generated separately if needed
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for user growth concern insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for user growth concern insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.USER_GROWTH_INSIGHT,
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"daily_user_metrics": daily_data},
            confidence_score=confidence,
            recommended_actions=[],
            expected_impact="通过针对性措施提升用户增长速率，扩大用户基础。",
            implementation_difficulty="中至高",
            data_sources=[f"{api_name}" for api_name in
                          daily_data.get("api_source_names", ["/api/sta/day", "/api/sta/user_daily"])],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="未来1个月重点关注"
        )

    async def _create_liquidity_warning_insight(self, sustainability_days: float, total_balance: float,
                                                daily_outflow_avg: float) -> BusinessInsight:
        """
        (私有) 创建流动性风险预警洞察。
        """
        insight_id = f"liquidity_warn_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        rules = self.business_rules['financial_health']
        title = "流动性风险预警：资金短期支撑能力需关注"
        summary = f"按当前日均出金约 ¥{daily_outflow_avg:,.0f} 计算，现有总余额 ¥{total_balance:,.0f} 预计可支撑 {sustainability_days:.1f} 天，低于 {rules.get('liquidity_warning_days', 30)} 天的警戒线。"
        detailed_analysis = summary
        key_metrics = {
            '资金可支撑天数': sustainability_days,
            '总余额': total_balance,
            '日均出金': daily_outflow_avg,
            '警戒线天数': rules.get('liquidity_warning_days', 30)
        }
        # 根据天数差距设定优先级和置信度
        priority = InsightPriority.HIGH
        confidence = 0.80
        if sustainability_days < (rules.get('liquidity_warning_days', 30) / 2):  # 例如少于15天
            priority = InsightPriority.CRITICAL
            confidence = 0.90
        elif sustainability_days < rules.get('liquidity_warning_days', 30) * 0.75:  # 例如少于22.5天
            priority = InsightPriority.HIGH
            confidence = 0.85

        if self.claude_client:
            prompt = f"""
            公司财务数据显示存在流动性风险：
            - 当前总余额: ¥{total_balance:,.0f}
            - 近期日均出金: ¥{daily_outflow_avg:,.0f}
            - 计算出的资金可支撑天数: {sustainability_days:.1f} 天
            - 业务设定的流动性预警天数为: {rules.get('liquidity_warning_days', 30)} 天

            当前支撑天数低于警戒线。请用中文分析：
            1. 此流动性风险的潜在原因（例如：近期大额集中出金、入金减少、投资周期错配等）。
            2. 此风险可能导致的短期和中期业务影响。
            3. 针对此情况，可以立即采取的缓解措施，以及中长期改善流动性管理的建议。

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "详细的中文风险分析及原因推测。",
            "potential_impacts_text": "潜在影响的描述。",
            "mitigation_suggestions": ["缓解措施建议1", "建议2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"balance_data": total_balance, "outflow_data": daily_outflow_avg}
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass  # Keep as string if not valid JSON

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        # You might want to add potential_impacts_text to detailed_analysis or store separately
                        impact_text = analysis_content.get('potential_impacts_text', '')
                        if impact_text:
                            detailed_analysis += f"\n\n潜在影响：{impact_text}"
                        # Recommendations will be handled by _generate_actionable_recommendations
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for liquidity warning insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for liquidity warning insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.RISK_WARNING,
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"total_balance": total_balance, "daily_outflow_avg": daily_outflow_avg,
                             "sustainability_days": sustainability_days},
            confidence_score=confidence,
            recommended_actions=[],  # 由 _generate_actionable_recommendations 填充
            expected_impact="及时应对可避免资金链紧张，保障业务正常运营。",
            implementation_difficulty="中等至高，取决于具体措施",
            data_sources=["/api/sta/system", "/api/sta/day"],  # 假设数据来源
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="立即关注，1周内采取初步措施"
        )

    async def _generate_user_growth_actions(self, insight: BusinessInsight,
                                            processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        (私有) 为用户增长相关的洞察生成具体的行动建议。
        """
        actions: List[Dict[str, Any]] = []
        action_prefix = f"user_growth_act_{insight.insight_id[-6:]}"

        if not self.claude_client:
            logger.warning("ClaudeClient not available for generating user growth actions.")
            actions.append({
                'action_id': f"{action_prefix}_analyze_channels", 'title': "分析获客渠道",
                'description': "评估不同用户获取渠道的成本和效益，优化投入。",
                'action_type': ActionType.MEDIUM_TERM_PLAN.value, 'priority': InsightPriority.MEDIUM.value
            })
            return actions

        prompt_context = {
            "insight_title": insight.title,
            "insight_summary": insight.summary,
            "insight_priority": insight.priority.value,
            "key_user_metrics": insight.key_metrics,  # 例如：日新增、增长率、活跃度等
            "current_user_analysis": insight.detailed_analysis
        }

        prompt = f"""
        您是一位经验丰富的用户增长策略师。以下是一项关于公司用户增长状况的业务洞察：
        洞察标题: "{insight.title}"
        洞察摘要: "{insight.summary}"
        优先级: {insight.priority.value}
        关键相关用户指标: {json.dumps(insight.key_metrics, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        系统生成的详细分析:
        ---
        {insight.detailed_analysis}
        ---

        基于以上洞察（特别是如果增长未达预期或存在机会），请提出 2-3 条具体的、可操作的用以提升用户增长或用户活跃度的行动建议。
        对于每条建议，请明确：
        1.  `action_title`: 建议的简洁标题 (例如：“启动新用户推荐奖励计划”)
        2.  `action_description`: 建议的详细描述和执行要点。
        3.  `action_type`: 从以下选择：{', '.join([e.value for e in ActionType])} (例如："short_term_plan")
        4.  `target_metric_to_improve`: 此建议主要针对哪个用户指标的提升 (例如：“日新增用户数”、“用户活跃率”)
        5.  `success_criteria`: 如何衡量此建议的成功 (例如：“日新增用户提升20%”)

        请以JSON数组的格式返回这些建议，每个元素是一个包含上述键的字典。
        如果洞察表明用户增长良好，建议可以是“维持现有策略并探索新渠道”或“提升高价值用户转化”。
        """
        try:
            ai_response = await self.claude_client.analyze_complex_query(query=prompt, context=prompt_context)
            if ai_response and ai_response.get('success'):
                response_content = ai_response.get('analysis', ai_response.get('response'))
                if isinstance(response_content, str):
                    try:
                        response_content = json.loads(response_content)
                    except json.JSONDecodeError:
                        logger.error(f"AI response for user growth actions is not valid JSON: {response_content[:200]}")

                if isinstance(response_content, list):
                    for idx, action_data in enumerate(response_content):
                        if isinstance(action_data, dict):
                            actions.append({
                                'action_id': f"{action_prefix}_{idx}",
                                'action_type': ActionType(
                                    action_data.get('action_type', ActionType.MEDIUM_TERM_PLAN.value)).value,
                                'title': action_data.get('action_title', f"用户增长建议 {idx + 1}"),
                                'description': action_data.get('action_description', "根据AI分析的具体用户增长建议。"),
                                'priority': insight.priority.value,
                                'timeline': action_data.get('timeline_suggestion', "根据目标制定"),  # AI 可能也返回 timeline
                                'responsible_party': "市场部门/运营部门",
                                'required_resources': ["营销预算", "运营人力"],
                                'success_metrics': [action_data.get('success_criteria',
                                                                    f"提升目标指标 {action_data.get('target_metric_to_improve', 'N/A')}")],
                                'expected_outcome': f"提升 {action_data.get('target_metric_to_improve', '用户增长相关')} 指标",
                                'potential_risks': ["市场竞争激烈", "用户偏好变化"]
                            })
            else:
                logger.warning(
                    f"AI call for user growth actions failed or no success: {ai_response.get('error') if ai_response else 'N/A'}")
        except Exception as e:
            logger.error(f"Error generating user growth actions with AI: {e}")

        if not actions:
            actions.append({
                'action_id': f"{action_prefix}_default_engagement", 'title': "提升用户活跃和留存",
                'description': "分析用户行为数据，优化产品体验，策划用户互动活动以提高用户粘性。",
                'action_type': ActionType.MEDIUM_TERM_PLAN.value, 'priority': InsightPriority.MEDIUM.value,
                'timeline': "持续优化", 'responsible_party': "产品/运营团队",
                'expected_outcome': "提高用户活跃度和留存率"
            })
        return actions

    async def _create_user_growth_opportunity_insight(self, avg_registrations: float, growth_rate: float,
                                                      daily_data: Dict[str, Any]) -> BusinessInsight:
        """
        (私有) 创建用户增长良好的机会洞察。
        """
        insight_id = f"user_growth_opp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        title = "用户增长势头强劲，可进一步扩大优势"
        summary = f"近期用户增长表现出色，日均新增用户达到约 {avg_registrations:.0f} 人，增长率高达 {growth_rate:.1%}。这是一个积极的信号，表明当前获客策略有效，并可能存在进一步扩大市场份额的机会。"
        detailed_analysis = f"{summary} 建议分析当前高效的获客渠道和用户画像，加大投入，并探索如何提高新用户的留存和转化。"
        key_metrics = {
            '日均新增注册': avg_registrations,
            '用户增长率': growth_rate,
            '增长势头评估': '强劲'
        }
        confidence = 0.90  # 基于良好数据，置信度较高
        priority = InsightPriority.HIGH

        if self.claude_client:
            prompt = f"""
            公司用户增长数据显示出强劲势头：
            - 日均新增用户: {avg_registrations:.0f} 人
            - 近期用户增长率: {growth_rate:.1%}

            这是一个非常积极的信号。请用中文分析：
            1. 这种快速增长可能的主要驱动因素是什么？（例如：成功的市场活动、产品特性吸引、口碑传播等）
            2. 如何保持并进一步加速这种增长势头？有哪些可以放大的策略？
            3. 在快速增长的同时，需要注意哪些潜在的挑战或风险？（例如：服务承载能力、新用户质量、留存问题等）

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "对增长驱动因素和持续增长策略的详细中文分析。",
            "acceleration_strategies": ["加速增长策略1", "策略2"],
            "potential_challenges": ["潜在挑战1", "挑战2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"user_growth_data": daily_data}  # daily_data 包含相关指标
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for user growth opportunity insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for user growth opportunity insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.OPPORTUNITY_IDENTIFICATION,  # 也可以是 USER_GROWTH_INSIGHT 但更偏机会
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"daily_user_metrics_snapshot": daily_data},
            confidence_score=confidence,
            recommended_actions=[],  # 由 _generate_actionable_recommendations (特别是 _generate_user_growth_actions) 填充
            expected_impact="巩固并扩大用户增长优势，快速提升市场占有率。",
            implementation_difficulty="中等",
            data_sources=[f"{api_name}" for api_name in
                          daily_data.get("api_source_names", ["/api/sta/day", "/api/sta/user_daily"])],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="未来1-3个月抓住增长窗口期"
        )



    async def _create_user_activation_insight(self, activation_rate: float,
                                              user_data_context: Dict[str, Any]) -> BusinessInsight:
        """
        (私有) 创建用户激活率偏低的改进洞察。
        activation_rate: 计算得出的用户激活率 (例如：活跃用户/总注册用户)。
        user_data_context: 相关的用户数据作为上下文。
        """
        insight_id = f"user_activation_concern_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        rules = self.business_rules['user_growth']  # 假设激活率相关的规则也在这里
        target_activation_rate = rules.get('activation_target',
                                           rules.get('retention_warning_threshold', 0.7))  # 使用一个激活目标或留存阈值

        title = "用户激活效率有提升空间"
        summary = f"当前用户激活率约为 {activation_rate:.1%}，可能低于业务目标（例如 {target_activation_rate:.0%}）。建议优化新用户引导流程和早期用户体验，以提高激活转化。"
        detailed_analysis = f"{summary} 低激活率可能意味着新用户在注册后未能充分体验产品价值，导致流失。需要关注新用户路径，识别流失节点。"
        key_metrics = {
            '用户激活率': activation_rate,
            '目标激活率': target_activation_rate,
            '激活率差距': target_activation_rate - activation_rate,
        }
        confidence = 0.78
        priority = InsightPriority.MEDIUM

        if self.claude_client:
            prompt = f"""
            公司用户数据显示，当前用户激活率（例如：活跃用户/注册用户 或 首次购买用户/注册用户）约为 {activation_rate:.1%}，
            而业务目标通常期望达到 {target_activation_rate:.0%} 或更高。

            请用中文分析：
            1. 用户激活率偏低可能存在哪些常见原因？（例如：新用户引导复杂、产品价值未被快速感知、早期体验不佳、推送消息不当等）
            2. 为了提升用户激活率，可以从哪些方面着手改进？（例如：优化注册后引导、提供新手任务或奖励、个性化内容推荐、优化首次使用体验等）
            3. 建议如何通过数据分析来定位激活流程中的具体瓶颈？

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "对激活率偏低原因和改进方向的详细中文分析。",
            "activation_improvement_strategies": ["提升激活率策略1", "策略2"],
            "bottleneck_analysis_suggestions": ["定位瓶颈的数据分析方法1", "方法2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"user_activation_data": user_data_context}  # user_data_context 应包含计算激活率的原始数据
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for user activation insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for user activation insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.USER_GROWTH_INSIGHT,  # 也可以是 OPERATIONAL_EFFICIENCY
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"user_conversion_metrics": user_data_context},
            confidence_score=confidence,
            recommended_actions=[],  # 由 _generate_actionable_recommendations (特别是 _generate_user_growth_actions) 填充
            expected_impact="提高新用户向活跃/付费用户的转化，提升用户生命周期价值。",
            implementation_difficulty="中等",
            data_sources=[f"{api_name}" for api_name in
                          user_data_context.get("api_source_names", ["/api/sta/user_daily", "/api/sta/user"])],
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="未来1-2个月内优化激活流程"
        )
    async def _generate_generic_actions(self, insight: BusinessInsight,
                                        processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        (私有) 为通用类型的洞察生成普适性的行动建议。
        """
        actions: List[Dict[str, Any]] = []
        action_prefix = f"generic_act_{insight.insight_id[-6:]}"

        logger.info(f"Generating generic actions for insight: {insight.title} (Type: {insight.insight_type.value})")

        if not self.claude_client:
            logger.warning("ClaudeClient not available for generating generic actions.")
            actions.append({
                'action_id': f"{action_prefix}_follow_up", 'title': "跟进此项洞察",
                'description': f"针对洞察 '{insight.title}' 进行进一步分析，并根据具体情况制定后续计划。",
                'action_type': ActionType.SHORT_TERM_PLAN.value, 'priority': insight.priority.value
            })
            return actions

        prompt_context = {
            "insight_title": insight.title,
            "insight_summary": insight.summary,
            "insight_type": insight.insight_type.value,
            "insight_priority": insight.priority.value,
            "key_insight_metrics": insight.key_metrics,
            "detailed_insight_analysis": insight.detailed_analysis
        }

        prompt = f"""
        您是一位经验丰富的业务策略顾问。以下是一项业务洞察：
        洞察标题: "{insight.title}"
        洞察类型: {insight.insight_type.value}
        洞察摘要: "{insight.summary}"
        优先级: {insight.priority.value}
        支持该洞察的关键指标: {json.dumps(insight.key_metrics, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}
        系统对该洞察的详细分析内容:
        ---
        {insight.detailed_analysis}
        ---

        基于以上信息，请为这项洞察提出 1-2 条最相关的、具有普遍适用性的行动建议。
        对于每条建议，请提供：
        1.  `action_title`: 建议的简洁标题。
        2.  `action_description`: 建议的详细描述。
        3.  `action_type`: （例如："monitoring", "short_term_plan", "medium_term_plan"）。
        4.  `general_expected_outcome`: 执行此建议通常期望达到什么类型的结果。

        请以JSON数组的格式返回这些建议。
        """
        try:
            ai_response = await self.claude_client.analyze_complex_query(query=prompt, context=prompt_context)
            if ai_response and ai_response.get('success'):
                response_content = ai_response.get('analysis', ai_response.get('response'))
                if isinstance(response_content, str):
                    try:
                        response_content = json.loads(response_content)
                    except json.JSONDecodeError:
                        logger.error(f"AI response for generic actions is not valid JSON: {response_content[:200]}")

                if isinstance(response_content, list):
                    for idx, action_data in enumerate(response_content):
                        if isinstance(action_data, dict):
                            actions.append({
                                'action_id': f"{action_prefix}_{idx}",
                                'action_type': ActionType(
                                    action_data.get('action_type', ActionType.MONITORING.value)).value,
                                'title': action_data.get('action_title', f"通用建议 {idx + 1}"),
                                'description': action_data.get('action_description', "根据AI分析的通用建议。"),
                                'priority': insight.priority.value,  # 继承洞察优先级
                                'timeline': "根据业务节奏安排",
                                'responsible_party': "相关业务团队",
                                'required_resources': ["业务分析", "团队讨论"],
                                'success_metrics': [action_data.get('general_expected_outcome', "相关指标改善")],
                                'expected_outcome': action_data.get('general_expected_outcome',
                                                                    "业务流程或指标得到优化"),
                                'potential_risks': ["执行不到位可能影响效果"]
                            })
            else:
                logger.warning(
                    f"AI call for generic actions failed or no success: {ai_response.get('error') if ai_response else 'N/A'}")
        except Exception as e:
            logger.error(f"Error generating generic actions with AI: {e}")

        if not actions:  # 如果AI调用失败或未返回有效内容
            actions.append({
                'action_id': f"{action_prefix}_default_discuss", 'title': "讨论并制定行动计划",
                'description': f"针对洞察 '{insight.title}' ({insight.summary[:50]}...)，组织相关团队进行讨论，评估其影响并制定具体的后续行动计划。",
                'action_type': ActionType.SHORT_TERM_PLAN.value, 'priority': insight.priority.value,
                'timeline': "1周内", 'responsible_party': "相关负责人",
                'expected_outcome': "明确此洞察的应对策略"
            })
        return actions
    def _extract_key_metrics_for_insight(self, analysis_result_item: Any, context_description: str = "") -> Dict[
        str, Any]:
        """
        (私有辅助) 从单个分析结果项中提取关键指标，用于特定洞察。
        这个方法与 Orchestrator 中的 _extract_key_metrics 功能类似但作用域更小。
        """
        metrics: Dict[str, Any] = {}
        if not analysis_result_item:
            return metrics

        data_to_search: Optional[Dict[str, Any]] = None
        if hasattr(analysis_result_item, 'to_dict') and callable(analysis_result_item.to_dict):
            data_to_search = analysis_result_item.to_dict()
        elif hasattr(analysis_result_item, '__dict__'):
            data_to_search = analysis_result_item.__dict__
        elif isinstance(analysis_result_item, dict):
            data_to_search = analysis_result_item

        if not data_to_search:
            return metrics

        # 尝试从常见的指标容器键中提取
        metric_container_keys = ['key_metrics', 'metrics', 'main_prediction', 'data']
        for container_key in metric_container_keys:
            if container_key in data_to_search and isinstance(data_to_search[container_key], dict):
                for k, v in data_to_search[container_key].items():
                    # 简单提取，不加复杂前缀，因为这是针对单个洞察的上下文
                    # 可以根据 context_description 进一步筛选相关指标
                    if isinstance(v, (str, int, float, bool)):  # 只取简单类型作为指标值
                        metrics[k] = v
                if metrics: break  # 如果在一个容器中找到，可能就够了

        logger.debug(f"Extracted {len(metrics)} key metrics for insight context '{context_description}'.")
        return metrics

    async def _create_data_anomaly_risk_insight(self, anomalies: List[Dict[str, Any]]) -> BusinessInsight:
        """
        (私有) 创建数据异常风险洞察。
        anomalies: 通常来自 FinancialDataAnalyzer.detect_anomalies 的结果列表。
                  每个 anomaly 字典应包含 'metric', 'date', 'actual_value', 'expected_value', 'severity' 等。
        """
        insight_id = f"data_anomaly_risk_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        title = "数据异常波动风险提示"

        # 总结异常情况
        num_anomalies = len(anomalies)
        critical_anomalies_count = sum(
            1 for anom in anomalies if anom.get('severity', '').lower() in ['high', 'critical'])
        affected_metrics = list(set(anom.get('metric', '未知指标') for anom in anomalies))

        summary = f"检测到 {num_anomalies} 个数据点存在异常波动，其中 {critical_anomalies_count} 个为高风险或严重异常。主要影响指标包括：{', '.join(affected_metrics[:3])}。"
        detailed_analysis = f"{summary} 需要进一步调查这些异常数据点产生的原因，评估其对业务决策的潜在影响。"
        key_metrics = {
            '检测到的异常总数': num_anomalies,
            '高风险/严重异常数': critical_anomalies_count,
            '受影响指标示例': affected_metrics[:3]
        }
        confidence = 0.80 + (0.10 if critical_anomalies_count > 0 else 0)  # 如果有严重异常，置信度更高
        priority = InsightPriority.HIGH if critical_anomalies_count > 0 else InsightPriority.MEDIUM

        if self.claude_client and anomalies:
            # 选取前几个最严重的异常进行分析
            anomalies_for_ai = sorted(anomalies, key=lambda x: (
            x.get('severity', 'low') == 'critical', x.get('severity', 'low') == 'high',
            -float(x.get('deviation_score', 0))), reverse=True)[:3]

            prompt = f"""
            金融数据分析系统检测到以下数据异常点：
            {json.dumps(anomalies_for_ai, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)}

            请用中文分析：
            1. 这些异常数据可能共同指向的潜在业务问题是什么？（例如：数据采集错误、欺诈行为、市场突变、系统故障等）
            2. 基于这些异常，最需要关注的风险点是什么？
            3. 为了确认异常原因并评估影响，建议立即采取哪些调查步骤？
            4. 简要说明这些异常如果属实，可能对业务决策带来哪些误导。

            请提供一个包含以下键的JSON对象：
            "detailed_analysis_text": "对异常的综合分析和潜在原因推测。",
            "primary_risks_identified": ["主要风险点1", "风险点2"],
            "investigation_steps_suggested": ["调查步骤1", "步骤2"]
            """
            try:
                ai_response = await self.claude_client.analyze_complex_query(
                    query=prompt,
                    context={"detected_anomalies_sample": anomalies_for_ai}
                )
                if ai_response and ai_response.get('success'):
                    analysis_content = ai_response.get('analysis', {})
                    if isinstance(analysis_content, str):
                        try:
                            analysis_content = json.loads(analysis_content)
                        except:
                            pass

                    if isinstance(analysis_content, dict):
                        detailed_analysis = analysis_content.get('detailed_analysis_text', detailed_analysis)
                        # Recommended actions can be derived from investigation_steps_suggested
                        confidence = analysis_content.get('confidence', confidence)
                else:
                    logger.warning(
                        f"AI call for data anomaly risk insight failed: {ai_response.get('error') if ai_response else 'N/A'}")
            except Exception as e:
                logger.error(f"Error during AI call for data anomaly risk insight: {e}")

        return BusinessInsight(
            insight_id=insight_id,
            insight_type=InsightType.RISK_WARNING,
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=detailed_analysis,
            key_metrics=key_metrics,
            supporting_data={"anomalies_detected_sample": anomalies[:5]},  # 只取前5个异常作为支持数据
            confidence_score=confidence,
            recommended_actions=[],
            # 由 _generate_actionable_recommendations (特别是 _generate_risk_mitigation_actions) 填充
            expected_impact="及时发现和处理数据异常，可避免基于错误数据的决策，降低运营风险。",
            implementation_difficulty="中等",
            data_sources=["Varies (based on anomaly source)"],  # 需要从anomalies对象中提取
            analysis_timestamp=datetime.now().isoformat(),
            applicable_timeframe="需立即关注和调查"
        )



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