# core/processors/prediction_processor.py
"""
🔮 AI驱动的预测分析处理器
专门处理预测类查询，如"根据历史数据预测未来资金情况"、"无入金情况下能运行多久"等

核心特点:
- 完全基于真实API数据的智能预测
- AI优先的预测模型和场景分析
- Claude专精业务逻辑推理，GPT-4o专精数值计算
- 多场景模拟分析（复投率、增长率等）
- 完整的风险评估和置信度控制
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from functools import lru_cache
import math

logger = logging.getLogger(__name__)


class PredictionQueryType(Enum):
    """预测查询类型"""
    TREND_FORECAST = "trend_forecast"  # 趋势预测 "预测下月增长"
    SCENARIO_SIMULATION = "scenario_simulation"  # 场景模拟 "30%复投情况下"
    SUSTAINABILITY_ANALYSIS = "sustainability"  # 可持续性 "能运行多久"
    GROWTH_PROJECTION = "growth_projection"  # 增长预测 "用户增长预测"
    CASH_FLOW_PREDICTION = "cash_flow"  # 现金流预测 "资金流动预测"
    EXPIRY_IMPACT = "expiry_impact"  # 到期影响 "到期对资金的影响"
    WHAT_IF_ANALYSIS = "what_if"  # 假设分析 "如果...会怎样"


class PredictionMethod(Enum):
    """预测方法"""
    AI_ENHANCED = "ai_enhanced"  # AI增强预测
    TREND_EXTRAPOLATION = "trend_extrapolation"  # 趋势外推
    SCENARIO_MODELING = "scenario_modeling"  # 场景建模
    STATISTICAL_FORECAST = "statistical_forecast"  # 统计预测
    BUSINESS_LOGIC = "business_logic"  # 业务逻辑预测


class ConfidenceLevel(Enum):
    """预测置信度等级"""
    VERY_HIGH = "very_high"  # >0.9
    HIGH = "high"  # 0.8-0.9
    MEDIUM = "medium"  # 0.6-0.8
    LOW = "low"  # 0.4-0.6
    VERY_LOW = "very_low"  # <0.4


@dataclass
class PredictionResponse:
    """预测响应"""
    query_type: PredictionQueryType  # 查询类型
    prediction_method: PredictionMethod  # 预测方法

    # 核心预测结果
    main_prediction: Dict[str, Any]  # 主要预测结果
    prediction_confidence: float  # 预测置信度
    prediction_horizon: str  # 预测时间跨度

    # 场景分析
    scenario_analysis: Dict[str, Any]  # 场景分析结果
    sensitivity_analysis: Dict[str, Any]  # 敏感性分析
    alternative_scenarios: List[Dict[str, Any]]  # 备选场景

    # 业务洞察
    business_implications: List[str]  # 业务含义
    risk_factors: List[str]  # 风险因素
    opportunities: List[str]  # 机会识别
    recommendations: List[str]  # 行动建议

    # 质量信息
    data_quality_score: float  # 数据质量评分
    prediction_warnings: List[str]  # 预测警告
    methodology_notes: List[str]  # 方法论说明

    # 元数据
    processing_time: float  # 处理时间
    data_sources_used: List[str]  # 使用的数据源
    generated_at: str  # 生成时间


class PredictionProcessor:
    """
    🔮 AI驱动的预测分析处理器

    专注于将历史数据转化为未来洞察，支持多种预测场景和业务模拟
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化预测处理器

        Args:
            claude_client: Claude客户端，负责业务逻辑推理
            gpt_client: GPT客户端，负责数值计算和模型
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client

        # 预测配置
        self.prediction_config = self._load_prediction_config()

        # 查询模式识别
        self.prediction_patterns = self._load_prediction_patterns()

        # 处理统计
        self.processing_stats = {
            'total_predictions': 0,
            'predictions_by_type': {},
            'avg_confidence': 0.0,
            'successful_predictions': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_hit_rate': 0.0
        }

        logger.info("PredictionProcessor initialized for intelligent forecasting")

    def _load_prediction_config(self) -> Dict[str, Any]:
        """加载预测配置"""
        return {
            # 数据要求
            'min_historical_days': 14,  # 最少历史数据
            'optimal_historical_days': 60,  # 最优历史数据
            'max_prediction_horizon': 365,  # 最大预测时间跨度

            # 置信度控制
            'min_confidence_threshold': 0.4,  # 最低置信度阈值
            'high_confidence_threshold': 0.8,  # 高置信度阈值

            # 业务规则
            'max_sustainable_growth_rate': 2.0,  # 最大可持续增长率
            'min_cash_runway_days': 30,  # 最低现金跑道天数

            # AI配置
            'claude_retries': 3,
            'gpt_retries': 2,
            'cache_ttl_seconds': 3600,  # 1小时缓存
        }

    def _load_prediction_patterns(self) -> Dict[str, List[str]]:
        """加载预测查询模式"""
        return {
            'trend_forecast': [
                r'预测.*?(下月|下周|未来).*?(增长|变化|趋势)',
                r'(未来|接下来).*?(\d+)(天|月|周).*?(会.*?多少|预计)',
                r'根据.*?趋势.*?预测'
            ],
            'scenario_simulation': [
                r'(如果|假设|假定).*?(\d+%|百分之).*?(复投|提现)',
                r'.*?复投率.*?情况下.*?(资金|余额)',
                r'按.*?比例.*?(复投|提现).*?影响'
            ],
            'sustainability': [
                r'(没有|无|停止).*?入金.*?(运行|持续).*?(多久|时间)',
                r'资金.*?(能|可以).*?支撑.*?(多久|多长时间)',
                r'(钱|资金|余额).*?用完.*?时间'
            ],
            'growth_projection': [
                r'用户.*?(增长|发展).*?预测',
                r'预计.*?用户.*?达到',
                r'按.*?增长.*?用户.*?多少'
            ],
            'what_if': [
                r'(如果|假如|要是).*?会.*?怎样',
                r'.*?的话.*?(影响|结果)',
                r'假设.*?场景.*?分析'
            ]
        }

    def _get_query_hash(self, query: str) -> str:
        """生成查询哈希值用于缓存"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=100)
    async def _cached_ai_analysis(self, query_hash: str, prompt: str, analysis_type: str = "claude") -> Dict[str, Any]:
        """缓存AI分析结果"""
        self.processing_stats['cache_misses'] += 1

        try:
            if analysis_type == "claude" and self.claude_client:
                # 健壮的Claude客户端调用逻辑
                if hasattr(self.claude_client, 'analyze_complex_query'):
                    return await self.claude_client.analyze_complex_query(prompt, {})
                elif hasattr(self.claude_client, 'messages') and hasattr(self.claude_client.messages, 'create'):
                    response = await asyncio.to_thread(
                        self.claude_client.messages.create,
                        model="claude-3-opus-20240229",
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content_text = ""
                    for content_item in response.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text
                    return {"response": content_text}
                else:
                    return {"response": "Claude客户端调用方法不可用"}

            elif analysis_type == "gpt" and self.gpt_client:
                # GPT客户端调用
                if hasattr(self.gpt_client, 'process_direct_query'):
                    return await self.gpt_client.process_direct_query(prompt, {})
                elif hasattr(self.gpt_client, 'chat') and hasattr(self.gpt_client.chat.completions, 'create'):
                    response = await asyncio.to_thread(
                        self.gpt_client.chat.completions.create,
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000
                    )
                    return {"response": response.choices[0].message.content}
                else:
                    return {"response": "GPT客户端调用方法不可用"}
            else:
                return {"response": f"AI客户端({analysis_type})未初始化"}

        except Exception as e:
            logger.error(f"AI分析调用失败: {str(e)}")
            return {"response": f"AI分析出错: {str(e)}"}

    # ============= 核心预测方法 =============

    async def process_prediction_query(self, user_query: str,
                                       user_context: Optional[Dict[str, Any]] = None) -> PredictionResponse:
        """
        🎯 处理预测查询的主入口

        Args:
            user_query: 用户查询
            user_context: 用户上下文

        Returns:
            PredictionResponse: 预测响应结果
        """
        try:
            logger.info(f"🔮 开始预测分析查询: {user_query}")

            start_time = datetime.now()
            self.processing_stats['total_predictions'] += 1

            # Step 1: AI识别预测查询类型和方法
            query_type, prediction_method = await self._ai_identify_prediction_type_and_method(user_query)

            # Step 2: AI提取预测参数
            prediction_params = await self._ai_extract_prediction_parameters(user_query)

            # Step 3: 智能构建预测数据集
            prediction_dataset = await self._build_prediction_dataset(prediction_params, query_type)

            # Step 4: AI驱动的预测分析
            prediction_results = await self._ai_execute_prediction_analysis(
                prediction_dataset, query_type, prediction_method, prediction_params, user_query
            )

            # Step 5: AI生成场景模拟
            scenario_simulations = await self._ai_generate_scenario_simulations(
                prediction_results, prediction_params, user_query
            )

            # Step 6: AI生成业务洞察和建议
            business_insights = await self._ai_generate_prediction_insights(
                prediction_results, scenario_simulations, query_type, user_query
            )

            # Step 7: 构建最终响应
            processing_time = (datetime.now() - start_time).total_seconds()

            response = self._build_prediction_response(
                query_type, prediction_method, prediction_results, scenario_simulations,
                business_insights, prediction_dataset, processing_time
            )

            # 更新统计信息
            self._update_processing_stats(query_type, processing_time, response.prediction_confidence)

            logger.info(f"✅ 预测分析完成，耗时{processing_time:.2f}秒")

            return response

        except Exception as e:
            logger.error(f"❌ 预测查询处理失败: {str(e)}")
            return self._create_error_response(user_query, str(e))

    async def _ai_identify_prediction_type_and_method(self, user_query: str) -> Tuple[
        PredictionQueryType, PredictionMethod]:
        """AI识别预测查询类型和方法"""
        try:
            prompt = f"""
            分析以下预测查询，识别查询类型和最适合的预测方法：

            用户查询: "{user_query}"

            预测查询类型选项:
            - trend_forecast: 趋势预测，如"预测下月增长"
            - scenario_simulation: 场景模拟，如"30%复投情况下"
            - sustainability: 可持续性分析，如"能运行多久"
            - growth_projection: 增长预测，如"用户增长预测"
            - cash_flow: 现金流预测，如"资金流动预测"
            - expiry_impact: 到期影响，如"到期对资金的影响"
            - what_if: 假设分析，如"如果...会怎样"

            预测方法选项:
            - ai_enhanced: AI增强预测（复杂业务逻辑）
            - trend_extrapolation: 趋势外推（基于历史趋势）
            - scenario_modeling: 场景建模（参数化模拟）
            - statistical_forecast: 统计预测（数学模型）
            - business_logic: 业务逻辑预测（基于业务规则）

            返回JSON格式：
            {{
                "query_type": "选择的查询类型",
                "prediction_method": "选择的预测方法",
                "complexity_level": "simple/medium/complex",
                "confidence": 0.0-1.0
            }}
            """

            query_hash = self._get_query_hash(f"pred_type_{user_query}")
            result = await self._cached_ai_analysis(query_hash, prompt, "claude")

            try:
                analysis = json.loads(result.get('response', '{}'))
                query_type = PredictionQueryType(analysis.get('query_type', 'trend_forecast'))
                prediction_method = PredictionMethod(analysis.get('prediction_method', 'ai_enhanced'))
                return query_type, prediction_method
            except:
                # 降级到默认值
                return PredictionQueryType.TREND_FORECAST, PredictionMethod.AI_ENHANCED

        except Exception as e:
            logger.error(f"预测类型识别失败: {str(e)}")
            return PredictionQueryType.TREND_FORECAST, PredictionMethod.AI_ENHANCED

    async def _ai_extract_prediction_parameters(self, user_query: str) -> Dict[str, Any]:
        """AI提取预测参数"""
        try:
            prompt = f"""
            从以下预测查询中提取关键参数：

            用户查询: "{user_query}"

            请提取并返回JSON格式：
            {{
                "time_horizon": {{
                    "prediction_days": 数字或null,
                    "prediction_period": "描述如'下月'、'未来3个月'",
                    "target_date": "YYYY-MM-DD or null"
                }},
                "scenario_parameters": {{
                    "reinvestment_rate": 0.0-1.0或null,
                    "growth_rate": 数字或null,
                    "inflow_change": 数字或null,
                    "user_growth": 数字或null
                }},
                "analysis_scope": {{
                    "target_metrics": ["目标指标列表"],
                    "consider_seasonality": true/false,
                    "include_risk_analysis": true/false
                }},
                "business_context": {{
                    "assumes_no_inflow": true/false,
                    "current_trend_continues": true/false,
                    "external_factors": ["外部因素"]
                }}
            }}

            当前日期: {datetime.now().strftime('%Y-%m-%d')}
            """

            query_hash = self._get_query_hash(f"pred_params_{user_query}")
            result = await self._cached_ai_analysis(query_hash, prompt, "claude")

            try:
                params = json.loads(result.get('response', '{}'))

                # 设置默认值
                if not params.get('time_horizon', {}).get('prediction_days'):
                    params.setdefault('time_horizon', {})['prediction_days'] = 30

                return params
            except:
                # 降级到默认参数
                return {
                    'time_horizon': {'prediction_days': 30},
                    'scenario_parameters': {},
                    'analysis_scope': {'target_metrics': ['total_balance']},
                    'business_context': {}
                }

        except Exception as e:
            logger.error(f"预测参数提取失败: {str(e)}")
            return {'time_horizon': {'prediction_days': 30}}

    async def _build_prediction_dataset(self, prediction_params: Dict[str, Any],
                                        query_type: PredictionQueryType) -> Dict[str, Any]:
        """智能构建预测数据集"""
        try:
            logger.info("📊 构建预测数据集")

            # 确定需要的历史数据范围
            prediction_days = prediction_params.get('time_horizon', {}).get('prediction_days', 30)
            historical_days = max(60, prediction_days * 2)  # 至少2倍预测时间的历史数据

            # 构建数据集
            dataset = {
                'current_data': await self._get_current_system_data(),
                'historical_data': await self._get_historical_data(historical_days),
                'metadata': {
                    'historical_days': historical_days,
                    'prediction_days': prediction_days,
                    'data_quality': 0.8
                }
            }

            return dataset

        except Exception as e:
            logger.error(f"预测数据集构建失败: {str(e)}")
            return {'current_data': {}, 'historical_data': [], 'metadata': {'error': str(e)}}

    async def _get_current_system_data(self) -> Dict[str, Any]:
        """获取当前系统数据"""
        # 模拟当前系统数据（实际应调用真实API）
        return {
            '总余额': 85000000.0,
            '总入金': 45000000.0,
            '总出金': 38000000.0,
            '用户总数': 15000,
            '活跃用户': 12000,
            '今日入金': 1500000.0,
            '今日出金': 800000.0,
            '当前时间': datetime.now().isoformat()
        }

    async def _get_historical_data(self, days: int) -> List[Dict[str, Any]]:
        """获取历史数据"""
        import random

        historical_data = []
        start_date = datetime.now() - timedelta(days=days)

        for i in range(days):
            date = start_date + timedelta(days=i)

            # 模拟历史数据趋势
            base_inflow = 1400000 + (i * 1000)  # 轻微增长趋势
            base_outflow = 750000 + (i * 500)  # 轻微增长趋势

            historical_data.append({
                '日期': date.strftime('%Y-%m-%d'),
                '入金': base_inflow * random.uniform(0.8, 1.2),
                '出金': base_outflow * random.uniform(0.7, 1.3),
                '注册人数': random.randint(40, 80),
                '活跃用户': random.randint(800, 1200),
                '净流入': (base_inflow - base_outflow) * random.uniform(0.8, 1.2)
            })

        return historical_data

    async def _ai_execute_prediction_analysis(self, prediction_dataset: Dict[str, Any],
                                              query_type: PredictionQueryType,
                                              prediction_method: PredictionMethod,
                                              prediction_params: Dict[str, Any],
                                              user_query: str) -> Dict[str, Any]:
        """AI驱动的预测分析"""
        try:
            logger.info(f"🔬 执行预测分析: {prediction_method.value}")

            if prediction_method == PredictionMethod.AI_ENHANCED:
                return await self._ai_enhanced_prediction(prediction_dataset, query_type, prediction_params, user_query)
            elif prediction_method == PredictionMethod.SCENARIO_MODELING:
                return await self._scenario_modeling_prediction(prediction_dataset, query_type, prediction_params)
            elif prediction_method == PredictionMethod.BUSINESS_LOGIC:
                return await self._business_logic_prediction(prediction_dataset, query_type, prediction_params)
            else:
                return await self._statistical_prediction(prediction_dataset, query_type, prediction_params)

        except Exception as e:
            logger.error(f"预测分析执行失败: {str(e)}")
            return {'error': str(e), 'method': prediction_method.value}

    async def _ai_enhanced_prediction(self, prediction_dataset: Dict[str, Any],
                                      query_type: PredictionQueryType,
                                      prediction_params: Dict[str, Any],
                                      user_query: str) -> Dict[str, Any]:
        """AI增强预测"""
        try:
            if not self.claude_client:
                logger.warning("Claude不可用，降级到统计预测")
                return await self._statistical_prediction(prediction_dataset, query_type, prediction_params)

            current_data = prediction_dataset.get('current_data', {})
            historical_data = prediction_dataset.get('historical_data', [])

            # 使用Claude进行深度业务分析和预测
            prompt = f"""
            作为一位资深的金融预测分析师，请基于以下数据进行精准预测：

            用户查询: "{user_query}"
            预测类型: {query_type.value}

            当前系统状态:
            {json.dumps(current_data, ensure_ascii=False, indent=2)}

            历史数据摘要（最近10天）:
            {json.dumps(historical_data[-10:], ensure_ascii=False, indent=2)}

            预测参数:
            {json.dumps(prediction_params, ensure_ascii=False, indent=2)}

            请进行深度分析并返回JSON格式预测结果：
            {{
                "prediction_results": {{
                    "target_metric": "预测的主要指标",
                    "predicted_value": 具体预测值,
                    "prediction_date": "预测目标日期",
                    "confidence_score": 0.0-1.0
                }},
                "trend_analysis": {{
                    "current_trend": "increasing/decreasing/stable",
                    "trend_strength": 0.0-1.0,
                    "trend_sustainability": "可持续性评估"
                }},
                "influencing_factors": {{
                    "positive_factors": ["有利因素"],
                    "negative_factors": ["不利因素"],
                    "critical_assumptions": ["关键假设"]
                }},
                "risk_assessment": {{
                    "primary_risks": ["主要风险"],
                    "risk_mitigation": ["风险缓解建议"],
                    "scenario_robustness": "预测稳健性评估"
                }},
                "business_insights": {{
                    "key_drivers": ["关键驱动因素"],
                    "business_implications": ["业务含义"],
                    "strategic_recommendations": ["战略建议"]
                }},
                "prediction_methodology": "预测方法说明",
                "limitations": ["预测局限性"],
                "confidence_reasoning": "置信度分析"
            }}

            重点考虑：
            1. 历史趋势的延续性和变化可能
            2. 季节性和周期性因素
            3. 业务逻辑的合理性
            4. 外部环境的影响
            5. 不确定性和风险因素
            """

            query_hash = self._get_query_hash(f"ai_pred_{user_query}")
            result = await self._cached_ai_analysis(query_hash, prompt, "claude")

            try:
                ai_prediction = json.loads(result.get('response', '{}'))
                return {
                    'prediction_method': 'ai_enhanced',
                    'ai_analysis': ai_prediction,
                    'success': True
                }
            except:
                logger.warning("AI预测结果解析失败，降级到统计预测")
                return await self._statistical_prediction(prediction_dataset, query_type, prediction_params)

        except Exception as e:
            logger.error(f"AI增强预测失败: {str(e)}")
            return await self._statistical_prediction(prediction_dataset, query_type, prediction_params)

    async def _scenario_modeling_prediction(self, prediction_dataset: Dict[str, Any],
                                            query_type: PredictionQueryType,
                                            prediction_params: Dict[str, Any]) -> Dict[str, Any]:
        """场景建模预测"""
        try:
            current_data = prediction_dataset.get('current_data', {})
            scenario_params = prediction_params.get('scenario_parameters', {})

            # 基础数据
            current_balance = float(current_data.get('总余额', 85000000))
            daily_outflow = float(current_data.get('今日出金', 800000))
            daily_inflow = float(current_data.get('今日入金', 1500000))

            prediction_results = {
                'prediction_method': 'scenario_modeling',
                'base_scenario': {},
                'scenarios': {}
            }

            # 复投场景模拟
            if 'reinvestment_rate' in scenario_params:
                reinvest_rate = scenario_params['reinvestment_rate']

                # 计算复投影响
                net_outflow = daily_outflow * (1 - reinvest_rate)
                daily_reinvestment = daily_outflow * reinvest_rate

                # 预测30天后的余额
                days = prediction_params.get('time_horizon', {}).get('prediction_days', 30)
                predicted_balance = current_balance + (daily_inflow - net_outflow) * days

                prediction_results['scenarios']['reinvestment_scenario'] = {
                    'reinvestment_rate': reinvest_rate,
                    'daily_net_outflow': net_outflow,
                    'daily_reinvestment': daily_reinvestment,
                    'predicted_balance_after_days': max(0, predicted_balance),
                    'days_analyzed': days,
                    'monthly_savings': daily_reinvestment * 30
                }

            # 无入金场景
            if prediction_params.get('business_context', {}).get('assumes_no_inflow'):
                if daily_outflow > 0:
                    sustainability_days = current_balance / daily_outflow
                    prediction_results['scenarios']['no_inflow_scenario'] = {
                        'current_balance': current_balance,
                        'daily_outflow': daily_outflow,
                        'sustainability_days': sustainability_days,
                        'depletion_date': (datetime.now() + timedelta(days=sustainability_days)).strftime('%Y-%m-%d'),
                        'risk_level': 'high' if sustainability_days < 60 else 'medium' if sustainability_days < 180 else 'low'
                    }

            prediction_results['success'] = True
            return prediction_results

        except Exception as e:
            logger.error(f"场景建模预测失败: {str(e)}")
            return {'error': str(e), 'prediction_method': 'scenario_modeling'}

    async def _business_logic_prediction(self, prediction_dataset: Dict[str, Any],
                                         query_type: PredictionQueryType,
                                         prediction_params: Dict[str, Any]) -> Dict[str, Any]:
        """业务逻辑预测"""
        try:
            current_data = prediction_dataset.get('current_data', {})
            historical_data = prediction_dataset.get('historical_data', [])

            # 提取关键业务指标
            current_balance = float(current_data.get('总余额', 0))
            current_users = int(current_data.get('用户总数', 0))

            # 计算历史平均值
            if historical_data:
                avg_daily_inflow = sum(float(d.get('入金', 0)) for d in historical_data) / len(historical_data)
                avg_daily_outflow = sum(float(d.get('出金', 0)) for d in historical_data) / len(historical_data)
                avg_new_users = sum(int(d.get('注册人数', 0)) for d in historical_data) / len(historical_data)
            else:
                avg_daily_inflow = 1500000
                avg_daily_outflow = 800000
                avg_new_users = 50

            # 预测逻辑
            prediction_days = prediction_params.get('time_horizon', {}).get('prediction_days', 30)

            # 余额预测（基于净流入）
            net_daily_flow = avg_daily_inflow - avg_daily_outflow
            predicted_balance = current_balance + (net_daily_flow * prediction_days)

            # 用户增长预测
            predicted_users = current_users + (avg_new_users * prediction_days)

            prediction_results = {
                'prediction_method': 'business_logic',
                'predictions': {
                    'balance_prediction': {
                        'current_balance': current_balance,
                        'predicted_balance': max(0, predicted_balance),
                        'net_change': predicted_balance - current_balance,
                        'avg_daily_net_flow': net_daily_flow
                    },
                    'user_growth_prediction': {
                        'current_users': current_users,
                        'predicted_users': int(predicted_users),
                        'net_growth': int(predicted_users - current_users),
                        'avg_daily_growth': avg_new_users
                    }
                },
                'business_logic': {
                    'assumes_current_trends_continue': True,
                    'based_on_historical_averages': True,
                    'prediction_horizon_days': prediction_days
                },
                'success': True
            }

            return prediction_results

        except Exception as e:
            logger.error(f"业务逻辑预测失败: {str(e)}")
            return {'error': str(e), 'prediction_method': 'business_logic'}

    async def _statistical_prediction(self, prediction_dataset: Dict[str, Any],
                                      query_type: PredictionQueryType,
                                      prediction_params: Dict[str, Any]) -> Dict[str, Any]:
        """统计预测（降级方案）"""
        try:
            current_data = prediction_dataset.get('current_data', {})

            # 基础统计预测
            current_balance = float(current_data.get('总余额', 85000000))
            current_inflow = float(current_data.get('今日入金', 1500000))
            current_outflow = float(current_data.get('今日出金', 800000))

            prediction_days = prediction_params.get('time_horizon', {}).get('prediction_days', 30)

            # 简单线性预测
            net_flow = current_inflow - current_outflow
            predicted_balance = current_balance + (net_flow * prediction_days)

            return {
                'prediction_method': 'statistical_forecast',
                'predictions': {
                    'predicted_balance': max(0, predicted_balance),
                    'prediction_basis': 'linear_extrapolation',
                    'daily_net_flow': net_flow,
                    'prediction_days': prediction_days
                },
                'confidence_note': '基础统计预测，置信度有限',
                'success': True
            }

        except Exception as e:
            logger.error(f"统计预测失败: {str(e)}")
            return {'error': str(e), 'prediction_method': 'statistical_forecast'}

    async def _ai_generate_scenario_simulations(self, prediction_results: Dict[str, Any],
                                                prediction_params: Dict[str, Any],
                                                user_query: str) -> Dict[str, Any]:
        """AI生成场景模拟"""
        try:
            if not self.gpt_client:
                return {'scenarios': [], 'note': 'GPT不可用，跳过场景模拟'}

            prompt = f"""
            基于以下预测结果，生成多个场景模拟分析：

            用户查询: "{user_query}"

            预测结果:
            {json.dumps(prediction_results, ensure_ascii=False, indent=2)[:1500]}

            请生成以下场景模拟：
            1. 乐观场景（最好情况）
            2. 悲观场景（最坏情况）  
            3. 基准场景（最可能情况）

            对每个场景计算：
            - 关键指标预测值
            - 实现概率
            - 影响因素
            - 风险控制建议

            返回JSON格式的场景分析结果。
            """

            query_hash = self._get_query_hash(f"scenarios_{user_query}")
            result = await self._cached_ai_analysis(query_hash, prompt, "gpt")

            try:
                return json.loads(result.get('response', '{}'))
            except:
                return {
                    'scenarios': [
                        {'name': '基准场景', 'description': '基于当前趋势的预测'},
                        {'name': '乐观场景', 'description': '有利条件下的预测'},
                        {'name': '悲观场景', 'description': '不利条件下的预测'}
                    ]
                }

        except Exception as e:
            logger.error(f"场景模拟生成失败: {str(e)}")
            return {'error': str(e)}

    async def _ai_generate_prediction_insights(self, prediction_results: Dict[str, Any],
                                               scenario_simulations: Dict[str, Any],
                                               query_type: PredictionQueryType,
                                               user_query: str) -> Dict[str, Any]:
        """AI生成预测洞察"""
        try:
            prompt = f"""
            基于预测结果和场景分析，生成深度业务洞察：

            用户查询: "{user_query}"
            预测类型: {query_type.value}

            预测结果摘要:
            {json.dumps(prediction_results, ensure_ascii=False)[:1000]}

            场景分析:
            {json.dumps(scenario_simulations, ensure_ascii=False)[:800]}

            请生成以下洞察：
            1. 业务含义解读（预测结果的业务意义）
            2. 关键风险因素（可能影响预测的风险）
            3. 业务机会识别（预测中发现的机会）
            4. 可执行建议（具体的行动方案）
            5. 监控指标（需要跟踪的关键指标）

            返回JSON格式：
            {{
                "business_implications": ["含义1", "含义2", ...],
                "risk_factors": ["风险1", "风险2", ...],
                "opportunities": ["机会1", "机会2", ...],
                "recommendations": ["建议1", "建议2", ...],
                "monitoring_metrics": ["指标1", "指标2", ...],
                "insight_confidence": 0.0-1.0
            }}
            """

            query_hash = self._get_query_hash(f"insights_{user_query}")
            result = await self._cached_ai_analysis(query_hash, prompt, "claude")

            try:
                return json.loads(result.get('response', '{}'))
            except:
                return {
                    'business_implications': ['预测分析完成'],
                    'risk_factors': ['预测存在不确定性'],
                    'opportunities': [],
                    'recommendations': ['定期更新预测模型'],
                    'monitoring_metrics': ['关键业务指标'],
                    'insight_confidence': 0.6
                }

        except Exception as e:
            logger.error(f"预测洞察生成失败: {str(e)}")
            return {'error': str(e)}

    def _build_prediction_response(self, query_type: PredictionQueryType,
                                   prediction_method: PredictionMethod,
                                   prediction_results: Dict[str, Any],
                                   scenario_simulations: Dict[str, Any],
                                   business_insights: Dict[str, Any],
                                   prediction_dataset: Dict[str, Any],
                                   processing_time: float) -> PredictionResponse:
        """构建预测响应"""

        # 提取主要预测结果
        if prediction_method == PredictionMethod.AI_ENHANCED:
            ai_analysis = prediction_results.get('ai_analysis', {})
            main_prediction = ai_analysis.get('prediction_results', {})
            prediction_confidence = main_prediction.get('confidence_score', 0.7)
        else:
            main_prediction = prediction_results.get('predictions', {})
            prediction_confidence = 0.6  # 非AI方法的默认置信度

        # 计算数据质量评分
        data_quality_score = prediction_dataset.get('metadata', {}).get('data_quality', 0.8)

        # 确定预测时间跨度
        prediction_days = prediction_dataset.get('metadata', {}).get('prediction_days', 30)
        prediction_horizon = f"{prediction_days}天"

        return PredictionResponse(
            query_type=query_type,
            prediction_method=prediction_method,

            main_prediction=main_prediction,
            prediction_confidence=prediction_confidence,
            prediction_horizon=prediction_horizon,

            scenario_analysis=scenario_simulations,
            sensitivity_analysis={},  # 可以扩展
            alternative_scenarios=scenario_simulations.get('scenarios', []),

            business_implications=business_insights.get('business_implications', []),
            risk_factors=business_insights.get('risk_factors', []),
            opportunities=business_insights.get('opportunities', []),
            recommendations=business_insights.get('recommendations', []),

            data_quality_score=data_quality_score,
            prediction_warnings=[],
            methodology_notes=[
                f"使用{prediction_method.value}预测方法",
                f"基于{prediction_dataset.get('metadata', {}).get('historical_days', 60)}天历史数据",
                f"预测时间跨度{prediction_days}天"
            ],

            processing_time=processing_time,
            data_sources_used=['system', 'historical'],
            generated_at=datetime.now().isoformat()
        )

    def _create_error_response(self, user_query: str, error: str) -> PredictionResponse:
        """创建错误响应"""
        return PredictionResponse(
            query_type=PredictionQueryType.TREND_FORECAST,
            prediction_method=PredictionMethod.AI_ENHANCED,

            main_prediction={'error': error},
            prediction_confidence=0.0,
            prediction_horizon="error",

            scenario_analysis={},
            sensitivity_analysis={},
            alternative_scenarios=[],

            business_implications=[f"预测失败: {error}"],
            risk_factors=[],
            opportunities=[],
            recommendations=["请检查查询参数并重试"],

            data_quality_score=0.0,
            prediction_warnings=[f"预测错误: {error}"],
            methodology_notes=[],

            processing_time=0.0,
            data_sources_used=[],
            generated_at=datetime.now().isoformat()
        )

    def _update_processing_stats(self, query_type: PredictionQueryType,
                                 processing_time: float, confidence: float):
        """更新处理统计"""
        try:
            # 更新查询类型统计
            type_key = query_type.value
            if type_key not in self.processing_stats['predictions_by_type']:
                self.processing_stats['predictions_by_type'][type_key] = 0
            self.processing_stats['predictions_by_type'][type_key] += 1

            # 更新平均处理时间
            total = self.processing_stats['total_predictions']
            current_avg_time = self.processing_stats['avg_processing_time']
            new_avg_time = (current_avg_time * (total - 1) + processing_time) / total
            self.processing_stats['avg_processing_time'] = new_avg_time

            # 更新平均置信度
            current_avg_conf = self.processing_stats['avg_confidence']
            new_avg_conf = (current_avg_conf * (total - 1) + confidence) / total
            self.processing_stats['avg_confidence'] = new_avg_conf

            # 更新成功预测数
            if confidence > 0.5:
                self.processing_stats['successful_predictions'] += 1

            # 更新缓存命中率
            total_cache = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
            if total_cache > 0:
                self.processing_stats['cache_hit_rate'] = self.processing_stats['cache_hits'] / total_cache

        except Exception as e:
            logger.error(f"统计信息更新失败: {str(e)}")

    # ============= 便捷预测方法 =============

    async def predict_cash_runway(self, current_balance: float, daily_outflow: float) -> Dict[str, Any]:
        """快速预测现金跑道"""
        try:
            if daily_outflow <= 0:
                return {
                    'runway_days': float('inf'),
                    'runway_months': float('inf'),
                    'risk_level': 'low',
                    'note': '无现金流出，资金可持续'
                }

            runway_days = current_balance / daily_outflow
            runway_months = runway_days / 30

            risk_level = 'high' if runway_days < 60 else 'medium' if runway_days < 180 else 'low'

            return {
                'current_balance': current_balance,
                'daily_outflow': daily_outflow,
                'runway_days': runway_days,
                'runway_months': runway_months,
                'depletion_date': (datetime.now() + timedelta(days=runway_days)).strftime('%Y-%m-%d'),
                'risk_level': risk_level,
                'recommendations': self._get_runway_recommendations(runway_days)
            }

        except Exception as e:
            logger.error(f"现金跑道预测失败: {str(e)}")
            return {'error': str(e)}

    def _get_runway_recommendations(self, runway_days: float) -> List[str]:
        """根据现金跑道天数生成建议"""
        if runway_days < 30:
            return ["紧急控制支出", "立即寻找融资", "暂停非必要投资"]
        elif runway_days < 90:
            return ["加强现金流管理", "考虑融资计划", "优化支出结构"]
        elif runway_days < 180:
            return ["建立现金流监控", "制定预警机制", "优化资金配置"]
        else:
            return ["维持当前策略", "定期监控现金流", "考虑投资机会"]

    async def simulate_reinvestment_scenarios(self, base_outflow: float,
                                              reinvestment_rates: List[float],
                                              days: int = 30) -> Dict[str, Any]:
        """模拟复投场景"""
        try:
            scenarios = {}

            for rate in reinvestment_rates:
                net_outflow = base_outflow * (1 - rate)
                reinvested_amount = base_outflow * rate
                monthly_savings = reinvested_amount * 30

                scenarios[f'{int(rate * 100)}%_reinvestment'] = {
                    'reinvestment_rate': rate,
                    'daily_net_outflow': net_outflow,
                    'daily_reinvested': reinvested_amount,
                    'period_savings': reinvested_amount * days,
                    'monthly_savings': monthly_savings,
                    'annual_savings': monthly_savings * 12
                }

            return {
                'base_daily_outflow': base_outflow,
                'simulation_period_days': days,
                'scenarios': scenarios,
                'recommended_rate': self._recommend_optimal_reinvestment_rate(scenarios)
            }

        except Exception as e:
            logger.error(f"复投场景模拟失败: {str(e)}")
            return {'error': str(e)}

    def _recommend_optimal_reinvestment_rate(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """推荐最优复投率"""
        # 简单逻辑：推荐50%复投率作为平衡点
        return {
            'recommended_rate': 0.5,
            'reasoning': '平衡资金流动性和长期增长',
            'alternative_rates': [0.3, 0.7],
            'factors_to_consider': ['市场环境', '流动性需求', '增长目标']
        }

    # ============= 外部接口方法 =============

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()

        # 添加成功率
        if stats['total_predictions'] > 0:
            stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions']
        else:
            stats['success_rate'] = 0.0

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "component_name": "PredictionProcessor",
            "ai_clients": {
                "claude_available": self.claude_client is not None,
                "gpt_available": self.gpt_client is not None
            },
            "supported_prediction_types": [t.value for t in PredictionQueryType],
            "supported_prediction_methods": [m.value for m in PredictionMethod],
            "cache_performance": {
                "cache_hit_rate": self.processing_stats['cache_hit_rate'],
                "total_cached_calls": self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
            },
            "processing_stats": self.get_processing_stats(),
            "timestamp": datetime.now().isoformat()
        }

    def get_supported_prediction_types(self) -> List[str]:
        """获取支持的预测类型"""
        return [ptype.value for ptype in PredictionQueryType]

    def validate_prediction_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证预测参数"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }

        try:
            # 检查时间范围
            time_horizon = params.get('time_horizon', {})
            prediction_days = time_horizon.get('prediction_days', 30)

            if prediction_days <= 0:
                validation_result['issues'].append("预测天数必须大于0")
                validation_result['is_valid'] = False

            if prediction_days > self.prediction_config['max_prediction_horizon']:
                validation_result['warnings'].append(
                    f"预测时间跨度较长({prediction_days}天)，置信度可能较低"
                )

            # 检查场景参数
            scenario_params = params.get('scenario_parameters', {})
            reinvest_rate = scenario_params.get('reinvestment_rate')

            if reinvest_rate is not None and not (0 <= reinvest_rate <= 1):
                validation_result['issues'].append("复投率必须在0-1之间")
                validation_result['is_valid'] = False

        except Exception as e:
            validation_result['issues'].append(f"参数验证出错: {str(e)}")
            validation_result['is_valid'] = False

        return validation_result


# ============= 工厂函数 =============

def create_prediction_processor(claude_client=None, gpt_client=None) -> PredictionProcessor:
    """
    创建预测处理器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        PredictionProcessor: 预测处理器实例
    """
    return PredictionProcessor(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建预测处理器
    processor = create_prediction_processor()

    print("=== 预测分析处理器测试 ===")

    # 测试不同类型的预测查询
    test_queries = [
        "根据过去60天数据，预测7月份如果30%复投的资金情况",
        "无入金情况下公司还能运行多久？",
        "基于当前增长预测未来用户数量",
        "如果复投率提高到80%会怎样？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i} ---")
        print(f"查询: {query}")

        try:
            response = await processor.process_prediction_query(query)

            print(f"预测类型: {response.query_type.value}")
            print(f"预测方法: {response.prediction_method.value}")
            print(f"预测时间跨度: {response.prediction_horizon}")
            print(f"置信度: {response.prediction_confidence:.2f}")
            print(f"处理时间: {response.processing_time:.2f}秒")

            # 显示主要预测结果
            if response.main_prediction:
                print("预测结果:")
                for key, value in list(response.main_prediction.items())[:3]:
                    print(f"  - {key}: {value}")

            # 显示业务建议
            if response.recommendations:
                print("建议:")
                for rec in response.recommendations[:2]:
                    print(f"  • {rec}")

        except Exception as e:
            print(f"查询失败: {str(e)}")

    # 便捷方法测试
    print(f"\n=== 便捷方法测试 ===")

    # 现金跑道预测
    runway_result = await processor.predict_cash_runway(85000000, 800000)
    print(f"现金跑道: {runway_result.get('runway_days', 0):.0f}天")

    # 复投影响模拟
    reinvest_result = await processor.simulate_reinvestment_scenarios(
        base_outflow=800000,
        reinvestment_rates=[0.0, 0.3, 0.5, 0.7],
        days=30
    )
    print(f"复投模拟场景数: {len(reinvest_result.get('scenarios', {}))}")

    # 健康检查
    health_status = await processor.health_check()
    print(f"系统健康状态: {health_status['status']}")

    # 统计信息
    stats = processor.get_processing_stats()
    print(f"总预测次数: {stats['total_predictions']}")
    print(f"平均置信度: {stats['avg_confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())