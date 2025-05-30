# core/analyzers/query_parser.py
"""
🧠 AI驱动的智能查询解析器
金融AI分析系统的核心"大脑"，负责理解和分解复杂的金融业务查询

核心特点:
- 双AI协作的查询理解 (Claude + GPT-4o)
- 智能复杂度评估和分级处理
- 动态执行计划生成
- 业务场景自动识别
- 上下文感知的分析策略
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
import re

# 导入我们的工具类
from utils.helpers.date_utils import DateUtils, create_date_utils, DateParseResult
from utils.helpers.validation_utils import ValidationUtils, create_validation_utils

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """查询复杂度等级"""
    SIMPLE = "simple"  # 简单查询 (单一数据获取)
    MEDIUM = "medium"  # 中等复杂 (基础分析计算)
    COMPLEX = "complex"  # 复杂查询 (多步骤分析)
    EXPERT = "expert"  # 专家级 (深度预测分析)


class QueryType(Enum):
    """查询类型分类"""
    DATA_RETRIEVAL = "data_retrieval"  # 数据查询
    TREND_ANALYSIS = "trend_analysis"  # 趋势分析
    PREDICTION = "prediction"  # 预测分析
    COMPARISON = "comparison"  # 对比分析
    CALCULATION = "calculation"  # 计算场景
    RISK_ASSESSMENT = "risk_assessment"  # 风险评估
    SCENARIO_SIMULATION = "scenario_simulation"  # 场景模拟


class BusinessScenario(Enum):
    """业务场景类型"""
    DAILY_OPERATIONS = "daily_operations"  # 日常运营
    FINANCIAL_PLANNING = "financial_planning"  # 财务规划
    RISK_MANAGEMENT = "risk_management"  # 风险管理
    GROWTH_ANALYSIS = "growth_analysis"  # 增长分析
    USER_BEHAVIOR = "user_behavior"  # 用户行为
    PRODUCT_PERFORMANCE = "product_performance"  # 产品表现
    COMPLIANCE_CHECK = "compliance_check"  # 合规检查


@dataclass
class ExecutionStep:
    """执行步骤数据类"""
    step_id: str  # 步骤ID
    step_type: str  # 步骤类型
    description: str  # 步骤描述
    required_data: List[str]  # 需要的数据
    processing_method: str  # 处理方法
    dependencies: List[str]  # 依赖的步骤
    estimated_time: float  # 估计耗时(秒)
    ai_model_preference: str  # 推荐的AI模型


@dataclass
class QueryAnalysisResult:
    """查询分析结果"""
    original_query: str  # 原始查询
    complexity: QueryComplexity  # 复杂度等级
    query_type: QueryType  # 查询类型
    business_scenario: BusinessScenario  # 业务场景
    confidence_score: float  # 分析置信度

    # 时间相关
    time_requirements: Dict[str, Any]  # 时间需求
    date_parse_result: DateParseResult  # 日期解析结果

    # 数据需求
    data_requirements: Dict[str, Any]  # 数据需求
    required_apis: List[str]  # 需要的API

    # 业务参数
    business_parameters: Dict[str, Any]  # 业务参数
    calculation_requirements: Dict[str, Any]  # 计算需求

    # 执行计划
    execution_plan: List[ExecutionStep]  # 执行步骤
    processing_strategy: str  # 处理策略

    # AI协作策略
    ai_collaboration_plan: Dict[str, Any]  # AI协作计划

    # 元数据
    analysis_timestamp: str  # 分析时间戳
    estimated_total_time: float  # 预估总耗时
    processing_metadata: Dict[str, Any]  # 处理元数据


class SmartQueryParser:
    """
    🧠 AI驱动的智能查询解析器

    功能架构:
    1. 双AI协作的查询理解
    2. 智能复杂度评估
    3. 动态执行计划生成
    4. 自适应处理策略
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化智能查询解析器

        Args:
            claude_client: Claude客户端，负责业务逻辑理解
            gpt_client: GPT客户端，负责数据需求分析
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.date_utils = create_date_utils(claude_client)
        self.validator = create_validation_utils(claude_client, gpt_client)

        # 查询模式识别
        self.query_patterns = self._load_query_patterns()

        # 处理统计
        self.processing_stats = {
            'total_queries': 0,
            'complexity_distribution': {
                'simple': 0,
                'medium': 0,
                'complex': 0,
                'expert': 0
            },
            'query_type_distribution': {},
            'ai_collaboration_usage': 0,
            'average_confidence': 0.0
        }

        logger.info("SmartQueryParser initialized with dual-AI capabilities")

    def _load_query_patterns(self) -> Dict[str, Any]:
        """加载查询模式库"""
        return {
            # 简单数据查询模式
            "simple_data_patterns": [
                r"今天|今日|当天.*?(余额|数据|情况)",
                r"(多少|什么).*?(用户|余额|金额)",
                r"显示|给我.*?(系统|概览|状态)",
                r"查看|看看.*?(产品|数据)"
            ],

            # 历史趋势分析模式
            "trend_analysis_patterns": [
                r"(过去|最近).*?(\d+)(天|周|月).*?(趋势|变化|增长)",
                r"对比.*?(上月|上周|去年)",
                r".*?增长.*?(如何|怎么样)",
                r"(平均|每日).*?(增长|变化)"
            ],

            # 预测分析模式
            "prediction_patterns": [
                r"预测|预计|预期.*?(未来|明天|下月|下周)",
                r"(如果|假设).*?(会|将).*?(多少|怎样)",
                r".*?月.*?会.*?(余额|资金)",
                r"基于.*?预测"
            ],

            # 计算场景模式
            "calculation_patterns": [
                r"(复投|提现).*?(\d+%|百分之)",
                r"计算.*?(如果|按照).*?比例",
                r"(\d+%|\d+分之\d+).*?(复投|提现)",
                r"不同.*?率.*?影响"
            ],

            # 风险评估模式
            "risk_assessment_patterns": [
                r"(没有|无).*?入金.*?(运行|持续).*?(多久|时间)",
                r"风险|危险|安全.*?评估",
                r"可持续.*?分析",
                r"资金.*?耗尽"
            ],

            # 场景模拟模式
            "scenario_simulation_patterns": [
                r"(假设|如果|假定).*?情况下",
                r"不同.*?场景.*?对比",
                r"模拟.*?(情况|场景)",
                r".*?情况.*?影响"
            ]
        }

    # ============= 核心查询分析方法 =============

    async def parse_complex_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysisResult:
        """
        🎯 解析复杂查询 - 核心入口方法

        Args:
            query: 用户查询文本
            context: 查询上下文 (用户信息、历史查询等)

        Returns:
            QueryAnalysisResult: 完整的查询分析结果
        """
        try:
            logger.info(f"🧠 开始解析复杂查询: {query}")
            self.processing_stats['total_queries'] += 1

            # 第1步: 基础查询预处理
            preprocessed_query = await self._preprocess_query(query)

            # 第2步: Claude深度理解查询意图和业务逻辑
            claude_analysis = await self._claude_understand_query(preprocessed_query, context)

            # 第3步: GPT分析数据需求和计算要求
            gpt_analysis = await self._gpt_analyze_data_requirements(preprocessed_query, claude_analysis)

            # 第4步: 综合分析，确定复杂度和类型
            complexity_analysis = await self._analyze_query_complexity(
                preprocessed_query, claude_analysis, gpt_analysis
            )

            # 第5步: 解析时间和日期要求
            time_analysis = await self._analyze_time_requirements(preprocessed_query, claude_analysis)

            # 第6步: 生成执行计划
            execution_plan = await self._generate_execution_plan(
                preprocessed_query, claude_analysis, gpt_analysis, complexity_analysis
            )

            # 第7步: 设计AI协作策略
            ai_collaboration_plan = self._design_ai_collaboration(
                complexity_analysis, execution_plan
            )

            # 第8步: 构建最终分析结果
            analysis_result = self._build_analysis_result(
                query, preprocessed_query, claude_analysis, gpt_analysis,
                complexity_analysis, time_analysis, execution_plan, ai_collaboration_plan
            )

            # 更新统计信息
            self._update_processing_stats(analysis_result)

            logger.info(
                f"✅ 查询解析完成: 复杂度={analysis_result.complexity.value}, 类型={analysis_result.query_type.value}")

            return analysis_result

        except Exception as e:
            logger.error(f"❌ 查询解析失败: {str(e)}")
            return self._create_error_analysis_result(query, str(e))

    # ============= Claude业务理解层 =============

    async def _claude_understand_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Claude深度理解查询的业务逻辑和意图"""

        if not self.claude_client:
            logger.warning("Claude客户端不可用，使用基础解析")
            return await self._fallback_query_understanding(query)

        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            understanding_prompt = f"""
你是一位资深的金融业务分析专家。请深度分析以下用户查询的业务意图和逻辑需求。

用户查询: "{query}"
当前日期: {current_date}
查询上下文: {json.dumps(context or {}, ensure_ascii=False)}

请从以下维度进行专业分析：

1. **核心业务意图**: 用户真正想了解什么业务问题？
2. **业务场景分类**: 这属于哪种业务场景？(日常运营/财务规划/风险管理/增长分析/用户行为/产品表现)
3. **分析深度要求**: 需要什么程度的分析？(概览/趋势/预测/深度建模)
4. **业务逻辑复杂度**: 涉及多少层业务逻辑？
5. **时间维度分析**: 涉及什么时间范围和时间概念？
6. **关键业务参数**: 涉及哪些重要的业务参数？(复投率、增长率、风险系数等)
7. **预期输出**: 用户期望得到什么样的答案格式？

请返回JSON格式的分析结果：
{{
    "business_intent": {{
        "primary_goal": "主要目标",
        "secondary_goals": ["次要目标1", "次要目标2"],
        "business_impact": "业务影响评估",
        "urgency_level": "high/medium/low"
    }},
    "business_scenario": {{
        "primary_scenario": "主要业务场景",
        "scenario_confidence": 0.0-1.0,
        "related_scenarios": ["相关场景1", "相关场景2"]
    }},
    "analysis_requirements": {{
        "depth_level": "overview/trend/prediction/deep_modeling",
        "analysis_type": "descriptive/diagnostic/predictive/prescriptive",
        "requires_forecasting": true/false,
        "requires_scenario_analysis": true/false
    }},
    "business_logic_complexity": {{
        "complexity_level": "simple/moderate/complex/expert",
        "reasoning_steps": 估计推理步骤数,
        "involves_multiple_factors": true/false,
        "requires_business_assumptions": true/false
    }},
    "key_business_parameters": {{
        "financial_metrics": ["关键财务指标"],
        "operational_metrics": ["关键运营指标"], 
        "risk_factors": ["风险因子"],
        "external_factors": ["外部因素"]
    }},
    "expected_output_format": {{
        "format_type": "summary/detailed_analysis/dashboard/report",
        "visualization_needs": ["图表类型"],
        "actionable_insights_required": true/false
    }},
    "confidence_assessment": {{
        "understanding_confidence": 0.0-1.0,
        "clarity_score": 0.0-1.0,
        "potential_ambiguities": ["模糊点1", "模糊点2"]
    }}
}}

重点关注业务逻辑的合理性和实用性，确保分析结果能够指导后续的数据获取和计算策略。
"""

            result = await self.claude_client.analyze_complex_query(understanding_prompt, {
                "query": query,
                "context": context,
                "current_date": current_date
            })

            if result.get("success"):
                claude_analysis = result["analysis"]
                logger.info("✅ Claude业务理解完成")
                return {
                    "success": True,
                    "claude_understanding": claude_analysis,
                    "processing_method": "claude_analysis"
                }
            else:
                logger.warning(f"Claude分析失败: {result.get('error', 'Unknown error')}")
                return await self._fallback_query_understanding(query)

        except Exception as e:
            logger.error(f"Claude业务理解异常: {str(e)}")
            return await self._fallback_query_understanding(query)

    # ============= GPT数据需求分析层 =============

    async def _gpt_analyze_data_requirements(self, query: str, claude_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """GPT分析数据需求和计算要求"""

        if not self.gpt_client:
            logger.warning("GPT客户端不可用，使用基础分析")
            return self._fallback_data_requirements_analysis(query, claude_analysis)

        try:
            claude_understanding = claude_analysis.get("claude_understanding", {})

            data_analysis_prompt = f"""
基于业务分析结果，请精确分析查询的数据需求和计算要求：

原始查询: "{query}"
业务理解结果: {json.dumps(claude_understanding, ensure_ascii=False)}

可用的API数据源:
1. /api/sta/system - 系统概览 (总余额、用户统计、今日到期等)
2. /api/sta/day - 每日数据 (注册、持仓、入金、出金)
3. /api/sta/product - 产品数据 (产品列表、持有情况、到期预测)
4. /api/sta/user_daily - 用户每日数据 (VIP分布、新增用户)
5. /api/sta/user - 用户详情 (投资额、奖励、投报比)
6. /api/sta/product_end - 单日到期数据
7. /api/sta/product_end_interval - 区间到期数据

请分析并返回JSON格式结果：
{{
    "required_data_sources": {{
        "primary_apis": ["必需的主要API"],
        "secondary_apis": ["可选的辅助API"],
        "data_freshness_requirements": "realtime/daily/weekly"
    }},
    "time_range_requirements": {{
        "historical_data_needed": true/false,
        "prediction_horizon_needed": true/false,
        "minimum_historical_days": 天数,
        "optimal_historical_days": 天数
    }},
    "calculation_requirements": {{
        "basic_calculations": ["基础计算类型"],
        "advanced_calculations": ["高级计算类型"],
        "requires_financial_modeling": true/false,
        "requires_statistical_analysis": true/false,
        "calculation_complexity": "simple/moderate/complex"
    }},
    "data_processing_needs": {{
        "data_cleaning_required": true/false,
        "data_alignment_needed": true/false,
        "missing_data_handling": "ignore/interpolate/estimate",
        "outlier_detection_needed": true/false
    }},
    "performance_considerations": {{
        "estimated_data_volume": "small/medium/large",
        "processing_intensity": "low/medium/high",
        "real_time_requirements": true/false,
        "caching_beneficial": true/false
    }},
    "validation_requirements": {{
        "data_quality_checks": ["检查类型"],
        "business_logic_validation": true/false,
        "result_verification_needed": true/false
    }}
}}

重点关注数据获取的效率和计算的准确性。
"""

            result = await self.gpt_client.process_direct_query(data_analysis_prompt, {
                "query": query,
                "claude_analysis": claude_understanding
            })

            if result.get("success"):
                # 解析GPT的文本响应
                gpt_response = result["response"]

                try:
                    # 尝试从响应中提取JSON
                    import re
                    json_match = re.search(r'\{.*\}', gpt_response, re.DOTALL)
                    if json_match:
                        gpt_analysis = json.loads(json_match.group())
                        logger.info("✅ GPT数据需求分析完成")
                        return {
                            "success": True,
                            "gpt_analysis": gpt_analysis,
                            "processing_method": "gpt_analysis"
                        }
                    else:
                        # 如果没有找到JSON，使用文本解析
                        return {
                            "success": True,
                            "gpt_analysis": {"raw_response": gpt_response},
                            "processing_method": "gpt_text_analysis"
                        }

                except json.JSONDecodeError:
                    logger.warning("GPT响应JSON解析失败，使用文本分析")
                    return {
                        "success": True,
                        "gpt_analysis": {"raw_response": gpt_response},
                        "processing_method": "gpt_text_fallback"
                    }
            else:
                logger.warning(f"GPT分析失败: {result.get('error', 'Unknown error')}")
                return self._fallback_data_requirements_analysis(query, claude_analysis)

        except Exception as e:
            logger.error(f"GPT数据需求分析异常: {str(e)}")
            return self._fallback_data_requirements_analysis(query, claude_analysis)

    # ============= 复杂度分析层 =============

    async def _analyze_query_complexity(self, query: str, claude_analysis: Dict[str, Any],
                                        gpt_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析查询复杂度并确定处理策略"""

        try:
            logger.info("🔍 分析查询复杂度")

            # 从Claude分析中提取复杂度指标
            claude_data = claude_analysis.get("claude_understanding", {})
            business_logic = claude_data.get("business_logic_complexity", {})
            analysis_requirements = claude_data.get("analysis_requirements", {})

            # 从GPT分析中提取计算复杂度
            gpt_data = gpt_analysis.get("gpt_analysis", {})
            calculation_requirements = gpt_data.get("calculation_requirements", {})
            processing_needs = gpt_data.get("data_processing_needs", {})

            # 🎯 复杂度评分计算
            complexity_score = 0
            complexity_factors = []

            # 1. 业务逻辑复杂度 (0-3分)
            logic_level = business_logic.get("complexity_level", "simple")
            if logic_level == "expert":
                complexity_score += 3
                complexity_factors.append("expert_business_logic")
            elif logic_level == "complex":
                complexity_score += 2
                complexity_factors.append("complex_business_logic")
            elif logic_level == "moderate":
                complexity_score += 1
                complexity_factors.append("moderate_business_logic")

            # 2. 分析深度要求 (0-2分)
            depth_level = analysis_requirements.get("depth_level", "overview")
            if depth_level == "deep_modeling":
                complexity_score += 2
                complexity_factors.append("deep_modeling_required")
            elif depth_level == "prediction":
                complexity_score += 1.5
                complexity_factors.append("prediction_analysis")
            elif depth_level == "trend":
                complexity_score += 1
                complexity_factors.append("trend_analysis")

            # 3. 计算复杂度 (0-2分)
            calc_complexity = calculation_requirements.get("calculation_complexity", "simple")
            if calc_complexity == "complex":
                complexity_score += 2
                complexity_factors.append("complex_calculations")
            elif calc_complexity == "moderate":
                complexity_score += 1
                complexity_factors.append("moderate_calculations")

            # 4. 数据处理需求 (0-2分)
            if processing_needs.get("data_alignment_needed"):
                complexity_score += 0.5
                complexity_factors.append("data_alignment")
            if processing_needs.get("outlier_detection_needed"):
                complexity_score += 0.5
                complexity_factors.append("outlier_detection")
            if gpt_data.get("time_range_requirements", {}).get("historical_data_needed"):
                complexity_score += 1
                complexity_factors.append("historical_analysis")

            # 5. 特殊需求 (0-1分)
            if analysis_requirements.get("requires_forecasting"):
                complexity_score += 1
                complexity_factors.append("forecasting_required")
            if analysis_requirements.get("requires_scenario_analysis"):
                complexity_score += 0.5
                complexity_factors.append("scenario_analysis")

            # 🎯 确定最终复杂度等级
            if complexity_score >= 6:
                final_complexity = QueryComplexity.EXPERT
            elif complexity_score >= 4:
                final_complexity = QueryComplexity.COMPLEX
            elif complexity_score >= 2:
                final_complexity = QueryComplexity.MEDIUM
            else:
                final_complexity = QueryComplexity.SIMPLE

            # 🎯 确定查询类型
            query_type = self._determine_query_type(claude_data, gpt_data, query)

            # 🎯 确定业务场景
            business_scenario = self._determine_business_scenario(claude_data)

            # 计算置信度
            claude_confidence = claude_data.get("confidence_assessment", {}).get("understanding_confidence", 0.8)
            gpt_success = gpt_analysis.get("success", False)
            confidence_score = claude_confidence * (0.9 if gpt_success else 0.7)

            return {
                "complexity": final_complexity,
                "complexity_score": complexity_score,
                "complexity_factors": complexity_factors,
                "query_type": query_type,
                "business_scenario": business_scenario,
                "confidence_score": confidence_score,
                "analysis_metadata": {
                    "claude_logic_complexity": logic_level,
                    "gpt_calc_complexity": calc_complexity,
                    "depth_requirement": depth_level,
                    "total_complexity_indicators": len(complexity_factors)
                }
            }

        except Exception as e:
            logger.error(f"复杂度分析失败: {str(e)}")
            # 降级到基础分析
            return {
                "complexity": QueryComplexity.MEDIUM,
                "complexity_score": 2.0,
                "complexity_factors": ["fallback_analysis"],
                "query_type": QueryType.DATA_RETRIEVAL,
                "business_scenario": BusinessScenario.DAILY_OPERATIONS,
                "confidence_score": 0.5,
                "analysis_metadata": {"fallback_reason": str(e)}
            }

    def _determine_query_type(self, claude_data: Dict[str, Any],
                              gpt_data: Dict[str, Any], query: str) -> QueryType:
        """确定查询类型"""

        # 从Claude分析中获取分析类型
        analysis_type = claude_data.get("analysis_requirements", {}).get("analysis_type", "descriptive")
        depth_level = claude_data.get("analysis_requirements", {}).get("depth_level", "overview")

        # 从GPT分析中获取计算需求
        requires_forecasting = claude_data.get("analysis_requirements", {}).get("requires_forecasting", False)
        requires_scenario = claude_data.get("analysis_requirements", {}).get("requires_scenario_analysis", False)

        # 基于模式匹配
        query_lower = query.lower()

        # 优先级判断
        if requires_forecasting or "预测" in query or "预计" in query:
            return QueryType.PREDICTION
        elif requires_scenario or "假设" in query or "如果" in query:
            return QueryType.SCENARIO_SIMULATION
        elif "对比" in query or "比较" in query or analysis_type == "comparative":
            return QueryType.COMPARISON
        elif "计算" in query or "复投" in query or "提现" in query:
            return QueryType.CALCULATION
        elif "风险" in query or "安全" in query or "可持续" in query:
            return QueryType.RISK_ASSESSMENT
        elif "趋势" in query or "增长" in query or depth_level == "trend":
            return QueryType.TREND_ANALYSIS
        else:
            return QueryType.DATA_RETRIEVAL

    def _determine_business_scenario(self, claude_data: Dict[str, Any]) -> BusinessScenario:
        """确定业务场景"""

        primary_scenario = claude_data.get("business_scenario", {}).get("primary_scenario", "")

        scenario_mapping = {
            "daily_operations": BusinessScenario.DAILY_OPERATIONS,
            "financial_planning": BusinessScenario.FINANCIAL_PLANNING,
            "risk_management": BusinessScenario.RISK_MANAGEMENT,
            "growth_analysis": BusinessScenario.GROWTH_ANALYSIS,
            "user_behavior": BusinessScenario.USER_BEHAVIOR,
            "product_performance": BusinessScenario.PRODUCT_PERFORMANCE,
            "compliance_check": BusinessScenario.COMPLIANCE_CHECK
        }

        return scenario_mapping.get(primary_scenario, BusinessScenario.DAILY_OPERATIONS)

    # ============= 时间需求分析 =============

    async def _analyze_time_requirements(self, query: str, claude_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析时间和日期需求"""

        try:
            logger.info("📅 分析时间需求")

            # 使用date_utils进行AI日期解析
            date_parse_result = await self.date_utils.parse_dates_from_query(query)

            # 从Claude分析中获取时间相关信息
            claude_data = claude_analysis.get("claude_understanding", {})
            analysis_requirements = claude_data.get("analysis_requirements", {})

            # 确定时间范围需求
            if date_parse_result.has_time_info:
                # 查询中明确包含时间信息
                time_requirements = {
                    "has_explicit_time": True,
                    "parsed_dates": date_parse_result.dates,
                    "parsed_ranges": [
                        {
                            "start_date": r.start_date,
                            "end_date": r.end_date,
                            "description": r.description
                        }
                        for r in date_parse_result.ranges
                    ],
                    "relative_terms": date_parse_result.relative_terms
                }
            else:
                # 没有明确时间信息，需要推断
                inferred_range = await self.date_utils.infer_optimal_time_range(query, "analysis")
                time_requirements = {
                    "has_explicit_time": False,
                    "inferred_range": {
                        "start_date": inferred_range.start_date,
                        "end_date": inferred_range.end_date,
                        "description": inferred_range.description
                    }
                }

            # 分析时间复杂度
            requires_historical = analysis_requirements.get("requires_forecasting", False)
            depth_level = analysis_requirements.get("depth_level", "overview")

            if depth_level in ["prediction", "deep_modeling"] or requires_historical:
                time_requirements["complexity"] = "high"
                time_requirements["min_historical_days"] = 90
                time_requirements["optimal_historical_days"] = 180
            elif depth_level == "trend":
                time_requirements["complexity"] = "medium"
                time_requirements["min_historical_days"] = 30
                time_requirements["optimal_historical_days"] = 60
            else:
                time_requirements["complexity"] = "low"
                time_requirements["min_historical_days"] = 7
                time_requirements["optimal_historical_days"] = 30

            return {
                "success": True,
                "time_requirements": time_requirements,
                "date_parse_result": date_parse_result
            }

        except Exception as e:
            logger.error(f"时间需求分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_time_requirements": {
                    "has_explicit_time": False,
                    "complexity": "medium",
                    "min_historical_days": 30
                }
            }

    # ============= 执行计划生成 =============

    async def _generate_execution_plan(self, query: str, claude_analysis: Dict[str, Any],
                                       gpt_analysis: Dict[str, Any],
                                       complexity_analysis: Dict[str, Any]) -> List[ExecutionStep]:
        """生成详细的执行计划"""

        try:
            logger.info("📋 生成执行计划")

            complexity = complexity_analysis["complexity"]
            query_type = complexity_analysis["query_type"]

            execution_steps = []

            # 🔥 根据复杂度和类型生成不同的执行计划
            if complexity == QueryComplexity.SIMPLE:
                execution_steps = self._generate_simple_execution_plan(query, claude_analysis, gpt_analysis)
            elif complexity == QueryComplexity.MEDIUM:
                execution_steps = self._generate_medium_execution_plan(query, claude_analysis, gpt_analysis)
            elif complexity == QueryComplexity.COMPLEX:
                execution_steps = self._generate_complex_execution_plan(query, claude_analysis, gpt_analysis)
            else:  # EXPERT
                execution_steps = self._generate_expert_execution_plan(query, claude_analysis, gpt_analysis)

            # 为每个步骤添加依赖关系和时间估计
            self._optimize_execution_plan(execution_steps)

            logger.info(f"✅ 生成{len(execution_steps)}步执行计划")
            return execution_steps

        except Exception as e:
            logger.error(f"执行计划生成失败: {str(e)}")
            # 返回基础执行计划
            return [
                ExecutionStep(
                    step_id="fallback_step",
                    step_type="data_retrieval",
                    description="基础数据获取",
                    required_data=["system_data"],
                    processing_method="direct_api_call",
                    dependencies=[],
                    estimated_time=3.0,
                    ai_model_preference="gpt"
                )
            ]

    def _generate_simple_execution_plan(self, query: str, claude_analysis: Dict[str, Any],
                                        gpt_analysis: Dict[str, Any]) -> List[ExecutionStep]:
        """生成简单查询执行计划"""
        return [
            ExecutionStep(
                step_id="simple_data_fetch",
                step_type="data_retrieval",
                description="获取当前数据",
                required_data=["system_data"],
                processing_method="single_api_call",
                dependencies=[],
                estimated_time=2.0,
                ai_model_preference="gpt"
            ),
            ExecutionStep(
                step_id="simple_format",
                step_type="data_formatting",
                description="格式化输出",
                required_data=["system_data"],
                processing_method="ai_formatting",
                dependencies=["simple_data_fetch"],
                estimated_time=1.0,
                ai_model_preference="gpt"
            )
        ]

    def _generate_medium_execution_plan(self, query: str, claude_analysis: Dict[str, Any],
                                        gpt_analysis: Dict[str, Any]) -> List[ExecutionStep]:
        """生成中等复杂度执行计划"""
        return [
            ExecutionStep(
                step_id="medium_data_collection",
                step_type="data_collection",
                description="收集相关数据",
                required_data=["system_data", "historical_data"],
                processing_method="intelligent_data_fetch",
                dependencies=[],
                estimated_time=5.0,
                ai_model_preference="api_connector"
            ),
            ExecutionStep(
                step_id="medium_analysis",
                step_type="trend_analysis",
                description="趋势分析计算",
                required_data=["historical_data"],
                processing_method="time_series_analysis",
                dependencies=["medium_data_collection"],
                estimated_time=8.0,
                ai_model_preference="gpt"
            ),
            ExecutionStep(
                step_id="medium_insights",
                step_type="insight_generation",
                description="生成业务洞察",
                required_data=["analysis_results"],
                processing_method="ai_insight_generation",
                dependencies=["medium_analysis"],
                estimated_time=5.0,
                ai_model_preference="claude"
            )
        ]

    def _generate_complex_execution_plan(self, query: str, claude_analysis: Dict[str, Any],
                                         gpt_analysis: Dict[str, Any]) -> List[ExecutionStep]:
        """生成复杂查询执行计划"""
        return [
            ExecutionStep(
                step_id="complex_planning",
                step_type="analysis_planning",
                description="分析策略规划",
                required_data=["query_analysis"],
                processing_method="ai_planning",
                dependencies=[],
                estimated_time=3.0,
                ai_model_preference="claude"
            ),
            ExecutionStep(
                step_id="complex_data_comprehensive",
                step_type="comprehensive_data_collection",
                description="综合数据获取",
                required_data=["system_data", "historical_data", "product_data"],
                processing_method="comprehensive_data_package",
                dependencies=["complex_planning"],
                estimated_time=10.0,
                ai_model_preference="api_connector"
            ),
            ExecutionStep(
                step_id="complex_preprocessing",
                step_type="data_preprocessing",
                description="数据预处理和清洗",
                required_data=["raw_data"],
                processing_method="time_series_building",
                dependencies=["complex_data_comprehensive"],
                estimated_time=8.0,
                ai_model_preference="time_series_builder"
            ),
            ExecutionStep(
                step_id="complex_analysis",
                step_type="multi_dimensional_analysis",
                description="多维度分析",
                required_data=["processed_data"],
                processing_method="dual_ai_analysis",
                dependencies=["complex_preprocessing"],
                estimated_time=15.0,
                ai_model_preference="claude_gpt_collaboration"
            ),
            ExecutionStep(
                step_id="complex_insights",
                step_type="comprehensive_insight_generation",
                description="综合洞察生成",
                required_data=["analysis_results"],
                processing_method="expert_insight_generation",
                dependencies=["complex_analysis"],
                estimated_time=10.0,
                ai_model_preference="claude"
            )
        ]

    def _generate_expert_execution_plan(self, query: str, claude_analysis: Dict[str, Any],
                                        gpt_analysis: Dict[str, Any]) -> List[ExecutionStep]:
        """生成专家级执行计划"""
        return [
            ExecutionStep(
                step_id="expert_strategy_design",
                step_type="strategic_planning",
                description="专家级策略设计",
                required_data=["query_analysis", "business_context"],
                processing_method="ai_strategic_planning",
                dependencies=[],
                estimated_time=5.0,
                ai_model_preference="claude"
            ),
            ExecutionStep(
                step_id="expert_data_ecosystem",
                step_type="data_ecosystem_building",
                description="构建数据生态系统",
                required_data=["all_available_data"],
                processing_method="comprehensive_data_ecosystem",
                dependencies=["expert_strategy_design"],
                estimated_time=15.0,
                ai_model_preference="api_connector"
            ),
            ExecutionStep(
                step_id="expert_time_series",
                step_type="advanced_time_series_analysis",
                description="高级时间序列分析",
                required_data=["historical_data"],
                processing_method="multi_metric_time_series",
                dependencies=["expert_data_ecosystem"],
                estimated_time=12.0,
                ai_model_preference="time_series_builder"
            ),
            ExecutionStep(
                step_id="expert_modeling",
                step_type="predictive_modeling",
                description="预测建模分析",
                required_data=["time_series_data"],
                processing_method="advanced_financial_modeling",
                dependencies=["expert_time_series"],
                estimated_time=20.0,
                ai_model_preference="gpt"
            ),
            ExecutionStep(
                step_id="expert_scenario_analysis",
                step_type="scenario_simulation",
                description="场景模拟分析",
                required_data=["model_results"],
                processing_method="scenario_simulation",
                dependencies=["expert_modeling"],
                estimated_time=15.0,
                ai_model_preference="financial_calculator"
            ),
            ExecutionStep(
                step_id="expert_validation",
                step_type="result_validation",
                description="结果验证和质量检查",
                required_data=["all_results"],
                processing_method="ai_validation",
                dependencies=["expert_scenario_analysis"],
                estimated_time=8.0,
                ai_model_preference="claude_gpt_collaboration"
            ),
            ExecutionStep(
                step_id="expert_insights",
                step_type="expert_insight_synthesis",
                description="专家级洞察综合",
                required_data=["validated_results"],
                processing_method="expert_insight_synthesis",
                dependencies=["expert_validation"],
                estimated_time=12.0,
                ai_model_preference="claude"
            )
        ]

    def _optimize_execution_plan(self, execution_steps: List[ExecutionStep]):
        """优化执行计划"""

        # 检查并行执行可能性
        for i, step in enumerate(execution_steps):
            # 检查是否可以与前面的步骤并行
            if i > 0:
                previous_step = execution_steps[i - 1]
                if (step.step_type != previous_step.step_type and
                        len(set(step.required_data) & set(previous_step.required_data)) == 0):
                    # 可能可以并行执行，标记为可优化
                    if step.step_id not in previous_step.dependencies:
                        step.estimated_time *= 0.8  # 并行执行时间优化

    # ============= AI协作策略设计 =============

    def _design_ai_collaboration(self, complexity_analysis: Dict[str, Any],
                                 execution_plan: List[ExecutionStep]) -> Dict[str, Any]:
        """设计AI协作策略"""

        complexity = complexity_analysis["complexity"]
        query_type = complexity_analysis["query_type"]

        # 🤖 根据复杂度和类型设计协作策略
        if complexity == QueryComplexity.SIMPLE:
            collaboration_plan = {
                "strategy_type": "single_ai",
                "primary_ai": "gpt",
                "collaboration_level": "minimal",
                "handoff_points": [],
                "quality_gates": ["basic_validation"]
            }
        elif complexity == QueryComplexity.MEDIUM:
            collaboration_plan = {
                "strategy_type": "sequential_collaboration",
                "primary_ai": "gpt",
                "secondary_ai": "claude",
                "collaboration_level": "moderate",
                "handoff_points": ["after_calculation", "before_insight_generation"],
                "quality_gates": ["data_validation", "logic_validation"]
            }
        elif complexity == QueryComplexity.COMPLEX:
            collaboration_plan = {
                "strategy_type": "parallel_collaboration",
                "primary_ai": "claude",
                "secondary_ai": "gpt",
                "collaboration_level": "high",
                "handoff_points": ["strategy_planning", "data_analysis", "insight_synthesis"],
                "quality_gates": ["strategy_validation", "calculation_validation", "insight_validation"]
            }
        else:  # EXPERT
            collaboration_plan = {
                "strategy_type": "deep_collaboration",
                "primary_ai": "claude",
                "secondary_ai": "gpt",
                "collaboration_level": "expert",
                "handoff_points": ["strategic_planning", "data_modeling", "scenario_analysis", "result_validation",
                                   "insight_synthesis"],
                "quality_gates": ["strategy_gate", "modeling_gate", "scenario_gate", "validation_gate", "insight_gate"]
            }

        # 添加具体的AI任务分配
        collaboration_plan["ai_task_allocation"] = self._allocate_ai_tasks(execution_plan)

        return collaboration_plan

    def _allocate_ai_tasks(self, execution_plan: List[ExecutionStep]) -> Dict[str, List[str]]:
        """分配AI任务"""

        claude_tasks = []
        gpt_tasks = []

        for step in execution_plan:
            if step.ai_model_preference in ["claude", "claude_gpt_collaboration"]:
                claude_tasks.append(step.step_id)
            elif step.ai_model_preference == "gpt":
                gpt_tasks.append(step.step_id)

        return {
            "claude_tasks": claude_tasks,
            "gpt_tasks": gpt_tasks,
            "collaborative_tasks": [
                step.step_id for step in execution_plan
                if step.ai_model_preference == "claude_gpt_collaboration"
            ]
        }

    # ============= 结果构建和辅助方法 =============

    def _build_analysis_result(self, original_query: str, preprocessed_query: str,
                               claude_analysis: Dict[str, Any], gpt_analysis: Dict[str, Any],
                               complexity_analysis: Dict[str, Any], time_analysis: Dict[str, Any],
                               execution_plan: List[ExecutionStep],
                               ai_collaboration_plan: Dict[str, Any]) -> QueryAnalysisResult:
        """构建最终分析结果"""

        # 计算总预估时间
        total_estimated_time = sum(step.estimated_time for step in execution_plan)

        # 提取数据需求
        gpt_data = gpt_analysis.get("gpt_analysis", {})
        data_requirements = gpt_data.get("required_data_sources", {})
        required_apis = data_requirements.get("primary_apis", []) + data_requirements.get("secondary_apis", [])

        # 提取业务参数
        claude_data = claude_analysis.get("claude_understanding", {})
        business_parameters = claude_data.get("key_business_parameters", {})
        calculation_requirements = gpt_data.get("calculation_requirements", {})

        # 确定处理策略
        complexity = complexity_analysis["complexity"]
        if complexity == QueryComplexity.SIMPLE:
            processing_strategy = "direct_processing"
        elif complexity == QueryComplexity.MEDIUM:
            processing_strategy = "standard_analysis_pipeline"
        elif complexity == QueryComplexity.COMPLEX:
            processing_strategy = "comprehensive_analysis_pipeline"
        else:
            processing_strategy = "expert_analysis_pipeline"

        return QueryAnalysisResult(
            original_query=original_query,
            complexity=complexity_analysis["complexity"],
            query_type=complexity_analysis["query_type"],
            business_scenario=complexity_analysis["business_scenario"],
            confidence_score=complexity_analysis["confidence_score"],

            time_requirements=time_analysis.get("time_requirements", {}),
            date_parse_result=time_analysis.get("date_parse_result"),

            data_requirements=data_requirements,
            required_apis=required_apis,

            business_parameters=business_parameters,
            calculation_requirements=calculation_requirements,

            execution_plan=execution_plan,
            processing_strategy=processing_strategy,

            ai_collaboration_plan=ai_collaboration_plan,

            analysis_timestamp=datetime.now().isoformat(),
            estimated_total_time=total_estimated_time,
            processing_metadata={
                "claude_analysis_success": claude_analysis.get("success", False),
                "gpt_analysis_success": gpt_analysis.get("success", False),
                "complexity_factors": complexity_analysis.get("complexity_factors", []),
                "total_execution_steps": len(execution_plan),
                "ai_collaboration_level": ai_collaboration_plan.get("collaboration_level", "minimal")
            }
        )

    async def _preprocess_query(self, query: str) -> str:
        """预处理查询文本"""

        # 基础清理
        cleaned_query = query.strip()

        # 移除多余空格
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)

        # 标准化常见术语
        replacements = {
            '複投': '复投',
            '現金': '现金',
            '資金': '资金',
            '預測': '预测',
            '預計': '预计'
        }

        for old, new in replacements.items():
            cleaned_query = cleaned_query.replace(old, new)

        return cleaned_query

    async def _fallback_query_understanding(self, query: str) -> Dict[str, Any]:
        """降级查询理解"""

        # 基于模式匹配的基础理解
        understanding = {
            "business_intent": {
                "primary_goal": "数据查询",
                "urgency_level": "medium"
            },
            "business_scenario": {
                "primary_scenario": "daily_operations",
                "scenario_confidence": 0.6
            },
            "analysis_requirements": {
                "depth_level": "overview",
                "analysis_type": "descriptive"
            },
            "business_logic_complexity": {
                "complexity_level": "simple",
                "reasoning_steps": 1
            },
            "confidence_assessment": {
                "understanding_confidence": 0.6,
                "clarity_score": 0.7
            }
        }

        return {
            "success": True,
            "claude_understanding": understanding,
            "processing_method": "fallback_pattern_matching"
        }

    def _fallback_data_requirements_analysis(self, query: str, claude_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """降级数据需求分析"""

        # 基础数据需求推断
        analysis = {
            "required_data_sources": {
                "primary_apis": ["system"],
                "data_freshness_requirements": "daily"
            },
            "calculation_requirements": {
                "basic_calculations": ["basic_stats"],
                "calculation_complexity": "simple"
            },
            "data_processing_needs": {
                "data_cleaning_required": True,
                "missing_data_handling": "ignore"
            }
        }

        return {
            "success": True,
            "gpt_analysis": analysis,
            "processing_method": "fallback_basic_requirements"
        }

    def _create_error_analysis_result(self, query: str, error: str) -> QueryAnalysisResult:
        """创建错误分析结果"""

        error_execution_plan = [
            ExecutionStep(
                step_id="error_handling",
                step_type="error_response",
                description=f"处理解析错误: {error}",
                required_data=[],
                processing_method="error_handling",
                dependencies=[],
                estimated_time=1.0,
                ai_model_preference="system"
            )
        ]

        return QueryAnalysisResult(
            original_query=query,
            complexity=QueryComplexity.SIMPLE,
            query_type=QueryType.DATA_RETRIEVAL,
            business_scenario=BusinessScenario.DAILY_OPERATIONS,
            confidence_score=0.0,

            time_requirements={},
            date_parse_result=None,

            data_requirements={},
            required_apis=[],

            business_parameters={},
            calculation_requirements={},

            execution_plan=error_execution_plan,
            processing_strategy="error_handling",

            ai_collaboration_plan={"strategy_type": "error_handling"},

            analysis_timestamp=datetime.now().isoformat(),
            estimated_total_time=1.0,
            processing_metadata={"error": error}
        )

    def _update_processing_stats(self, result: QueryAnalysisResult):
        """更新处理统计"""

        self.processing_stats['complexity_distribution'][result.complexity.value] += 1

        query_type_key = result.query_type.value
        if query_type_key not in self.processing_stats['query_type_distribution']:
            self.processing_stats['query_type_distribution'][query_type_key] = 0
        self.processing_stats['query_type_distribution'][query_type_key] += 1

        if result.ai_collaboration_plan.get("collaboration_level") in ["high", "expert"]:
            self.processing_stats['ai_collaboration_usage'] += 1

        # 更新平均置信度
        total = self.processing_stats['total_queries']
        current_avg = self.processing_stats['average_confidence']
        new_avg = (current_avg * (total - 1) + result.confidence_score) / total
        self.processing_stats['average_confidence'] = new_avg

    # ============= 工具方法 =============

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()

    async def validate_query(self, query: str) -> Dict[str, Any]:
        """验证查询有效性"""

        if not query or len(query.strip()) == 0:
            return {"valid": False, "error": "查询为空"}

        if len(query) > 1000:
            return {"valid": False, "error": "查询过长"}

        # 使用validation_utils验证
        if self.validator:
            validation_result = await self.validator.validate_data(
                {"query": query}, "query_data"
            )
            return {
                "valid": validation_result.is_valid,
                "validation_details": validation_result
            }

        return {"valid": True}


# ============= 工厂函数 =============

def create_smart_query_parser(claude_client=None, gpt_client=None) -> SmartQueryParser:
    """
    创建智能查询解析器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        SmartQueryParser: 智能查询解析器实例
    """
    return SmartQueryParser(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建查询解析器
    parser = create_smart_query_parser()

    print("=== 智能查询解析器测试 ===")

    # 测试不同复杂度的查询
    test_queries = [
        "今天系统总余额是多少？",  # Simple
        "过去30天每日入金趋势如何？",  # Medium
        "根据过去3个月增长预测7月份如果30%复投的资金情况",  # Expert
        "假设无入金情况下公司还能运行多久？"  # Complex
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i} ---")
        print(f"查询: {query}")

        # 验证查询
        validation = await parser.validate_query(query)
        print(f"验证: {'通过' if validation['valid'] else '失败'}")

        if validation['valid']:
            # 解析查询
            result = await parser.parse_complex_query(query)
            print(f"复杂度: {result.complexity.value}")
            print(f"类型: {result.query_type.value}")
            print(f"场景: {result.business_scenario.value}")
            print(f"置信度: {result.confidence_score:.2f}")
            print(f"执行步骤: {len(result.execution_plan)}步")
            print(f"预估时间: {result.estimated_total_time:.1f}秒")

    # 统计信息
    stats = parser.get_processing_stats()
    print(f"\n=== 处理统计 ===")
    print(f"总查询数: {stats['total_queries']}")
    print(f"平均置信度: {stats['average_confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())