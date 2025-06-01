"""
🧠 Claude驱动的智能查询解析器 (重构版)
专注于让Claude一步到位完成查询理解和执行决策

核心改进:
- Claude直接决定API调用策略
- 删除冗余的GPT数据需求分析
- 简化数据结构和流程
- 专注于决策而非复杂分析
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import re
import asyncio
import traceback
from dataclasses import dataclass, field
from enum import Enum

# 导入AI客户端和工具
from core.models.claude_client import ClaudeClient
from core.models.openai_client import OpenAIClient
from utils.helpers.date_utils import DateUtils, create_date_utils

logger = logging.getLogger(__name__)


# ============= 枚举定义 =============

class QueryComplexity(Enum):
    """查询复杂度等级"""
    SIMPLE = "simple"  # 简单查询 (直接数据获取)
    MEDIUM = "medium"  # 中等复杂 (基础分析计算)
    COMPLEX = "complex"  # 复杂查询 (多步骤分析)
    EXPERT = "expert"  # 专家级 (深度预测分析)


class QueryType(Enum):
    """查询意图类型"""
    DATA_RETRIEVAL = "data_retrieval"  # 数据获取
    CALCULATION = "calculation"  # 计算请求
    TREND_ANALYSIS = "trend_analysis"  # 趋势分析
    COMPARISON = "comparison"  # 对比分析
    PREDICTION = "prediction"  # 预测请求
    SCENARIO_SIMULATION = "scenario_simulation"  # 场景模拟
    RISK_ASSESSMENT = "risk_assessment"  # 风险评估
    GENERAL_KNOWLEDGE = "general_knowledge"  # 一般知识
    UNKNOWN = "unknown"  # 未知类型


class BusinessScenario(Enum):
    """业务场景类型"""
    FINANCIAL_OVERVIEW = "financial_overview"  # 财务概览
    DAILY_OPERATIONS = "daily_operations"  # 日常运营
    USER_ANALYSIS = "user_analysis"  # 用户行为分析
    PRODUCT_ANALYSIS = "product_analysis"  # 产品表现分析
    HISTORICAL_PERFORMANCE = "historical_performance"  # 历史业绩
    FUTURE_PROJECTION = "future_projection"  # 未来预测
    RISK_MANAGEMENT = "risk_management"  # 风险管理
    UNKNOWN_SCENARIO = "unknown_scenario"  # 未知场景


# ============= 数据类定义 =============

@dataclass
class QueryAnalysisResult:
    """
    🎯 简化版查询分析结果
    专注于Claude的理解和决策结果
    """
    # 基础理解结果
    original_query: str
    complexity: QueryComplexity
    query_type: QueryType
    business_scenario: BusinessScenario
    confidence_score: float

    # 🎯 核心：Claude直接决定的执行策略
    api_calls_needed: List[Dict[str, Any]] = field(default_factory=list)
    needs_calculation: bool = False
    calculation_type: Optional[str] = None

    # 简化的时间信息
    time_range: Optional[Dict[str, str]] = None

    # 元数据
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'original_query': self.original_query,
            'complexity': self.complexity.value,
            'query_type': self.query_type.value,
            'business_scenario': self.business_scenario.value,
            'confidence_score': self.confidence_score,
            'api_calls_needed': self.api_calls_needed,
            'needs_calculation': self.needs_calculation,
            'calculation_type': self.calculation_type,
            'time_range': self.time_range,
            'analysis_timestamp': self.analysis_timestamp,
            'processing_metadata': self.processing_metadata
        }


# ============= 主要类定义 =============

class SmartQueryParser:
    """
    🧠 Claude驱动的智能查询解析器 (重构版)

    核心功能:
    1. Claude一步到位理解查询
    2. 直接决定API调用策略
    3. 判断是否需要GPT计算
    4. 输出简化的执行计划
    """

    def __init__(self, claude_client: Optional[ClaudeClient] = None, gpt_client=None):
        """
        初始化智能查询解析器

        Args:
            claude_client: Claude客户端，负责查询理解和决策
            gpt_client: 保留参数以兼容，但不再使用
        """
        self.claude_client = claude_client

        # 🆕 添加时间处理能力
        self.date_utils = create_date_utils(claude_client) if claude_client else None

        # 处理统计
        self.processing_stats = {
            'total_queries': 0,
            'successful_parses': 0,
            'fallback_parses': 0,
            'claude_failures': 0,
            'average_confidence': 0.0,
            'complexity_distribution': {
                'simple': 0, 'medium': 0, 'complex': 0, 'expert': 0
            },
            'query_type_distribution': {}
        }

        logger.info("SmartQueryParser (Refactored) initialized with Claude-driven architecture")

    # ============= 核心查询解析方法 =============

    async def parse_complex_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysisResult:
        """
        🎯 简化版查询解析 - Claude一步到位

        Args:
            query: 用户查询文本
            context: 查询上下文 (对话历史等)

        Returns:
            QueryAnalysisResult: 简化的查询分析结果
        """
        try:
            logger.info(f"🧠 Claude解析查询: {query[:50]}...")
            self.processing_stats['total_queries'] += 1

            # 基础预处理
            clean_query = self._preprocess_query(query)

            # 🎯 核心：Claude一步到位理解查询并制定执行计划
            claude_plan = await self._claude_understand_and_plan(clean_query, context)

            if not claude_plan.get("success"):
                logger.warning(f"Claude理解失败: {claude_plan.get('error')}, 使用降级解析")
                self.processing_stats['claude_failures'] += 1
                return await self._fallback_analysis(query, claude_plan.get('error', ''))

            # 构建分析结果
            result = self._build_analysis_result(query, claude_plan)

            # 更新统计
            self._update_processing_stats(result)

            logger.info(
                f"✅ Claude解析完成: {result.query_type.value} | APIs: {len(result.api_calls_needed)} | "
                f"需要计算: {result.needs_calculation} | 置信度: {result.confidence_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"❌ 查询解析失败: {str(e)}\n{traceback.format_exc()}")
            self.processing_stats['claude_failures'] += 1
            return await self._fallback_analysis(query, str(e))

    async def _claude_understand_and_plan(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        🎯 Claude一步到位：理解查询 + 决定执行策略
        """
        if not self.claude_client:
            logger.warning("Claude客户端不可用")
            return {"success": False, "error": "Claude客户端未配置"}

        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_date_api = datetime.now().strftime("%Y%m%d")

            # 🎯 让Claude直接决定API调用和计算需求
            planning_prompt = f"""
你是金融AI系统的决策大脑。请分析用户查询并直接制定执行计划。

用户查询: "{query}"
当前日期: {current_date} (API格式: {current_date_api})
对话历史: {json.dumps(context or {}, ensure_ascii=False)[:200]}

请分析并返回JSON格式的执行计划：

{{
    "query_understanding": {{
        "complexity": "simple|medium|complex|expert",
        "query_type": "data_retrieval|trend_analysis|prediction|calculation|comparison|risk_assessment|general_knowledge",
        "business_scenario": "financial_overview|daily_operations|user_analysis|product_analysis|historical_performance|future_projection|risk_management",
        "user_intent": "用户想要了解什么",
        "confidence": 0.8
    }},
    "execution_plan": {{
        "api_calls": [
            {{
                "api_method": "get_system_data",
                "params": {{}},
                "reason": "获取系统概览数据"
            }}
        ],
        "needs_calculation": false,
        "calculation_type": "statistics|trend_analysis|growth_calculation|prediction|comparison|none",
        "calculation_description": "需要GPT做什么计算"
    }},
    "time_analysis": {{
        "has_time_requirement": false,
        "start_date": "20240501", 
        "end_date": "20240531",
        "time_description": "时间范围描述"
    }}
}}

可用的API方法：
- get_system_data(): 当前系统概览 (总余额、用户统计、今日数据)
- get_daily_data(date): 特定日期的业务数据，date格式为YYYYMMDD
- get_product_data(): 产品信息和持有情况
- get_product_end_data(date): 特定日期到期产品，date格式为YYYYMMDD
- get_product_end_interval(start_date, end_date): 区间到期数据，日期格式为YYYYMMDD
- get_user_daily_data(date): 用户每日统计，date格式为YYYYMMDD
- get_user_data(page): 详细用户数据，page为页码

计算类型说明：
- statistics: 基础统计（均值、总和、增长率等）
- trend_analysis: 趋势计算和模式识别
- growth_calculation: 增长率和变化计算
- prediction: 基于历史数据的预测
- comparison: 对比分析
- none: 不需要计算，直接展示数据

分析规则：
- 如果是"今天/今日"查询 → get_system_data + get_daily_data(今日日期)
- 如果提到"历史/过去N天"时间 → 需要时间范围的API + 可能需要计算
- 如果是余额/概览查询 → get_system_data
- 如果是产品相关 → get_product_data
- 如果提到"到期" → get_product_end_* 相关API
- 如果提到用户/注册 → get_user_daily_data 或 get_user_data
- 如果需要"分析/计算/趋势/预测" → needs_calculation = true


🔍 **智能API选择规则：**

**时间相关查询：**
- 具体日期（如"6月1日"） → get_product_end_data(date) / get_daily_data(date)
- 日期区间（如"6月1日至6月30日"） → get_product_end_interval(start, end)
- "今天/今日" → get_system_data() + get_expiring_products_today()
- "本周" → get_expiring_products_week()
- "历史趋势" → get_date_range_data() 多个日期

**数据类型识别：**
- 提到"到期/过期" → 优先使用 get_product_end_* 系列API
- 提到"总资金/余额/概览" → 必须包含 get_system_data()
- 提到"入金/出金/注册" → 使用 get_daily_data()
- 提到"用户/VIP/活跃" → 使用 get_user_* 系列API
- 提到"产品/持仓" → 使用 get_product_data()

**计算需求识别：**
- 提到"复投/再投资" → calculation_type: "reinvestment_analysis"
- 提到"趋势/增长/变化" → calculation_type: "trend_analysis"  
- 提到"预测/预计/还能运行" → calculation_type: "cash_runway"
- 提到"对比/比较" → calculation_type: "comparison"

**智能组合示例：**

查询："6月1日的有多少产品到期，总资金多少"
→ API组合：get_product_end_data(20240601) + get_system_data()

查询："6月1日至6月30日产品到期金额，25%复投，7月1日剩余资金"  
→ API组合：get_product_end_interval(20240601,20240630) + get_system_data()
→ 计算：reinvestment_analysis

查询："本周到期金额和今日到期金额变化趋势"
→ API组合：get_expiring_products_today() + get_expiring_products_week() 
→ 计算：trend_analysis

查询："5月28日入金"
→ API组合：get_daily_data(20240528)

查询："目前公司活跃会员有多少"  
→ API组合：get_system_data()

查询："假设现在没入金公司还能运行多久"
→ API组合：get_system_data() + get_date_range_data(最近30天)
→ 计算：cash_runway

**日期解析增强：**
- 自动将"6月1日"转换为"20240601"格式
- 识别"上个星期"为具体日期范围
- 处理"5月11日至5月31日"为区间查询

请根据以上规则分析用户查询，选择最优的API组合策略。


日期格式要求：
- 所有API的日期参数必须使用YYYYMMDD格式（如：20240501）
- 今日日期是：{current_date_api}

请根据用户查询选择最合适的API和计算类型，确保日期格式正确。
"""
            # 调用Claude
            result = await asyncio.wait_for(
                self.claude_client.analyze_complex_query(planning_prompt, {
                    "query": query,
                    "context": context,
                    "current_date": current_date
                }),
                timeout=30.0  # 30秒超时
            )

            if result.get("success"):
                # 解析Claude的分析结果
                analysis_text = result.get("analysis", "{}")

                # 尝试提取JSON
                analysis = self._extract_json_from_response(analysis_text)

                if analysis:
                    # 验证必要字段
                    if self._validate_claude_response(analysis):
                        return {
                            "success": True,
                            "claude_plan": analysis,
                            "processing_method": "claude_integrated_planning"
                        }
                    else:
                        logger.error("Claude响应缺少必要字段")
                        return {"success": False, "error": "Claude响应格式不完整"}
                else:
                    logger.error(f"无法解析Claude的JSON响应: {analysis_text[:200]}")
                    return {"success": False, "error": "Claude响应JSON解析失败"}
            else:
                logger.error(f"Claude分析失败: {result.get('error')}")
                return {"success": False, "error": result.get('error', 'Claude调用失败')}

        except asyncio.TimeoutError:
            logger.error("Claude调用超时")
            return {"success": False, "error": "Claude响应超时"}
        except Exception as e:
            logger.error(f"Claude理解和规划异常: {str(e)}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def _validate_claude_response(self, analysis: Dict[str, Any]) -> bool:
        """验证Claude响应的完整性"""
        try:
            # 检查必要的顶级字段
            required_top_fields = ["query_understanding", "execution_plan"]
            for field in required_top_fields:
                if field not in analysis:
                    logger.error(f"Claude响应缺少字段: {field}")
                    return False

            # 检查query_understanding字段
            understanding = analysis["query_understanding"]
            required_understanding_fields = ["complexity", "query_type", "confidence"]
            for field in required_understanding_fields:
                if field not in understanding:
                    logger.error(f"query_understanding缺少字段: {field}")
                    return False

            # 检查execution_plan字段
            execution = analysis["execution_plan"]
            required_execution_fields = ["api_calls", "needs_calculation"]
            for field in required_execution_fields:
                if field not in execution:
                    logger.error(f"execution_plan缺少字段: {field}")
                    return False

            # 检查api_calls是否为列表
            if not isinstance(execution["api_calls"], list):
                logger.error("api_calls必须是列表")
                return False

            return True
        except Exception as e:
            logger.error(f"验证Claude响应时出错: {e}")
            return False

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """从Claude响应中提取JSON"""
        try:
            # 直接尝试解析整个响应
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            try:
                # 尝试提取代码块中的JSON
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # 尝试提取大括号中的内容
                brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if brace_match:
                    return json.loads(brace_match.group())

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")

        logger.error(f"无法从响应中提取有效JSON: {response_text[:300]}")
        return None

    def _build_analysis_result(self, original_query: str, claude_plan: Dict[str, Any]) -> QueryAnalysisResult:
        """构建分析结果"""
        try:
            plan_data = claude_plan["claude_plan"]
            understanding = plan_data["query_understanding"]
            execution = plan_data["execution_plan"]
            time_info = plan_data.get("time_analysis", {})

            # 🆕 增强API调用参数处理
            api_calls = []
            for api_call in execution.get("api_calls", []):
                method = api_call.get("api_method", "get_system_data")
                params = api_call.get("params", {})

                # 🎯 处理日期参数格式转换
                params = self._process_api_params(params)

                api_calls.append({
                    "method": method,
                    "params": params,
                    "reason": api_call.get("reason", "数据获取")
                })

            # 构建时间范围
            time_range = None
            if time_info.get("has_time_requirement"):
                time_range = {
                    "start_date": time_info.get("start_date"),
                    "end_date": time_info.get("end_date"),
                    "description": time_info.get("time_description", "")
                }

            # 安全地获取枚举值
            complexity = self._safe_get_enum(QueryComplexity, understanding.get("complexity", "medium"))
            query_type = self._safe_get_enum(QueryType, understanding.get("query_type", "data_retrieval"))
            business_scenario = self._safe_get_enum(BusinessScenario,
                                                    understanding.get("business_scenario", "daily_operations"))

            return QueryAnalysisResult(
                original_query=original_query,
                complexity=complexity,
                query_type=query_type,
                business_scenario=business_scenario,
                confidence_score=float(understanding.get("confidence", 0.75)),

                # 核心执行信息
                api_calls_needed=api_calls,
                needs_calculation=execution.get("needs_calculation", False),
                calculation_type=execution.get("calculation_type") if execution.get("needs_calculation") else None,

                # 时间信息
                time_range=time_range,

                processing_metadata={
                    "user_intent": understanding.get("user_intent", ""),
                    "api_count": len(api_calls),
                    "processing_method": "claude_integrated",
                    "calculation_description": execution.get("calculation_description", "")
                }
            )
        except Exception as e:
            logger.error(f"构建分析结果失败: {e}\n{traceback.format_exc()}")
            # 返回一个基本的结果
            return QueryAnalysisResult(
                original_query=original_query,
                complexity=QueryComplexity.SIMPLE,
                query_type=QueryType.DATA_RETRIEVAL,
                business_scenario=BusinessScenario.DAILY_OPERATIONS,
                confidence_score=0.5,
                api_calls_needed=[{"method": "get_system_data", "params": {}, "reason": "降级数据获取"}],
                processing_metadata={"error": str(e), "fallback": True}
            )

    def _safe_get_enum(self, enum_class, value: str):
        """安全地获取枚举值"""
        try:
            return enum_class(value)
        except ValueError:
            logger.warning(f"无效的枚举值 {value} for {enum_class.__name__}, 使用默认值")
            return list(enum_class)[0]  # 返回第一个枚举值作为默认

    def _process_api_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理API参数，特别是日期格式"""
        processed_params = params.copy()

        # 处理日期参数
        date_fields = ["date", "start_date", "end_date"]
        for field in date_fields:
            if field in processed_params and processed_params[field]:
                processed_params[field] = self._convert_date_for_api(processed_params[field])

        return processed_params

    def _convert_date_for_api(self, date_str: str) -> str:
        """将日期转换为API需要的YYYYMMDD格式"""
        if not date_str:
            return datetime.now().strftime('%Y%m%d')

        try:
            # 如果已经是YYYYMMDD格式
            if len(date_str) == 8 and date_str.isdigit():
                return date_str

            # 如果是YYYY-MM-DD格式
            if len(date_str) == 10 and '-' in date_str:
                return date_str.replace('-', '')

            # 如果是其他格式，尝试解析
            if self.date_utils:
                # 使用date_utils进行智能转换
                try:
                    parsed_date = self.date_utils.api_format_to_date(date_str)
                    return parsed_date.strftime('%Y%m%d')
                except:
                    pass

            # 默认返回今天
            logger.warning(f"无法解析日期格式: {date_str}, 使用今天日期")
            return datetime.now().strftime('%Y%m%d')

        except Exception as e:
            logger.error(f"日期转换失败: {date_str}, 错误: {e}")
            return datetime.now().strftime('%Y%m%d')

    # ============= 降级和工具方法 =============

    async def _fallback_analysis(self, query: str, error: str = "") -> QueryAnalysisResult:
        """降级分析：增强版基于规则的解析"""
        logger.info(f"执行降级解析: {query[:50]}...")
        self.processing_stats['fallback_parses'] += 1

        query_lower = query.lower()

        # 🆕 更详细的关键词判断
        if any(kw in query_lower for kw in ["今天", "今日", "当前", "现在"]):
            query_type = QueryType.DATA_RETRIEVAL
            api_calls = [
                {"method": "get_system_data", "params": {}, "reason": "获取当前系统数据"},
                {"method": "get_daily_data", "params": {"date": datetime.now().strftime('%Y%m%d')},
                 "reason": "获取今日数据"}
            ]
            complexity = QueryComplexity.SIMPLE
            scenario = BusinessScenario.DAILY_OPERATIONS

        elif any(kw in query_lower for kw in ["余额", "总资产", "总金额", "资金"]):
            query_type = QueryType.DATA_RETRIEVAL
            api_calls = [{"method": "get_system_data", "params": {}, "reason": "获取资产概览"}]
            complexity = QueryComplexity.SIMPLE
            scenario = BusinessScenario.FINANCIAL_OVERVIEW

        elif any(kw in query_lower for kw in ["产品", "到期", "持有"]):
            query_type = QueryType.DATA_RETRIEVAL
            if "到期" in query_lower:
                api_calls = [
                    {"method": "get_product_data", "params": {}, "reason": "获取产品信息"},
                    {"method": "get_product_end_data", "params": {"date": datetime.now().strftime('%Y%m%d')},
                     "reason": "获取今日到期产品"}
                ]
            else:
                api_calls = [{"method": "get_product_data", "params": {}, "reason": "获取产品信息"}]
            complexity = QueryComplexity.SIMPLE
            scenario = BusinessScenario.PRODUCT_ANALYSIS

        elif any(kw in query_lower for kw in ["用户", "注册", "活跃"]):
            query_type = QueryType.DATA_RETRIEVAL
            api_calls = [
                {"method": "get_user_daily_data", "params": {"date": datetime.now().strftime('%Y%m%d')},
                 "reason": "获取用户数据"}
            ]
            complexity = QueryComplexity.SIMPLE
            scenario = BusinessScenario.USER_ANALYSIS

        elif any(kw in query_lower for kw in ["趋势", "增长", "变化", "历史"]):
            query_type = QueryType.TREND_ANALYSIS
            api_calls = [
                {"method": "get_system_data", "params": {}, "reason": "获取基础数据进行趋势分析"},
                {"method": "get_daily_data", "params": {"date": datetime.now().strftime('%Y%m%d')},
                 "reason": "获取今日数据"}
            ]
            complexity = QueryComplexity.MEDIUM
            scenario = BusinessScenario.HISTORICAL_PERFORMANCE

        elif any(kw in query_lower for kw in ["预测", "预计", "未来", "预期"]):
            query_type = QueryType.PREDICTION
            api_calls = [{"method": "get_system_data", "params": {}, "reason": "获取数据进行预测"}]
            complexity = QueryComplexity.COMPLEX
            scenario = BusinessScenario.FUTURE_PROJECTION

        elif any(kw in query_lower for kw in ["风险", "安全", "危险"]):
            query_type = QueryType.RISK_ASSESSMENT
            api_calls = [{"method": "get_system_data", "params": {}, "reason": "获取数据进行风险评估"}]
            complexity = QueryComplexity.COMPLEX
            scenario = BusinessScenario.RISK_MANAGEMENT

        else:
            # 默认情况
            query_type = QueryType.DATA_RETRIEVAL
            api_calls = [{"method": "get_system_data", "params": {}, "reason": "获取系统概览"}]
            complexity = QueryComplexity.SIMPLE
            scenario = BusinessScenario.DAILY_OPERATIONS

        return QueryAnalysisResult(
            original_query=query,
            complexity=complexity,
            query_type=query_type,
            business_scenario=scenario,
            confidence_score=0.6,  # 降级解析置信度较低
            api_calls_needed=api_calls,
            needs_calculation=query_type in [QueryType.TREND_ANALYSIS, QueryType.PREDICTION, QueryType.CALCULATION],
            calculation_type="statistics" if query_type == QueryType.TREND_ANALYSIS else
            "prediction" if query_type == QueryType.PREDICTION else
            "calculation" if query_type == QueryType.CALCULATION else None,
            processing_metadata={
                "parsing_method": "fallback_rule_based",
                "fallback_reason": error or "Claude分析不可用",
                "keywords_matched": [kw for kw in ["今天", "余额", "产品", "用户", "趋势", "预测"] if kw in query_lower]
            }
        )

    def _preprocess_query(self, query: str) -> str:
        """预处理查询文本"""
        # 基础清理
        cleaned = query.strip()
        # 移除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # 标准化常见术语
        replacements = {
            '複投': '复投', '現金': '现金', '資金': '资金',
            '預測': '预测', '預計': '预计', '餘額': '余额'
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return cleaned

    def _update_processing_stats(self, result: QueryAnalysisResult):
        """更新处理统计"""
        self.processing_stats['successful_parses'] += 1

        # 更新复杂度分布
        complexity_key = result.complexity.value
        self.processing_stats['complexity_distribution'][complexity_key] += 1

        # 更新查询类型分布
        query_type_key = result.query_type.value
        if query_type_key not in self.processing_stats['query_type_distribution']:
            self.processing_stats['query_type_distribution'][query_type_key] = 0
        self.processing_stats['query_type_distribution'][query_type_key] += 1

        # 更新平均置信度
        total = self.processing_stats['total_queries']
        current_avg = self.processing_stats['average_confidence']
        self.processing_stats['average_confidence'] = (
                (current_avg * (total - 1) + result.confidence_score) / total
        )

    # ============= 外部接口 =============

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        total = stats['total_queries']
        if total > 0:
            stats['success_rate'] = stats['successful_parses'] / total
            stats['fallback_rate'] = stats['fallback_parses'] / total
            stats['claude_failure_rate'] = stats['claude_failures'] / total
        else:
            stats['success_rate'] = 0.0
            stats['fallback_rate'] = 0.0
            stats['claude_failure_rate'] = 0.0
        return stats

    async def validate_query(self, query: str) -> Dict[str, Any]:
        """验证查询有效性"""
        if not query or len(query.strip()) == 0:
            return {"valid": False, "error": "查询为空"}

        if len(query) > 1000:
            return {"valid": False, "error": "查询过长（超过1000字符）"}

        # 检查是否包含有意义的内容
        if len(query.strip()) < 2:
            return {"valid": False, "error": "查询内容过短"}

        return {"valid": True}

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        status = "healthy"
        issues = []

        if not self.claude_client:
            status = "degraded"
            issues.append("Claude客户端未配置")

        # 检查统计信息是否健康
        stats = self.get_processing_stats()
        if stats['total_queries'] > 10:  # 有足够样本时检查
            if stats['claude_failure_rate'] > 0.5:  # Claude失败率超过50%
                status = "degraded"
                issues.append("Claude失败率过高")

            if stats['fallback_rate'] > 0.8:  # 降级率超过80%
                status = "degraded"
                issues.append("降级解析率过高")

        return {
            "status": status,
            "claude_available": self.claude_client is not None,
            "date_utils_available": self.date_utils is not None,
            "processing_stats": stats,
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }


# ============= 工厂函数 =============

def create_smart_query_parser(claude_client: Optional[ClaudeClient] = None,
                              gpt_client=None) -> SmartQueryParser:
    """
    创建智能查询解析器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: 保留兼容性，但不再使用

    Returns:
        SmartQueryParser: 重构后的查询解析器实例
    """
    return SmartQueryParser(claude_client, gpt_client)

