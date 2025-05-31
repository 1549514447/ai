# utils/helpers/date_utils.py
"""
🗓️ AI驱动的智能日期工具类
专为金融AI分析系统设计，支持复杂的日期解析和时间范围计算

核心特点:
- AI优先的日期理解和解析
- 支持自然语言日期表达
- 智能时间范围推断
- 完全兼容API的YYYYMMDD格式
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, date
import calendar
import asyncio
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DateRange:
    """日期范围数据类"""
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    range_type: str  # specific/relative/inferred
    confidence: float  # 0.0-1.0
    description: str


@dataclass
class DateParseResult:
    """日期解析结果数据类"""
    dates: List[str]  # 具体日期列表
    ranges: List[DateRange]  # 日期范围列表
    relative_terms: List[str]  # 相对时间术语
    has_time_info: bool  # 是否包含时间信息
    parse_confidence: float  # 解析置信度
    api_format_dates: List[str]  # API格式的日期(YYYYMMDD)


class DateUtils:
    """
    🤖 AI驱动的智能日期工具类

    功能特点:
    1. AI理解自然语言日期表达
    2. 智能推断时间范围
    3. 自动格式转换(支持API格式)
    4. 复杂时间逻辑处理
    """

    def __init__(self, claude_client=None):
        """
        初始化日期工具

        Args:
            claude_client: Claude客户端实例，用于AI日期理解
        """
        self.claude_client = claude_client
        self.current_date = datetime.now().date()

        # 常用日期模式缓存
        self._pattern_cache = {}

        logger.info("DateUtils initialized with AI capabilities")

    # ============= 核心AI驱动方法 =============

    async def parse_dates_from_query(self, query: str, context: Dict[str, Any] = None) -> DateParseResult:
        """
        🧠 AI驱动的查询日期解析

        Args:
            query: 用户查询文本
            context: 额外上下文信息

        Returns:
            DateParseResult: 完整的日期解析结果
        """
        try:
            logger.info(f"🔍 AI解析查询中的日期信息: {query}")

            # 如果有Claude客户端，使用AI解析
            if self.claude_client:
                ai_result = await self._ai_parse_dates(query, context)
                if ai_result["success"]:
                    # 直接使用AI解析结果，但确保格式兼容
                    analysis_text = ai_result["analysis"]
                    
                    # 尝试将字符串解析为JSON
                    try:
                        # 查找JSON内容的开始和结束位置
                        start_idx = analysis_text.find('{')
                        end_idx = analysis_text.rfind('}') + 1
                        
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = analysis_text[start_idx:end_idx]
                            analysis = json.loads(json_str)
                            logger.info("成功从AI响应中提取JSON数据")
                        else:
                            logger.warning("无法在AI响应中找到JSON数据，使用基础解析")
                            return self._basic_date_parse(query)
                    except json.JSONDecodeError as e:
                        logger.warning(f"AI响应解析JSON失败: {e}，使用基础解析")
                        return self._basic_date_parse(query)
                    
                    # 处理日期范围，确保是DateRange对象列表
                    ranges = []
                    if "date_ranges" in analysis:
                        for range_item in analysis["date_ranges"]:
                            if isinstance(range_item, dict) and "start_date" in range_item and "end_date" in range_item:
                                ranges.append(DateRange(
                                    start_date=range_item["start_date"],
                                    end_date=range_item["end_date"],
                                    range_type=range_item.get("range_type", "explicit"),
                                    confidence=range_item.get("confidence", 0.8),
                                    description=range_item.get("description", "从AI解析的日期范围")
                                ))
                    
                    # 提取具体日期列表
                    dates = []
                    if "specific_dates" in analysis:
                        for date_item in analysis["specific_dates"]:
                            if isinstance(date_item, dict) and "parsed_date" in date_item:
                                dates.append(date_item["parsed_date"])
                    
                    # 提取相对时间术语
                    relative_terms = []
                    if "relative_periods" in analysis:
                        for period in analysis["relative_periods"]:
                            if isinstance(period, dict) and "original_text" in period:
                                relative_terms.append(period["original_text"])
                    
                    # 生成API格式日期
                    api_format_dates = []
                    for date_str in dates:
                        if date_str:  # 确保日期字符串不为空
                            api_format_dates.append(self.date_to_api_format(date_str))
                    
                    return DateParseResult(
                        dates=dates,
                        ranges=ranges,
                        relative_terms=relative_terms,
                        has_time_info=analysis.get("has_time_info", bool(dates or ranges)),
                        parse_confidence=analysis.get("parsing_confidence", 0.8),
                        api_format_dates=api_format_dates
                    )

            # 降级到基础解析
            logger.warning("Claude不可用，使用基础日期解析")
            return self._basic_date_parse(query)

        except Exception as e:
            logger.error(f"❌ 日期解析失败: {str(e)}")
            return self._create_empty_parse_result()

    async def _ai_parse_dates(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI解析日期的核心方法"""

        current_date_str = self.current_date.strftime("%Y-%m-%d")

        date_analysis_prompt = f"""
你是一个专业的时间分析专家。请从以下金融查询中提取所有时间和日期信息。

当前日期: {current_date_str}
用户查询: "{query}"
额外上下文: {json.dumps(context or {}, ensure_ascii=False)}

请识别和分析:
1. 具体日期 (如: 2024年5月1日, 5月5日, 今天, 昨天)
2. 日期范围 (如: 5月1日到31日, 过去30天, 本月)
3. 相对时间 (如: 上周, 上个月, 去年同期)
4. 隐含时间 (如: "最近的增长" -> 可能指最近1个月)

返回JSON格式:
{{
    "specific_dates": [
        {{
            "original_text": "查询中的原文",
            "parsed_date": "YYYY-MM-DD",
            "confidence": 0.0-1.0
        }}
    ],
    "date_ranges": [
        {{
            "original_text": "查询中的原文",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD", 
            "range_type": "explicit/inferred/relative",
            "confidence": 0.0-1.0,
            "description": "范围描述"
        }}
    ],
    "relative_periods": [
        {{
            "original_text": "查询中的原文",
            "period_type": "last_week/last_month/last_quarter/etc",
            "estimated_start": "YYYY-MM-DD",
            "estimated_end": "YYYY-MM-DD",
            "confidence": 0.0-1.0
        }}
    ],
    "inferred_requirements": {{
        "analysis_period": "如果没有明确时间，推荐的分析时间范围",
        "prediction_horizon": "如果涉及预测，推荐的预测时间范围",
        "data_freshness": "数据新鲜度要求: realtime/daily/weekly"
    }},
    "time_complexity": "simple/medium/complex",
    "parsing_confidence": 0.0-1.0,
    "has_time_info": true/false
}}

重要指导原则:
1. 当查询中包含"今天"、"昨天"、"明天"等明确的单日表达时:
   - 只返回对应的单日范围
   - 不要生成额外的月度或周度范围
   - 将time_complexity设置为"simple"
   - 不要在inferred_requirements中推断额外的时间范围

2. 当查询中包含"本月"、"上个月"等明确的月度表达时:
   - 返回准确的月份范围
   - 将time_complexity设置为"medium"

3. 当查询中没有明确时间表达时:
   - 根据查询类型推断合理的时间范围
   - 考虑金融分析的常见时间模式(日/周/月/季度)
   - 将time_complexity设置为"medium"或"complex"

4. 时间复杂度评估标准:
   - simple: 单日查询、明确的单个日期
   - medium: 月度查询、明确的日期范围、最近N天
   - complex: 多个时间段比较、季度/年度分析、复杂的时间模式

其他注意事项:
- 将所有日期统一转换为YYYY-MM-DD格式
- 对于"过去30天"等相对时间，计算具体的日期范围
"""

        try:
            result = await self.claude_client.analyze_complex_query(date_analysis_prompt, {
                "query": query,
                "context": context,
                "current_date": current_date_str
            })

            if result.get("success"):
                logger.info("✅ AI日期解析成功")
                return result
            else:
                logger.warning(f"⚠️ AI日期解析返回失败: {result.get('error', 'Unknown error')}")
                return {"success": False, "error": "AI analysis failed"}

        except Exception as e:
            logger.error(f"❌ AI日期解析异常: {str(e)}")
            return {"success": False, "error": str(e)}

    def _process_ai_parse_result(self, ai_analysis: Dict[str, Any]) -> DateParseResult:
        """处理AI解析结果，转换为标准格式"""

        try:
            # 确保ai_analysis是字典类型
            if isinstance(ai_analysis, str):
                try:
                    logger.info("AI分析结果是字符串类型，尝试解析为JSON")
                    # 尝试找到JSON部分
                    if '{' in ai_analysis and '}' in ai_analysis:
                        # 尝试提取JSON部分
                        start_idx = ai_analysis.find('{')
                        end_idx = ai_analysis.rfind('}') + 1
                        json_str = ai_analysis[start_idx:end_idx]
                        try:
                            ai_analysis = json.loads(json_str)
                            logger.info("成功从字符串中提取并解析JSON部分")
                        except json.JSONDecodeError:
                            logger.warning(f"提取的JSON部分解析失败: {json_str[:100]}...")
                            # 继续使用字符串解析
                    else:
                        logger.warning("字符串中没有找到JSON格式内容")
                except json.JSONDecodeError:
                    logger.error("无法将AI分析结果解析为JSON")
                    
                # 如果仍然是字符串，尝试直接从文本中提取日期信息
                if isinstance(ai_analysis, str):
                    logger.info("尝试从文本中直接提取日期信息")
                    return self._extract_dates_from_text(ai_analysis)
            
            if not isinstance(ai_analysis, dict):
                logger.error(f"AI分析结果不是字典类型: {type(ai_analysis)}")
                logger.debug(f"AI分析结果内容预览: {str(ai_analysis)[:200]}...")
                return self._create_empty_parse_result()

            # 提取具体日期
            dates = []
            for date_info in ai_analysis.get("specific_dates", []):
                if isinstance(date_info, dict):
                    dates.append(date_info.get("parsed_date", ""))
                else:
                    logger.warning(f"日期信息不是字典类型: {date_info}")

            # 提取日期范围
            ranges = []
            for range_info in ai_analysis.get("date_ranges", []):
                if isinstance(range_info, dict) and "start_date" in range_info and "end_date" in range_info:
                    ranges.append(DateRange(
                        start_date=range_info["start_date"],
                        end_date=range_info["end_date"],
                        range_type=range_info.get("range_type", "explicit"),
                        confidence=range_info.get("confidence", 0.8),
                        description=range_info.get("description", "")
                    ))
                else:
                    logger.warning(f"范围信息不是有效的字典: {range_info}")

            # 处理相对时间期间
            for period_info in ai_analysis.get("relative_periods", []):
                if isinstance(period_info, dict) and "estimated_start" in period_info and "estimated_end" in period_info:
                    ranges.append(DateRange(
                        start_date=period_info["estimated_start"],
                        end_date=period_info["estimated_end"],
                        range_type="relative",
                        confidence=period_info.get("confidence", 0.7),
                        description=period_info.get("original_text", "")
                    ))
                else:
                    logger.warning(f"时间段信息不是有效的字典: {period_info}")

            # 提取相对时间术语
            relative_terms = []
            for period in ai_analysis.get("relative_periods", []):
                if isinstance(period, dict) and "period_type" in period:
                    relative_terms.append(period["period_type"])

            # 生成API格式日期
            api_format_dates = []
            for date_str in dates:
                if date_str:  # 确保日期字符串不为空
                    api_format_dates.append(self.date_to_api_format(date_str))

            return DateParseResult(
                dates=dates,
                ranges=ranges,
                relative_terms=relative_terms,
                has_time_info=ai_analysis.get("has_time_info", True),
                parse_confidence=ai_analysis.get("parsing_confidence", 0.8),
                api_format_dates=api_format_dates
            )

        except Exception as e:
            logger.error(f"❌ AI结果处理失败: {str(e)}")
            return self._create_empty_parse_result()
            
    def _extract_dates_from_text(self, text: str) -> DateParseResult:
        """从文本中直接提取日期信息"""
        logger.info("开始从文本中提取日期信息")
        
        # 初始化结果
        dates = []
        ranges = []
        relative_terms = []
        
        try:
            # 记录原始文本的前200个字符用于调试
            logger.debug(f"文本内容预览: {text[:200]}...")
            
            # 1. 尝试提取YYYY-MM-DD格式的日期
            import re
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            found_dates = re.findall(date_pattern, text)
            
            if found_dates:
                logger.info(f"从文本中提取到 {len(found_dates)} 个日期")
                dates.extend(found_dates)
                
                # 2. 尝试识别日期范围
                # 查找相邻的日期对，可能是日期范围
                if len(found_dates) >= 2:
                    # 简单处理：将第一个和最后一个日期作为范围
                    ranges.append(DateRange(
                        start_date=found_dates[0],
                        end_date=found_dates[-1],
                        range_type="inferred",
                        confidence=0.6,
                        description="从文本推断的日期范围"
                    ))
                    logger.info(f"推断日期范围: {found_dates[0]} 到 {found_dates[-1]}")
            else:
                logger.warning("未从文本中找到标准格式的日期")
                
                # 3. 尝试识别相对时间术语
                relative_keywords = [
                    "今天", "昨天", "明天", "上周", "本周", "下周", 
                    "上个月", "本月", "下个月", "去年", "今年", "明年",
                    "过去", "最近", "未来"
                ]
                
                for keyword in relative_keywords:
                    if keyword in text:
                        relative_terms.append(keyword)
                        logger.info(f"找到相对时间术语: {keyword}")
                
                # 如果找到相对时间术语，创建一个基于当前日期的范围
                if relative_terms:
                    today = self.current_date
                    
                    # 默认为最近30天
                    start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
                    end_date = today.strftime("%Y-%m-%d")
                    
                    # 根据相对术语调整范围
                    if "上周" in relative_terms:
                        start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
                        end_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                    elif "本月" in relative_terms:
                        start_date = today.replace(day=1).strftime("%Y-%m-%d")
                    elif "上个月" in relative_terms:
                        last_month = today.replace(day=1) - timedelta(days=1)
                        start_date = last_month.replace(day=1).strftime("%Y-%m-%d")
                        end_date = last_month.strftime("%Y-%m-%d")
                    
                    ranges.append(DateRange(
                        start_date=start_date,
                        end_date=end_date,
                        range_type="relative",
                        confidence=0.5,
                        description=f"基于相对时间术语'{','.join(relative_terms)}'推断的范围"
                    ))
                    logger.info(f"基于相对时间推断范围: {start_date} 到 {end_date}")
                    
                    # 将推断的日期也添加到dates列表
                    dates.append(end_date)
            
            # 生成API格式日期
            api_format_dates = []
            for date_str in dates:
                if date_str:  # 确保日期字符串不为空
                    api_format_dates.append(self.date_to_api_format(date_str))
            
            return DateParseResult(
                dates=dates,
                ranges=ranges,
                relative_terms=relative_terms,
                has_time_info=bool(dates or ranges or relative_terms),
                parse_confidence=0.6 if dates else 0.4,
                api_format_dates=api_format_dates
            )
            
        except Exception as e:
            logger.error(f"从文本提取日期失败: {str(e)}")
            return self._create_empty_parse_result()

    async def infer_optimal_time_range(self, query: str, query_type: str = "analysis") -> DateRange:
        """
        🎯 智能推断最优时间范围

        Args:
            query: 用户查询
            query_type: 查询类型 (analysis/prediction/comparison)

        Returns:
            DateRange: 推荐的时间范围
        """

        if self.claude_client:
            return await self._ai_infer_time_range(query, query_type)
        else:
            return self._basic_infer_time_range(query_type)

    async def _ai_infer_time_range(self, query: str, query_type: str) -> DateRange:
        """AI智能推断时间范围"""

        inference_prompt = f"""
基于用户查询和分析类型，推荐最适合的时间范围：

用户查询: "{query}"
分析类型: {query_type}
当前日期: {self.current_date.strftime("%Y-%m-%d")}

推荐原则:
- 趋势分析: 通常需要30-90天数据
- 预测分析: 至少需要60天历史数据
- 对比分析: 需要完整的对比期间
- 实时查询: 当天或最新数据

请返回JSON:
{{
    "recommended_start": "YYYY-MM-DD", 
    "recommended_end": "YYYY-MM-DD",
    "reasoning": "推荐理由",
    "confidence": 0.0-1.0,
    "alternative_ranges": [
        {{
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD", 
            "description": "备选方案描述"
        }}
    ]
}}
"""

        try:
            result = await self.claude_client.analyze_complex_query(inference_prompt, {
                "query": query,
                "query_type": query_type
            })

            if result.get("success"):
                analysis = result.get("analysis")
                
                # 确保analysis是字典类型
                if not isinstance(analysis, dict):
                    logger.warning(f"AI返回的分析结果不是字典类型: {type(analysis)}")
                    return self._basic_infer_time_range(query_type)
                
                # 确保必要的键存在
                if "recommended_start" not in analysis or "recommended_end" not in analysis:
                    logger.warning("AI分析结果缺少必要的日期字段")
                    return self._basic_infer_time_range(query_type)
                
                return DateRange(
                    start_date=analysis["recommended_start"],
                    end_date=analysis["recommended_end"],
                    range_type="inferred",
                    confidence=analysis.get("confidence", 0.8),
                    description=analysis.get("reasoning", "AI推荐范围")
                )

        except Exception as e:
            logger.error(f"AI时间范围推断失败: {str(e)}")

        # 降级到基础推断
        return self._basic_infer_time_range(query_type)

    def _basic_infer_time_range(self, query_type: str) -> DateRange:
        """基础时间范围推断"""

        end_date = self.current_date

        # 根据查询类型确定默认范围
        if query_type == "realtime":
            start_date = end_date
        elif query_type == "analysis":
            start_date = end_date - timedelta(days=30)  # 30天分析
        elif query_type == "prediction":
            start_date = end_date - timedelta(days=90)  # 90天用于预测
        elif query_type == "comparison":
            start_date = end_date - timedelta(days=60)  # 60天对比
        else:
            start_date = end_date - timedelta(days=7)  # 默认一周

        return DateRange(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            range_type="inferred",
            confidence=0.6,
            description=f"基于{query_type}类型的默认范围"
        )

    # ============= 日期格式转换工具 =============

    @staticmethod
    def date_to_api_format(date_str: str) -> str:
        """
        将日期转换为API格式(YYYYMMDD)

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)

        Returns:
            str: API格式日期 (YYYYMMDD)
        """
        try:
            if len(date_str) == 8 and date_str.isdigit():
                return date_str  # 已经是API格式

            # 解析YYYY-MM-DD格式
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y%m%d")

        except ValueError:
            logger.error(f"❌ 无效的日期格式: {date_str}")
            return datetime.now().strftime("%Y%m%d")  # 返回今天

    @staticmethod
    def api_format_to_date(api_date: str) -> str:
        """
        将API格式日期转换为标准格式

        Args:
            api_date: API格式日期 (YYYYMMDD)

        Returns:
            str: 标准格式日期 (YYYY-MM-DD)
        """
        try:
            if len(api_date) != 8:
                raise ValueError(f"API日期格式错误: {api_date}")

            date_obj = datetime.strptime(api_date, "%Y%m%d")
            return date_obj.strftime("%Y-%m-%d")

        except ValueError as e:
            logger.error(f"❌ API日期转换失败: {str(e)}")
            return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def generate_date_range(start_date: str, end_date: str, format_type: str = "standard") -> List[str]:
        """
        生成日期范围列表

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            format_type: 输出格式 ("standard": YYYY-MM-DD, "api": YYYYMMDD)

        Returns:
            List[str]: 日期列表
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            if start > end:
                logger.warning(f"⚠️ 开始日期晚于结束日期: {start_date} > {end_date}")
                start, end = end, start  # 自动交换

            dates = []
            current = start

            while current <= end:
                if format_type == "api":
                    dates.append(current.strftime("%Y%m%d"))
                else:
                    dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

            logger.info(f"✅ 生成日期范围: {len(dates)}天 ({start_date} 到 {end_date})")
            return dates

        except ValueError as e:
            logger.error(f"❌ 日期范围生成失败: {str(e)}")
            return []

    # ============= 业务时间工具 =============

    def get_business_periods(self, reference_date: str = None) -> Dict[str, DateRange]:
        """
        获取常用的业务时间周期

        Args:
            reference_date: 参考日期，默认为当前日期

        Returns:
            Dict[str, DateRange]: 业务周期字典
        """

        if reference_date:
            ref_date = datetime.strptime(reference_date, "%Y-%m-%d").date()
        else:
            ref_date = self.current_date

        periods = {}

        # 今天
        periods["today"] = DateRange(
            start_date=ref_date.strftime("%Y-%m-%d"),
            end_date=ref_date.strftime("%Y-%m-%d"),
            range_type="specific",
            confidence=1.0,
            description="今天"
        )

        # 昨天
        yesterday = ref_date - timedelta(days=1)
        periods["yesterday"] = DateRange(
            start_date=yesterday.strftime("%Y-%m-%d"),
            end_date=yesterday.strftime("%Y-%m-%d"),
            range_type="specific",
            confidence=1.0,
            description="昨天"
        )

        # 本周 (周一到今天)
        days_since_monday = ref_date.weekday()
        week_start = ref_date - timedelta(days=days_since_monday)
        periods["this_week"] = DateRange(
            start_date=week_start.strftime("%Y-%m-%d"),
            end_date=ref_date.strftime("%Y-%m-%d"),
            range_type="relative",
            confidence=1.0,
            description="本周"
        )

        # 上周 (完整一周)
        last_week_end = week_start - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        periods["last_week"] = DateRange(
            start_date=last_week_start.strftime("%Y-%m-%d"),
            end_date=last_week_end.strftime("%Y-%m-%d"),
            range_type="relative",
            confidence=1.0,
            description="上周"
        )

        # 本月 (月初到今天)
        month_start = ref_date.replace(day=1)
        periods["this_month"] = DateRange(
            start_date=month_start.strftime("%Y-%m-%d"),
            end_date=ref_date.strftime("%Y-%m-%d"),
            range_type="relative",
            confidence=1.0,
            description="本月"
        )

        # 上月 (完整一个月)
        last_month_end = month_start - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        periods["last_month"] = DateRange(
            start_date=last_month_start.strftime("%Y-%m-%d"),
            end_date=last_month_end.strftime("%Y-%m-%d"),
            range_type="relative",
            confidence=1.0,
            description="上月"
        )

        # 最近7天
        week_ago = ref_date - timedelta(days=6)
        periods["last_7_days"] = DateRange(
            start_date=week_ago.strftime("%Y-%m-%d"),
            end_date=ref_date.strftime("%Y-%m-%d"),
            range_type="relative",
            confidence=1.0,
            description="最近7天"
        )

        # 最近30天
        month_ago = ref_date - timedelta(days=29)
        periods["last_30_days"] = DateRange(
            start_date=month_ago.strftime("%Y-%m-%d"),
            end_date=ref_date.strftime("%Y-%m-%d"),
            range_type="relative",
            confidence=1.0,
            description="最近30天"
        )

        return periods

    def calculate_business_days(self, start_date: str, end_date: str) -> int:
        """
        计算工作日天数 (排除周末)

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            int: 工作日天数
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()

            business_days = 0
            current = start

            while current <= end:
                if current.weekday() < 5:  # 0-4是周一到周五
                    business_days += 1
                current += timedelta(days=1)

            return business_days

        except ValueError as e:
            logger.error(f"❌ 工作日计算失败: {str(e)}")
            return 0

    # ============= 辅助方法 =============

    def _basic_date_parse(self, query: str) -> DateParseResult:
        """基础日期解析 (降级方案)"""

        # 简单的关键词匹配
        today_keywords = ["今天", "今日", "当天"]
        yesterday_keywords = ["昨天", "昨日"]

        dates = []
        ranges = []
        relative_terms = []

        query_lower = query.lower()

        # 检测今天
        if any(keyword in query for keyword in today_keywords):
            dates.append(self.current_date.strftime("%Y-%m-%d"))
            relative_terms.append("today")

        # 检测昨天
        if any(keyword in query for keyword in yesterday_keywords):
            yesterday = (self.current_date - timedelta(days=1)).strftime("%Y-%m-%d")
            dates.append(yesterday)
            relative_terms.append("yesterday")

        # 检测"最近"、"过去"等
        if "最近" in query or "过去" in query:
            # 默认最近30天
            end_date = self.current_date.strftime("%Y-%m-%d")
            start_date = (self.current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            ranges.append(DateRange(
                start_date=start_date,
                end_date=end_date,
                range_type="inferred",
                confidence=0.6,
                description="推断的最近时期"
            ))
            relative_terms.append("recent_period")

        api_format_dates = [self.date_to_api_format(d) for d in dates]

        return DateParseResult(
            dates=dates,
            ranges=ranges,
            relative_terms=relative_terms,
            has_time_info=bool(dates or ranges or relative_terms),
            parse_confidence=0.5,  # 基础解析置信度较低
            api_format_dates=api_format_dates
        )

    def _create_empty_parse_result(self) -> DateParseResult:
        """创建空的解析结果"""
        return DateParseResult(
            dates=[],
            ranges=[],
            relative_terms=[],
            has_time_info=False,
            parse_confidence=0.0,
            api_format_dates=[]
        )

    # ============= 验证和工具方法 =============

    @staticmethod
    def validate_date_format(date_str: str, date_format: str = "%Y-%m-%d") -> bool:
        """
        验证日期格式

        Args:
            date_str: 日期字符串
            date_format: 期望的日期格式

        Returns:
            bool: 是否有效
        """
        try:
            datetime.strptime(date_str, date_format)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_api_date_format(api_date: str) -> bool:
        """验证API日期格式 (YYYYMMDD)"""
        return (
                len(api_date) == 8 and
                api_date.isdigit() and
                DateUtils.validate_date_format(api_date, "%Y%m%d")
        )

    def get_date_info(self, date_str: str) -> Dict[str, Any]:
        """
        获取日期的详细信息

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)

        Returns:
            Dict: 日期详细信息
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

            return {
                "date": date_str,
                "api_format": date_obj.strftime("%Y%m%d"),
                "year": date_obj.year,
                "month": date_obj.month,
                "day": date_obj.day,
                "weekday": date_obj.strftime("%A"),
                "weekday_cn": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date_obj.weekday()],
                "is_weekend": date_obj.weekday() >= 5,
                "is_month_start": date_obj.day == 1,
                "is_month_end": date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1],
                "quarter": (date_obj.month - 1) // 3 + 1,
                "days_from_today": (date_obj - self.current_date).days
            }

        except ValueError as e:
            logger.error(f"❌ 日期信息获取失败: {str(e)}")
            return {}


# ============= 工厂函数 =============

def create_date_utils(claude_client=None) -> DateUtils:
    """
    创建日期工具实例

    Args:
        claude_client: Claude客户端实例

    Returns:
        DateUtils: 日期工具实例
    """
    return DateUtils(claude_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建日期工具 (不带AI)
    date_utils = create_date_utils()

    # 基础功能测试
    print("=== 基础日期功能测试 ===")

    # 生成日期范围
    dates = date_utils.generate_date_range("2024-06-01", "2024-06-07", "api")
    print(f"API格式日期范围: {dates}")

    # 获取业务时间周期
    periods = date_utils.get_business_periods()
    print(f"本周: {periods['this_week'].start_date} 到 {periods['this_week'].end_date}")
    print(f"最近30天: {periods['last_30_days'].start_date} 到 {periods['last_30_days'].end_date}")

    # 日期格式转换
    api_date = date_utils.date_to_api_format("2024-06-01")
    standard_date = date_utils.api_format_to_date("20240601")
    print(f"格式转换: 2024-06-01 -> {api_date} -> {standard_date}")

    # 工作日计算
    business_days = date_utils.calculate_business_days("2024-06-01", "2024-06-07")
    print(f"工作日天数: {business_days}")

    # 基础日期解析 (不使用AI)
    parse_result = await date_utils.parse_dates_from_query("给我看看昨天的数据")
    print(f"解析结果: {parse_result.dates}, 置信度: {parse_result.parse_confidence}")


if __name__ == "__main__":
    asyncio.run(main())