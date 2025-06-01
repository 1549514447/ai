# core/detectors/query_type_detector.py
from enum import Enum
from typing import List, Dict, Any
import re


class QueryType(Enum):
    """查询类型枚举"""
    SIMPLE_DATA = "simple_data"  # 简单数据查询
    COMPARISON = "comparison"  # 对比分析
    REINVESTMENT = "reinvestment"  # 复投计算
    PREDICTION = "prediction"  # 预测分析
    HISTORICAL_REVIEW = "historical"  # 历史回顾
    RISK_ASSESSMENT = "risk"  # 风险评估
    AGGREGATION = "aggregation"  # 🆕 新增聚合计算类型


class QueryTypeResult:
    """查询类型检测结果"""

    def __init__(self, query_type: QueryType, confidence: float,
                 keywords_matched: List[str] = None, special_requirements: Dict[str, Any] = None):
        self.type = query_type
        self.confidence = confidence
        self.keywords_matched = keywords_matched or []
        self.special_requirements = special_requirements or {}


class QueryTypeDetector:
    """简单高效的查询类型检测器"""

    def __init__(self):
        # 关键词模式定义
        self.patterns = {
            QueryType.REINVESTMENT: [
                r'复投', r'再投资', r'重新投资',
                r'\d+%.*复投', r'复投.*\d+%',
                r'到期.*复投', r'复投.*剩余'
            ],
            QueryType.AGGREGATION: [  # 🆕 新增聚合模式
                r'平均', r'均值', r'平均数', r'平均值',
                r'总计', r'合计', r'汇总', r'统计',
                r'每日平均', r'月均', r'日均',
                r'\d+月.*平均', r'最近.*平均',
                r'平均', r'均值', r'平均数', r'平均值',
            ],
            QueryType.PREDICTION: [
                r'预测', r'预计', r'估计', r'还能运行',
                r'多久', r'什么时候', r'会如何',
                r'假设.*没入金', r'能维持多长时间'
            ],

            QueryType.COMPARISON: [
                r'比较', r'相比', r'对比', r'变化',
                r'与.*相比', r'和.*比较',
                r'本周.*上周', r'今天.*昨天'
            ],

            QueryType.HISTORICAL_REVIEW: [
                r'历史', r'过去', r'回顾', r'总结',
                r'最近\d+[天月年]', r'历年', r'趋势',
                r'\d+月份', r'上个月', r'本月'  # 🆕 增强时间范围识别
            ],

            QueryType.RISK_ASSESSMENT: [
                r'风险', r'安全', r'危险', r'评估',
                r'风险评估', r'安全性'
            ]
        }

    def detect(self, user_query: str) -> QueryTypeResult:
        """检测查询类型"""
        query_lower = user_query.lower()

        # 检测每种类型的匹配情况
        type_scores = {}
        all_matched_keywords = {}

        for query_type, patterns in self.patterns.items():
            matched_keywords = []
            score = 0

            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    matched_keywords.extend(matches)
                    score += len(matches)

            if score > 0:
                type_scores[query_type] = score
                all_matched_keywords[query_type] = matched_keywords

        # 如果没有匹配到特殊类型，返回简单数据查询
        if not type_scores:
            return QueryTypeResult(
                query_type=QueryType.SIMPLE_DATA,
                confidence=0.9,
                keywords_matched=[],
                special_requirements={}
            )

        # 找到得分最高的类型
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        matched_keywords = all_matched_keywords[best_type]

        # 计算置信度
        confidence = min(0.95, 0.7 + (best_score * 0.1))

        # 生成特殊要求
        special_requirements = self._generate_special_requirements(
            best_type, matched_keywords, user_query
        )

        return QueryTypeResult(
            query_type=best_type,
            confidence=confidence,
            keywords_matched=matched_keywords,
            special_requirements=special_requirements
        )

    def _generate_special_requirements(self, query_type: QueryType,
                                       keywords: List[str], query: str) -> Dict[str, Any]:
        """根据查询类型生成特殊处理要求"""

        requirements = {}

        if query_type == QueryType.REINVESTMENT:
            requirements = {
                'needs_calculation': True,
                'calculation_type': 'reinvestment_analysis',
                'requires_apis': ['get_product_end_interval', 'get_system_data'],
                'prompt_enhancement': 'reinvestment_calculation'
            }

            # 提取复投比例
            import re
            percentage_match = re.search(r'(\d+)%', query)
            if percentage_match:
                requirements['reinvestment_rate'] = float(percentage_match.group(1)) / 100

        elif query_type == QueryType.PREDICTION:
            requirements = {
                'needs_calculation': True,
                'calculation_type': 'cash_runway',
                'requires_apis': ['get_system_data', 'get_daily_data'],
                'prompt_enhancement': 'prediction_analysis'
            }

        elif query_type == QueryType.COMPARISON:
            requirements = {
                'needs_calculation': True,
                'calculation_type': 'comparison_analysis',
                'requires_apis': ['get_daily_data'],
                'prompt_enhancement': 'comparison_analysis'
            }

        return requirements


# 工厂函数
def create_query_type_detector() -> QueryTypeDetector:
    """创建查询类型检测器实例"""
    return QueryTypeDetector()