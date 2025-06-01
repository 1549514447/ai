"""
Claude智能提取器 - 第二层 (完整版)
负责使用Claude理解语义化数据并提取业务信息
"""

import logging
import json
import re
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ClaudeIntelligentExtractor:
    """Claude智能提取器"""

    def __init__(self, claude_client):
        self.claude_client = claude_client
        self.extraction_templates = self._load_extraction_templates()

    async def extract_with_intelligence(self,
                                      semantic_data_result: Dict[str, Any],
                                      user_query: str,
                                      query_analysis) -> Dict[str, Any]:
        """
        Claude智能提取主方法

        Args:
            semantic_data_result: 语义化收集器的结果
            user_query: 用户原始查询
            query_analysis: 查询分析结果

        Returns:
            Claude提取的智能结果
        """
        try:
            if 'error' in semantic_data_result:
                logger.error(f"语义数据收集失败: {semantic_data_result['error']}")
                return self._create_error_result(semantic_data_result['error'])

            semantic_data = semantic_data_result.get('semantic_data', {})
            if not semantic_data:
                return self._create_error_result("无有效的语义化数据")

            # 🎯 根据查询意图选择提取策略
            intent = getattr(query_analysis, 'intent', '数据查询')
            extraction_strategy = self._determine_extraction_strategy(intent, semantic_data)

            # 🎯 构建针对性的提取提示词
            extraction_prompt = self._build_extraction_prompt(
                semantic_data, user_query, query_analysis, extraction_strategy
            )

            # 🎯 调用Claude进行智能提取
            claude_result = await self._call_claude_extraction(extraction_prompt)

            if claude_result.get('success'):
                # 🎯 验证和增强Claude的提取结果
                validated_result = self._validate_and_enhance_result(
                    claude_result['extracted_data'], semantic_data, query_analysis
                )
                return validated_result
            else:
                # Claude提取失败，使用智能降级
                logger.warning("Claude提取失败，启用智能降级")
                return self._intelligent_fallback_extraction(semantic_data, query_analysis)

        except Exception as e:
            logger.error(f"Claude智能提取异常: {e}")
            return self._intelligent_fallback_extraction(semantic_data, query_analysis)

    def _determine_extraction_strategy(self, intent: str, semantic_data: Dict[str, Any]) -> str:
        """确定提取策略"""

        # 检查是否是对比查询
        if any(keyword in intent for keyword in ['比较', '变化', '对比', '相比']):
            return 'comparison_analysis'

        # 检查是否是趋势查询
        if any(keyword in intent for keyword in ['趋势', '增长', '变化率']):
            return 'trend_analysis'

        # 检查是否是计算查询
        if any(keyword in intent for keyword in ['计算', '复投', '预测']):
            return 'calculation_analysis'

        # 检查是否是汇总查询
        if any(keyword in intent for keyword in ['总计', '汇总', '统计']):
            return 'summary_analysis'

        # 检查语义数据的特征
        semantic_keys = list(semantic_data.keys())

        # 如果有多个时间维度的数据，很可能是对比
        time_dimensions = sum(1 for key in semantic_keys
                            if any(time_word in key for time_word in
                                 ['current_week', 'last_week', 'today', 'yesterday']))

        if time_dimensions >= 2:
            return 'comparison_analysis'
        elif time_dimensions == 1:
            return 'single_period_analysis'
        else:
            return 'general_analysis'

    def _build_extraction_prompt(self,
                               semantic_data: Dict[str, Any],
                               user_query: str,
                               query_analysis,
                               strategy: str) -> str:
        """构建提取提示词"""

        base_info = f"""
        作为金融数据分析专家，请从语义化数据中智能提取信息：
        
        用户查询: "{user_query}"
        查询意图: {getattr(query_analysis, 'intent', '数据查询')}
        当前时间: {datetime.now().strftime('%Y年%m月%d日')}
        提取策略: {strategy}
        
        语义化数据:
        {json.dumps(semantic_data, ensure_ascii=False, indent=2)}
        """

        # 🎯 根据策略选择专门的提示词模板
        if strategy == 'comparison_analysis':
            return self._build_comparison_prompt(base_info, semantic_data)
        elif strategy == 'trend_analysis':
            return self._build_trend_prompt(base_info, semantic_data)
        elif strategy == 'calculation_analysis':
            return self._build_calculation_prompt(base_info, semantic_data)
        else:
            return self._build_general_prompt(base_info, semantic_data)

    def _build_calculation_prompt(self, base_info: str, semantic_data: Dict[str, Any]) -> str:
        """构建计算分析提示词"""
        return f"""
        {base_info}

        🎯 **计算分析任务**：
        1. 识别计算类型（复投、预测、收益计算等）
        2. 提取计算所需的基础数据
        3. 执行精确的数值计算
        4. 提供计算过程的透明展示

        **严格按照以下JSON格式返回**：
        {{
            "extraction_type": "calculation_analysis",
            "calculation_type": "reinvestment_analysis|yield_calculation|prediction",
            "base_data": {{
                "到期金额": 到期总金额,
                "当前余额": 系统余额,
                "计算参数": {{
                    "提现比例": 0.5,
                    "复投比例": 0.5
                }}
            }},
            "detailed_calculations": {{
                "到期金额": {{
                    "calculation_formula": "从API数据中提取",
                    "step_by_step": [
                        "步骤1: 获取5月11日至5月31日到期产品数据",
                        "步骤2: 汇总所有到期金额",
                        "步骤3: 计算利息收益"
                    ],
                    "total_expiry_amount": 具体金额,
                    "interest_earned": 利息金额
                }},
                "复投分配": {{
                    "calculation_formula": "到期总金额 × 各比例",
                    "step_by_step": [
                        "步骤1: 提现金额 = 到期总金额 × 50%",
                        "步骤2: 复投金额 = 到期总金额 × 50%"
                    ],
                    "withdrawal_amount": 提现金额,
                    "reinvestment_amount": 复投金额
                }}
            }},
            "key_insights": [
                "基于实际数据的计算结果摘要",
                "风险提示和建议"
            ],
            "extraction_confidence": 0.92
        }}

        **重要提醒**：
        - 所有计算必须基于实际API数据
        - 提供完整的计算步骤
        - 确保数值精确性
        - 标明数据来源和计算假设
        """
    def _build_comparison_prompt(self, base_info: str, semantic_data: Dict[str, Any]) -> str:
        """构建对比分析提示词"""
        return f"""
        {base_info}
        
        🎯 **对比分析任务**：
        1. 自动识别对比的时间段（如本周vs上周、今天vs昨天）
        2. 提取每个时间段的关键业务指标
        3. 计算变化量、变化率和变化方向
        4. 评估变化的业务意义和重要程度
        
        **严格按照以下JSON格式返回**：
        {{
            "extraction_type": "comparison_analysis",
            "comparison_periods": {{
                "current_period": {{
                    "period_name": "本周（5月26日-6月1日）",
                    "semantic_key": "current_week_get_daily_data",
                    "key_metrics": {{
                        "入金": 229469.35,
                        "出金": 185886.25,
                        "净流入": 43583.10,
                        "注册人数": 156
                    }},
                    "period_summary": "数据完整度较好，涵盖7天"
                }},
                "baseline_period": {{
                    "period_name": "上周（5月19日-5月25日）",
                    "semantic_key": "last_week_get_daily_data", 
                    "key_metrics": {{
                        "入金": 198234.50,
                        "出金": 167234.80,
                        "净流入": 31000.70,
                        "注册人数": 134
                    }},
                    "period_summary": "数据完整度较好，涵盖7天"
                }}
            }},
            "comparison_analysis": {{
                "入金": {{
                    "current_value": 229469.35,
                    "baseline_value": 198234.50,
                    "absolute_change": 31234.85,
                    "percentage_change": 0.157,
                    "change_direction": "增长",
                    "significance_level": "显著",
                    "business_impact": "正面"
                }},
                "出金": {{
                    "current_value": 185886.25,
                    "baseline_value": 167234.80,
                    "absolute_change": 18651.45,
                    "percentage_change": 0.111,
                    "change_direction": "增长",
                    "significance_level": "中等",
                    "business_impact": "需关注"
                }},
                "净流入": {{
                    "current_value": 43583.10,
                    "baseline_value": 31000.70,
                    "absolute_change": 12582.40,
                    "percentage_change": 0.406,
                    "change_direction": "大幅增长",
                    "significance_level": "非常显著",
                    "business_impact": "非常正面"
                }}
            }},
            "key_insights": [
                "本周入金较上周增长15.7%，显示业务增长势头良好",
                "出金虽然也有增长，但净流入实现40.6%的大幅增长",
                "注册用户数增长16.4%，用户增长与资金增长保持同步"
            ],
            "data_quality_assessment": {{
                "overall_quality": 0.95,
                "completeness": 0.98,
                "consistency": 0.92,
                "issues": []
            }},
            "extraction_confidence": 0.94
        }}
        
        **重要提醒**：
        - 所有百分比 = (新值 - 旧值) / 旧值
        - 金额保留2位小数，百分比保留3位小数
        - 必须基于实际数据计算，不能编造数字
        - 识别数据异常并在issues中说明
        """

    def _build_trend_prompt(self, base_info: str, semantic_data: Dict[str, Any]) -> str:
        """构建趋势分析提示词"""
        return f"""
        {base_info}
        
        🎯 **趋势分析任务**：
        1. 识别时间序列数据中的趋势模式
        2. 计算趋势强度和方向
        3. 分析周期性和季节性特征
        4. 预测短期发展方向
        
        **严格按照以下JSON格式返回**：
        {{
            "extraction_type": "trend_analysis",
            "time_series_data": {{
                "data_points": [
                    {{
                        "period": "5月26日",
                        "key_metrics": {{
                            "入金": 32781.34,
                            "出金": 26555.18,
                            "净流入": 6226.16
                        }}
                    }},
                    {{
                        "period": "5月27日",
                        "key_metrics": {{
                            "入金": 33156.89,
                            "出金": 26890.44,
                            "净流入": 6266.45
                        }}
                    }}
                ],
                "period_type": "daily",
                "total_periods": 7
            }},
            "trend_analysis": {{
                "入金": {{
                    "trend_direction": "上升",
                    "trend_strength": "中等",
                    "growth_rate": 0.023,
                    "volatility": "低",
                    "trend_score": 0.75
                }},
                "出金": {{
                    "trend_direction": "上升",
                    "trend_strength": "轻微",
                    "growth_rate": 0.015,
                    "volatility": "中等",
                    "trend_score": 0.65
                }},
                "净流入": {{
                    "trend_direction": "稳定上升",
                    "trend_strength": "强",
                    "growth_rate": 0.045,
                    "volatility": "低",
                    "trend_score": 0.88
                }}
            }},
            "pattern_recognition": {{
                "weekly_pattern": "工作日较高，周末较低",
                "daily_peak": "上午10-12点",
                "seasonality": "月初较活跃",
                "anomalies": []
            }},
            "trend_predictions": {{
                "short_term_outlook": "继续上升趋势",
                "confidence_level": 0.82,
                "risk_factors": ["市场波动", "季节性影响"],
                "recommended_actions": ["维持当前策略", "关注用户反馈"]
            }},
            "key_insights": [
                "入金呈现稳定上升趋势，增长率2.3%",
                "净流入趋势强劲，显示健康的资金流动",
                "建议关注出金增长，确保流动性充足"
            ],
            "extraction_confidence": 0.87
        }}
        
        **分析要点**：
        - 趋势强度：强/中等/轻微/无明显趋势
        - 增长率：日增长率或周增长率
        - 波动性：高/中等/低
        - 预测置信度：基于数据质量和趋势稳定性
        """

    def _build_comparison_prompt(self, base_info: str, semantic_data: Dict[str, Any]) -> str:
        """构建对比分析提示词 - 增强版"""
        return f"""
        {base_info}

        🎯 **对比分析任务 - 详细版**：
        1. 自动识别对比的时间段（如本周vs上周、今天vs昨天）
        2. 提取每个时间段的**每日明细数据**
        3. 计算变化量、变化率和变化方向
        4. 保留**完整的数据轨迹**用于透明展示
        5. 基于用户的问题进行计算

        **严格按照以下JSON格式返回**：
        {{
            "extraction_type": "comparison_analysis",
            "raw_data_details": {{
                "current_period_daily": [
                    {{
                        "date": "2025-05-26",
                        "入金": 32781.34,
                        "出金": 26555.18,
                        "净流入": 6226.16,
                        "注册人数": 23
                    }},
                    {{
                        "date": "2025-05-27", 
                        "入金": 33156.89,
                        "出金": 26890.44,
                        "净流入": 6266.45,
                        "注册人数": 25
                    }}
                    // ... 继续每一天
                ],
                "baseline_period_daily": [
                    {{
                        "date": "2025-05-19",
                        "入金": 28345.67,
                        "出金": 23456.78,
                        "净流入": 4888.89,
                        "注册人数": 19
                    }}
                    // ... 继续每一天
                ]
            }},
            "aggregated_totals": {{
                "current_period": {{
                    "period_name": "本周（5月26日-6月1日）",
                    "total_days": 7,
                    "总入金": 229469.35,
                    "总出金": 185886.25,
                    "总净流入": 43583.10,
                    "总注册人数": 156,
                    "日均入金": 32781.34,
                    "日均出金": 26555.18
                }},
                "baseline_period": {{
                    "period_name": "上周（5月19日-5月25日）",
                    "total_days": 7,
                    "总入金": 198234.50,
                    "总出金": 167234.80,
                    "总净流入": 31000.70,
                    "总注册人数": 134,
                    "日均入金": 28319.21,
                    "日均出金": 23890.69
                }}
            }},
            "detailed_calculations": {{
                "入金": {{
                    "calculation_formula": "(229469.35 - 198234.50) / 198234.50 * 100",
                    "step_by_step": [
                        "步骤1: 本周总入金 = 32781.34 + 33156.89 + 31945.67 + 34567.23 + 35890.45 + 29123.56 + 32004.21 = 229,469.35元",
                        "步骤2: 上周总入金 = 28345.67 + 29876.54 + 27892.33 + 30456.78 + 32123.45 + 26789.12 + 22750.61 = 198,234.50元",
                        "步骤3: 变化金额 = 229,469.35 - 198,234.50 = 31,234.85元",
                        "步骤4: 变化率 = 31,234.85 ÷ 198,234.50 × 100% = 15.75%"
                    ],
                    "current_value": 229469.35,
                    "baseline_value": 198234.50,
                    "absolute_change": 31234.85,
                    "percentage_change": 0.1575,
                    "change_direction": "增长",
                    "significance_level": "显著"
                }},
                "出金": {{
                    "calculation_formula": "(185886.25 - 167234.80) / 167234.80 * 100",
                    "step_by_step": [
                        "步骤1: 本周总出金 = 26555.18 + 26890.44 + 25123.78 + 27890.12 + 28456.90 + 24567.89 + 26401.94 = 185,886.25元",
                        "步骤2: 上周总出金 = 23456.78 + 24567.89 + 22890.45 + 25678.90 + 26789.12 + 22345.67 + 21505.99 = 167,234.80元",
                        "步骤3: 变化金额 = 185,886.25 - 167,234.80 = 18,651.45元",
                        "步骤4: 变化率 = 18,651.45 ÷ 167,234.80 × 100% = 11.15%"
                    ],
                    "current_value": 185886.25,
                    "baseline_value": 167234.80,
                    "absolute_change": 18651.45,
                    "percentage_change": 0.1115,
                    "change_direction": "增长",
                    "significance_level": "中等"
                }}
            }},
            "data_validation": {{
                "data_anomalies": [],
                "quality_checks": {{
                    "duplicate_values": false,
                    "missing_dates": false,
                    "unrealistic_values": false,
                    "zero_variance_detected": false
                }},
                "confidence_assessment": {{
                    "data_completeness": 1.0,
                    "calculation_accuracy": 0.98,
                    "business_logic_consistency": 0.95
                }}
            }},
            "business_insights": [
                "本周工作日平均入金33,892元，较上周的29,539元增长14.7%",
                "周末表现：本周周末日均30,564元，上周24,770元，提升23.4%",
                "出金控制良好：入金增长15.75%的同时，出金仅增长11.15%",
                "净流入大幅改善：从上周31,000元增至43,583元，增幅40.6%"
            ],
            "extraction_confidence": 0.94
        }}

        **关键要求**：
        - 必须提供每日明细数据，不能只有汇总
        - 计算步骤要完整，包含具体的加法过程
        - 所有数值必须基于实际数据，不能编造
        - 如果发现数据异常（如完全相同的值），必须在data_anomalies中标注
        - 百分比计算精确到小数点后2位
        """

    def _build_general_prompt(self, base_info: str, semantic_data: Dict[str, Any]) -> str:
        """构建通用分析提示词"""
        return f"""
        {base_info}
        
        🎯 **通用数据提取任务**：
        1. 识别并提取所有可用的业务指标
        2. 计算衍生指标（如净流入、活跃率等）
        3. 评估数据质量和完整性
        4. 提供数据摘要和业务洞察
        
        **返回JSON格式**：
        {{
            "extraction_type": "general_analysis",
            "extracted_metrics": {{
                "primary_metrics": {{
                    "入金": 171403.26,
                    "出金": 161710.52,
                    "总余额": 8223695.07,
                    "活跃用户数": 3911,
                    "注册人数": 87
                }},
                "derived_metrics": {{
                    "净流入": 9692.74,
                    "活跃率": 0.285,
                    "资金流入比": 1.06,
                    "人均余额": 2102.47
                }}
            }},
            "business_health_indicators": {{
                "liquidity_status": "良好",
                "growth_momentum": "积极",
                "user_engagement": "中等",
                "risk_level": "低"
            }},
            "data_summary": {{
                "data_sources": ["system_data", "daily_data"],
                "time_coverage": "当前时点",
                "metrics_count": 9,
                "completeness": 0.95
            }},
            "key_insights": [
                "资金净流入为正，显示健康的资金流",
                "用户活跃率28.5%，处于合理水平",
                "人均余额2,102元，用户价值较高"
            ],
            "recommendations": [
                "维持当前的运营策略",
                "继续监控资金流动性",
                "考虑提升用户活跃度"
            ],
            "data_quality_assessment": {{
                "overall_quality": 0.88,
                "completeness": 0.90,
                "consistency": 0.86,
                "reliability": 0.89
            }},
            "extraction_confidence": 0.87
        }}
        
        **分析要点**：
        - 涵盖所有可用的业务指标
        - 计算有意义的衍生指标
        - 提供业务健康度评估
        - 给出可操作的建议
        """

    async def _call_claude_extraction(self, prompt: str) -> Dict[str, Any]:
        """调用Claude进行提取"""
        try:
            result = await self.claude_client.generate_text(prompt, max_tokens=5000)

            if result.get('success'):
                response_text = result.get('text', '{}')

                # 🎯 解析Claude返回的JSON
                extracted_data = self._parse_claude_json_response(response_text)

                if extracted_data:
                    return {
                        'success': True,
                        'extracted_data': extracted_data,
                        'raw_response': response_text[:500]  # 保留部分原始响应用于调试
                    }
                else:
                    return {
                        'success': False,
                        'error': 'JSON解析失败',
                        'raw_response': response_text[:200]
                    }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Claude调用失败')
                }

        except Exception as e:
            logger.error(f"Claude提取调用异常: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _parse_claude_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """解析Claude的JSON响应"""
        try:
            # 直接解析
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            try:
                # 提取代码块中的JSON
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # 提取大括号中的内容
                brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if brace_match:
                    return json.loads(brace_match.group())

            except json.JSONDecodeError:
                pass

        logger.error(f"无法解析Claude响应为JSON: {response_text[:300]}")
        return None

    def _validate_and_enhance_result(self,
                                   claude_result: Dict[str, Any],
                                   semantic_data: Dict[str, Any],
                                   query_analysis) -> Dict[str, Any]:
        """验证和增强Claude的提取结果"""

        enhanced_result = claude_result.copy()

        # 🎯 添加提取方法标识
        enhanced_result['extraction_method'] = 'claude_intelligent'
        enhanced_result['extraction_timestamp'] = datetime.now().isoformat()

        # 🎯 验证关键字段存在性
        validation_result = self._validate_extraction_completeness(claude_result, semantic_data)
        enhanced_result['validation_result'] = validation_result

        # 🎯 添加原始数据引用
        enhanced_result['source_data_summary'] = {
            'semantic_keys': list(semantic_data.keys()),
            'data_sources': len(semantic_data)
        }

        return enhanced_result

    def _validate_extraction_completeness(self,
                                        extraction: Dict[str, Any],
                                        semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证提取结果的完整性"""

        validation = {
            'is_complete': True,
            'missing_elements': [],
            'quality_score': 1.0
        }

        # 检查必要字段
        required_fields = ['extraction_type', 'extraction_confidence']
        for field in required_fields:
            if field not in extraction:
                validation['missing_elements'].append(field)
                validation['quality_score'] -= 0.2

        # 检查数据完整性
        if 'extraction_type' in extraction:
            extraction_type = extraction['extraction_type']

            if extraction_type == 'comparison_analysis':
                if 'comparison_periods' not in extraction:
                    validation['missing_elements'].append('comparison_periods')
                    validation['quality_score'] -= 0.3
                if 'comparison_analysis' not in extraction:
                    validation['missing_elements'].append('comparison_analysis')
                    validation['quality_score'] -= 0.3

        validation['is_complete'] = len(validation['missing_elements']) == 0
        validation['quality_score'] = max(0.0, validation['quality_score'])

        return validation

    def _intelligent_fallback_extraction(self,
                                       semantic_data: Dict[str, Any],
                                       query_analysis) -> Dict[str, Any]:
        """智能降级提取"""
        logger.info("执行智能降级提取")

        try:
            # 🎯 分析语义数据的特征
            analysis = self._analyze_semantic_data_features(semantic_data)

            # 🎯 根据特征选择降级策略
            if analysis['has_comparison_data']:
                return self._fallback_comparison_extraction(semantic_data, analysis)
            elif analysis['has_time_series']:
                return self._fallback_time_series_extraction(semantic_data, analysis)
            else:
                return self._fallback_general_extraction(semantic_data, analysis)

        except Exception as e:
            logger.error(f"智能降级提取失败: {e}")
            return self._create_error_result(f"智能降级提取失败: {str(e)}")

    def _analyze_semantic_data_features(self, semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析语义数据的特征"""

        semantic_keys = list(semantic_data.keys())

        # 检查是否有对比数据
        has_current_week = any('current_week' in key for key in semantic_keys)
        has_last_week = any('last_week' in key for key in semantic_keys)
        has_today = any('today' in key for key in semantic_keys)
        has_yesterday = any('yesterday' in key for key in semantic_keys)

        has_comparison_data = (has_current_week and has_last_week) or (has_today and has_yesterday)

        # 检查是否有时间序列数据
        time_related_keys = [key for key in semantic_keys
                           if any(time_word in key for time_word in
                                ['week', 'day', 'date_', 'today', 'yesterday'])]

        has_time_series = len(time_related_keys) > 1

        return {
            'semantic_keys': semantic_keys,
            'has_comparison_data': has_comparison_data,
            'has_time_series': has_time_series,
            'time_related_keys': time_related_keys,
            'data_count': len(semantic_data)
        }

    def _fallback_comparison_extraction(self,
                                      semantic_data: Dict[str, Any],
                                      analysis: Dict[str, Any]) -> Dict[str, Any]:
        """降级对比提取"""

        # 找到对比的两个数据集
        current_data = None
        baseline_data = None
        current_key = ""
        baseline_key = ""

        for key, data_entry in semantic_data.items():
            if 'current_week' in key or 'today' in key:
                current_data = data_entry
                current_key = key
            elif 'last_week' in key or 'yesterday' in key:
                baseline_data = data_entry
                baseline_key = key

        if not current_data or not baseline_data:
            return self._create_error_result("无法找到对比数据集")

        # 提取对比指标
        current_metrics = self._extract_key_metrics(current_data['data'])
        baseline_metrics = self._extract_key_metrics(baseline_data['data'])

        # 计算变化
        comparison_analysis = {}
        for metric in current_metrics:
            if metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]

                if baseline_val != 0:
                    change_rate = (current_val - baseline_val) / baseline_val
                    comparison_analysis[metric] = {
                        'current_value': current_val,
                        'baseline_value': baseline_val,
                        'absolute_change': current_val - baseline_val,
                        'percentage_change': change_rate,
                        'change_direction': '增长' if change_rate > 0 else '下降' if change_rate < 0 else '持平',
                        'significance_level': self._assess_change_significance(abs(change_rate)),
                        'business_impact': self._assess_business_impact(metric, change_rate)
                    }

        # 生成洞察
        insights = self._generate_fallback_insights(comparison_analysis)

        return {
            'extraction_type': 'fallback_comparison',
            'comparison_periods': {
                'current_period': {
                    'semantic_key': current_key,
                    'key_metrics': current_metrics,
                    'period_name': self._generate_period_name(current_key)
                },
                'baseline_period': {
                    'semantic_key': baseline_key,
                    'key_metrics': baseline_metrics,
                    'period_name': self._generate_period_name(baseline_key)
                }
            },
            'comparison_analysis': comparison_analysis,
            'key_insights': insights,
            'extraction_method': 'intelligent_fallback',
            'extraction_confidence': 0.75,
            'data_quality_assessment': {
                'overall_quality': 0.7,
                'method': 'fallback',
                'completeness': len(comparison_analysis) / max(len(current_metrics), len(baseline_metrics))
            }
        }

    def _fallback_time_series_extraction(self,
                                       semantic_data: Dict[str, Any],
                                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """降级时间序列提取"""

        time_series_data = []

        # 按时间顺序整理数据
        for key in sorted(analysis['time_related_keys']):
            if key in semantic_data:
                data_entry = semantic_data[key]
                metrics = self._extract_key_metrics(data_entry['data'])

                time_series_data.append({
                    'period': self._generate_period_name(key),
                    'semantic_key': key,
                    'key_metrics': metrics
                })

        # 基础趋势分析
        trend_analysis = {}
        if len(time_series_data) >= 2:
            first_metrics = time_series_data[0]['key_metrics']
            last_metrics = time_series_data[-1]['key_metrics']

            for metric in first_metrics:
                if metric in last_metrics:
                    first_val = first_metrics[metric]
                    last_val = last_metrics[metric]

                    if first_val != 0:
                        growth_rate = (last_val - first_val) / first_val
                        trend_analysis[metric] = {
                            'trend_direction': '上升' if growth_rate > 0 else '下降' if growth_rate < 0 else '持平',
                            'growth_rate': growth_rate,
                            'trend_strength': self._assess_trend_strength(growth_rate),
                            'volatility': 'N/A'  # 需要更多数据点才能计算
                        }

        insights = [
            f"时间序列包含{len(time_series_data)}个数据点",
            "基于首末数据点进行趋势分析"
        ]

        for metric, trend in trend_analysis.items():
            insights.append(f"{metric}{trend['trend_direction']}趋势，增长率{trend['growth_rate']:.1%}")

        return {
            'extraction_type': 'fallback_time_series',
            'time_series_data': {
                'data_points': time_series_data,
                'period_type': 'mixed',
                'total_periods': len(time_series_data)
            },
            'trend_analysis': trend_analysis,
            'key_insights': insights,
            'extraction_method': 'intelligent_fallback',
            'extraction_confidence': 0.65,
            'data_quality_assessment': {
                'overall_quality': 0.6,
                'method': 'fallback_time_series',
                'data_points': len(time_series_data)
            }
        }

    def _fallback_general_extraction(self,
                                   semantic_data: Dict[str, Any],
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """降级通用提取"""

        # 合并所有可用数据
        all_metrics = {}
        data_sources = []

        for key, data_entry in semantic_data.items():
            metrics = self._extract_key_metrics(data_entry['data'])

            # 避免重复指标，优先使用最新/最相关的数据
            for metric, value in metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = value

            data_sources.append(key)

        # 计算衍生指标
        derived_metrics = {}
        if '入金' in all_metrics and '出金' in all_metrics:
            derived_metrics['净流入'] = all_metrics['入金'] - all_metrics['出金']
            derived_metrics['资金流入比'] = all_metrics['入金'] / all_metrics['出金'] if all_metrics['出金'] > 0 else 0

        if '活跃用户数' in all_metrics and '总用户数' in all_metrics:
            derived_metrics['活跃率'] = all_metrics['活跃用户数'] / all_metrics['总用户数'] if all_metrics['总用户数'] > 0 else 0

        # 生成基础洞察
        insights = []
        if '净流入' in derived_metrics:
            net_flow = derived_metrics['净流入']
            if net_flow > 0:
                insights.append(f"净流入为正({net_flow:,.2f})，资金状况健康")
            else:
                insights.append(f"净流入为负({net_flow:,.2f})，需关注资金流动")

        if '活跃率' in derived_metrics:
            activity_rate = derived_metrics['活跃率']
            insights.append(f"用户活跃率{activity_rate:.1%}，{'处于良好水平' if activity_rate > 0.3 else '有提升空间'}")

        insights.append(f"数据来源包括{len(data_sources)}个数据集")

        return {
            'extraction_type': 'fallback_general',
            'extracted_metrics': {
                'primary_metrics': all_metrics,
                'derived_metrics': derived_metrics
            },
            'data_summary': {
                'data_sources': data_sources,
                'metrics_count': len(all_metrics) + len(derived_metrics),
                'time_coverage': '混合时间点'
            },
            'key_insights': insights,
            'extraction_method': 'intelligent_fallback',
            'extraction_confidence': 0.70,
            'data_quality_assessment': {
                'overall_quality': 0.65,
                'method': 'fallback_general',
                'completeness': min(1.0, len(all_metrics) / 8)  # 假设8个是理想指标数量
            }
        }

    def _extract_key_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取关键指标"""
        metrics = {}

        # 聚合多天数据（如果是列表）
        if isinstance(data, list):
            # 累加所有天的数据
            aggregated = {}
            for day_data in data:
                if isinstance(day_data, dict):
                    for key, value in day_data.items():
                        if isinstance(value, (int, float)):
                            aggregated[key] = aggregated.get(key, 0) + value
            data = aggregated

        # 提取数值字段
        key_fields = ['入金', '出金', '注册人数', '购买产品数量', '到期产品数量', '持仓人数', '总余额', '总入金', '总出金', '活跃用户数', '总用户数']

        for key, value in data.items():
            if key in key_fields and isinstance(value, (int, float)):
                metrics[key] = float(value)

        # 处理嵌套的用户统计数据
        if '用户统计' in data and isinstance(data['用户统计'], dict):
            user_stats = data['用户统计']
            for field in ['总用户数', '活跃用户数']:
                if field in user_stats:
                    try:
                        metrics[field] = float(user_stats[field])
                    except (ValueError, TypeError):
                        pass

        # 计算基础衍生指标
        if '入金' in metrics and '出金' in metrics:
            metrics['净流入'] = metrics['入金'] - metrics['出金']

        return metrics

    def _assess_change_significance(self, change_rate: float) -> str:
        """评估变化显著性"""
        if change_rate > 0.5:
            return "极显著"
        elif change_rate > 0.2:
            return "非常显著"
        elif change_rate > 0.1:
            return "显著"
        elif change_rate > 0.05:
            return "中等"
        elif change_rate > 0.01:
            return "轻微"
        else:
            return "微弱"

    def _assess_business_impact(self, metric: str, change_rate: float) -> str:
        """评估业务影响"""
        impact_direction = "正面" if change_rate > 0 else "负面"

        if metric in ['入金', '注册人数', '购买产品数量']:
            return "正面" if change_rate > 0 else "需关注"
        elif metric in ['出金']:
            return "需关注" if change_rate > 0 else "正面"
        else:
            return impact_direction

    def _assess_trend_strength(self, growth_rate: float) -> str:
        """评估趋势强度"""
        abs_rate = abs(growth_rate)
        if abs_rate > 0.3:
            return "强"
        elif abs_rate > 0.1:
            return "中等"
        elif abs_rate > 0.02:
            return "轻微"
        else:
            return "微弱"

    def _generate_period_name(self, semantic_key: str) -> str:
        """根据语义键生成期间名称"""
        if 'current_week' in semantic_key:
            return "本周"
        elif 'last_week' in semantic_key:
            return "上周"
        elif 'today' in semantic_key:
            return "今天"
        elif 'yesterday' in semantic_key:
            return "昨天"
        elif 'date_' in semantic_key:
            # 提取日期
            parts = semantic_key.split('_')
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    try:
                        date_obj = datetime.strptime(part, '%Y%m%d')
                        return date_obj.strftime('%m月%d日')
                    except:
                        pass
            return f"特定日期({semantic_key})"
        else:
            return semantic_key.replace('_', ' ')

    def _generate_fallback_insights(self, comparison_analysis: Dict[str, Any]) -> List[str]:
        """生成降级分析的洞察"""
        insights = []

        for metric, analysis in comparison_analysis.items():
            change_rate = analysis.get('percentage_change', 0)
            direction = analysis.get('change_direction', '持平')

            if abs(change_rate) > 0.01:  # 变化超过1%才报告
                insights.append(f"{metric}{direction}{abs(change_rate):.1%}，{analysis.get('significance_level', '中等')}变化")

        if not insights:
            insights.append("各指标变化较小，总体保持稳定")

        return insights

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'extraction_type': 'error',
            'extraction_method': 'error_fallback',
            'success': False,
            'error': error_message,
            'extraction_confidence': 0.0,
            'extraction_timestamp': datetime.now().isoformat()
        }

    def _load_extraction_templates(self) -> Dict[str, str]:
        """加载提取模板（未来可以从配置文件加载）"""
        return {
            'comparison': '对比分析模板',
            'trend': '趋势分析模板',
            'calculation': '计算分析模板',
            'general': '通用分析模板'
        }