# core/prompts/prompt_manager.py
from typing import Dict, Any, Optional
from datetime import datetime
import json
from core.detectors.query_type_detector import QueryType, QueryTypeResult


class PromptManager:
    """统一提示词管理器"""

    def __init__(self):
        self.current_date = datetime.now()
        self.base_templates = self._load_base_templates()
        self.enhancement_templates = self._load_enhancement_templates()

    def build_quick_judgment_prompt(self, user_query: str,
                                    query_type_result: Optional[QueryTypeResult] = None) -> str:
        """构建快速判断prompt - 提取自编排器"""

        base_prompt = f"""
        作为智能助手，请快速判断这个查询是否可以通过简单的API调用直接回答：

        查询: "{user_query}"
        当前日期: {self.current_date.strftime('%Y年%m月%d日')}

        判断标准:
        1. 单一数据查询（如"总资金多少"、"今天入金"）
        2. 特定日期数据查询（如"昨天入金数据"）
        3. 简单状态查询（如"活跃用户数"）

        需要复杂处理的:
        1. 多步骤计算（如复投分析）
        2. 趋势分析
        3. 预测计算
        4. 对比分析

        请返回JSON格式:
        {{
            "is_quick": true/false,
            "confidence": 0.9,
            "reason": "判断理由",
            "suggested_api": "get_system_data",
            "date_mentioned": "20250531"
        }}
        """

        # 如果检测到特殊查询类型，增强判断逻辑
        if query_type_result and query_type_result.type != QueryType.SIMPLE_DATA:
            enhancement = self._get_quick_judgment_enhancement(query_type_result.type)
            base_prompt += f"\n\n{enhancement}"

        return base_prompt

    def build_intent_analysis_prompt(self, user_query: str,
                                     query_type_result: Optional[QueryTypeResult] = None,
                                     quick_decision: Optional[Dict[str, Any]] = None) -> str:
        """构建意图分析prompt - 提取自编排器"""

        # 基础prompt（从编排器中提取）
        base_prompt = f"""
        作为金融AI专家，深度分析用户查询的意图和需求：

        查询: "{user_query}"
        当前时间: {self.current_date.strftime('%Y年%m月%d日')}

        系统可用的API接口:
        1. get_system_data() - 获取系统总体数据，包括总余额、总入金、总出金等
        2. get_daily_data(date) - 获取指定日期的数据，包括日期、注册人数、持仓人数、购买产品数量、到期产品数量、入金、出金
        3. get_product_data() - 获取当前产品数据
        4. get_product_end_data(date) - 获取指定日期到期的产品数据
        5. get_product_end_interval(start_date, end_date) - 获取指定日期范围内到期的产品数据
        6. get_user_daily_data(date) - 获取指定日期的用户数据
        7. get_user_data(page) - 获取用户列表数据

        请详细分析：
        1. 用户的核心意图是什么？
        2. 查询的复杂程度如何？(simple, medium, complex)
        3. 需要什么类型的数据？
        4. 是否需要计算分析？
        5. 建议的API调用策略(具体到API名称和参数)

        返回JSON格式：
        {{
            "intent": "获取余额信息/比较数据/趋势分析/复投计算/...",
            "complexity": "simple/medium/complex",
            "data_needed": ["日期数据", "产品数据", "用户数据", ...],
            "calculation_required": true/false,
            "api_strategy": [
                {{
                    "method": "get_daily_data",
                    "params": {{ "date": "20250531" }},
                    "reason": "获取昨天的数据"
                }}
            ]
        }}
        """

        # 根据查询类型添加专门的增强逻辑
        if query_type_result and query_type_result.type != QueryType.SIMPLE_DATA:
            enhancement = self._get_intent_analysis_enhancement(query_type_result)
            base_prompt += f"\n\n{enhancement}"

        # 添加快速决策上下文
        if quick_decision:
            context = self._build_quick_decision_context(quick_decision)
            base_prompt += f"\n\n{context}"

        return base_prompt

    def build_date_recognition_prompt(self, user_query: str,
                                      query_type_result: Optional[QueryTypeResult] = None) -> str:
        """构建日期识别prompt - 提取自编排器并增强"""

        current_year = self.current_date.year
        base_prompt = f"""
        作为金融AI专家，识别并转换查询中的日期表达：

        查询: "{user_query}"
        当前时间: {current_year}年6月1日

        请识别所有日期表达，并转换为标准格式(YYYYMMDD)：
        - 昨天/昨日 = 2025年5月31日 = 20250531
        - 前天 = 2025年5月30日 = 20250530
        - X月X日 = 转换为{current_year}年的对应日期
        - 本周 = 当前周的日期范围 (20250526-20250601)
        - 上周 = 上一周的日期范围 (20250519-20250525)
        - 本月 = 当前月的日期范围 (20250601-20250630)
        - 上月 = 上个月的日期范围 (20250501-20250531)

        返回JSON格式：
        {{
            "has_dates": true,
            "date_expressions": ["昨天"],
            "converted_dates": {{
                "昨天": "20250531"
            }},
            "date_type": "single/range",
            "api_dates": {{
                "start_date": "20250531",
                "end_date": null
            }},
            "period_type": null,
            "confidence": 0.95
        }}
        """

        # 🆕 根据查询类型添加特殊的日期识别逻辑
        if query_type_result:
            if query_type_result.type == QueryType.COMPARISON:
                base_prompt += self._get_comparison_date_enhancement()
            elif query_type_result.type == QueryType.HISTORICAL_REVIEW:
                base_prompt += self._get_historical_date_enhancement()
            elif query_type_result.type == QueryType.REINVESTMENT:
                base_prompt += self._get_reinvestment_date_enhancement()

        return base_prompt

    def build_response_generation_prompt(self, user_query: str,
                                         query_analysis: Any,
                                         extracted_data: Dict[str, Any],
                                         calculation_results: Dict[str, Any],
                                         query_type_result: Optional[QueryTypeResult] = None) -> str:
        """构建响应生成prompt - 通用智能数据展示完整版"""

        # 🔧 修复：首先定义response_data
        response_data = {
            'extracted_metrics': extracted_data.get('extracted_metrics', {}),
            'derived_metrics': extracted_data.get('derived_metrics', {}),
            'key_insights': extracted_data.get('key_insights', []),
            'business_health_indicators': extracted_data.get('business_health_indicators', {}),
            'detailed_daily_analysis': extracted_data.get('detailed_daily_analysis', {}),
            'weekly_pattern_analysis': extracted_data.get('weekly_pattern_analysis', {}),
            'recommendations': extracted_data.get('recommendations', []),
            'data_quality_assessment': extracted_data.get('data_quality_assessment', {}),
            'direct_answer': extracted_data.get('direct_answer', ''),
            'time_series_data': extracted_data.get('time_series_extraction', {}),
            'comparison_analysis': extracted_data.get('comparison_analysis', {}),
            'calculation_summary': self._summarize_calculation_results(calculation_results),
            'data_sources': extracted_data.get('source_data_summary', {}),
            'extraction_method': extracted_data.get('extraction_method', 'unknown'),
            'business_insights': extracted_data.get('business_insights', []),
            'detailed_calculations': extracted_data.get('detailed_calculations', {}),
            'raw_api_results': extracted_data.get('raw_api_results', {}),  # 🆕 通用原始数据
            'multi_data_detected': extracted_data.get('multi_data_detected', False),  # 🆕 多数据标记
            'data_count': extracted_data.get('data_count', 0),  # 🆕 数据条数
            'raw_data_details': extracted_data.get('raw_data_details', {})  # 🆕 详细原始数据
        }

        # 检查是否有直接答案和关键洞察
        direct_answer = extracted_data.get('direct_answer', '')
        key_insights = extracted_data.get('key_insights', [])
        multi_data_detected = extracted_data.get('multi_data_detected', False)
        data_count = extracted_data.get('data_count', 0)

        base_prompt = f"""
        作为专业的金融AI助手，基于已完成的数据分析生成详细、透明、专业的分析回答：

        用户查询: "{user_query}"
        查询意图: {getattr(query_analysis, 'intent', '数据查询')}

        🎯 **重要提醒：以下数据已经完成分析和计算，请直接使用！**

        ✅ **直接答案（已计算完成）**: {direct_answer}

        ✅ **关键洞察（已分析完成）**: 
        {chr(10).join(f"- {insight}" for insight in key_insights)}

        ✅ **多数据检测结果**: {"检测到多条数据" if multi_data_detected else "单条数据"}，共{data_count}条数据

        ✅ **完整分析数据**:
        {json.dumps(response_data, ensure_ascii=False, indent=2)}

        **🎯 智能数据展示规则（自动适配）**：

        ### 📊 数据展示判断标准
        1. **单条数据查询**（如"今天入金多少"）→ 简洁展示关键指标
        2. **多条数据查询**（如"5月份每日数据"、"所有产品信息"）→ **必须展示详细列表/表格**
        3. **聚合计算查询**（如"平均值"、"总计"）→ 展示计算过程 + 详细数据源
        4. **对比分析查询**（如"本周vs上周"）→ 展示对比数据明细

        ### 🔍 自动识别多数据场景规则
        **如果满足以下任一条件，必须展示详细数据表格**：
        - 数据条数 > 1 条
        - 查询涉及时间范围（天/周/月）
        - 包含"每日"、"所有"、"详细"等关键词
        - 计算基于多个数据点
        - raw_api_results中包含多个API调用结果

        ### 📋 详细数据展示格式标准

        **🗓️ 时间序列数据**（如多日数据）：
        ```
        ## 📅 [时间范围] 详细数据明细

        | 日期 | 入金金额(元) | 出金金额(元) | 净流入(元) | 注册人数 | 购买数量 | 备注 |
        |------|-------------|-------------|-----------|----------|----------|------|
        | [逐行展示每个数据点，不能省略任何一行] |
        | 2025-05-01 | 12,345.67 | 8,765.43 | 3,580.24 | 5 | 12 | 正常 |
        | 2025-05-02 | 15,678.90 | 9,876.54 | 5,802.36 | 8 | 15 | 正常 |
        | ... [必须显示所有数据，不能用省略号] ... |
        | **合计/平均** | **XXX,XXX.XX** | **XXX,XXX.XX** | **XXX,XXX.XX** | **XXX** | **XXX** | **统计** |
        ```

        **🛍️ 产品/项目类数据**：
        ```
        ## 🛍️ 产品详细信息列表

        | 产品名称 | 投资金额(元) | 持有数量 | 到期时间 | 收益率 | 状态 |
        |----------|-------------|----------|----------|--------|------|
        | [逐个展示每个产品，不能省略] |
        | 产品A | 50,000.00 | 100 | 2025-06-15 | 8.5% | 持有中 |
        | 产品B | 30,000.00 | 60 | 2025-07-01 | 7.2% | 持有中 |
        | ... [显示所有产品] ... |
        | **合计** | **XXX,XXX.XX** | **XXX** | **-** | **平均X.X%** | **-** |
        ```

        **👥 用户/分类数据**：
        ```
        ## 👥 用户详细分析

        | VIP等级 | 用户数量 | 占比 | 平均投资额(元) | 活跃度 |
        |---------|----------|------|---------------|--------|
        | [逐级展示详细信息] |
        | VIP0 | 1,234 | 45.2% | 5,678.90 | 72% |
        | VIP1 | 567 | 20.8% | 12,345.67 | 85% |
        | ... [显示所有级别] ... |
        | **总计** | **X,XXX** | **100%** | **平均XX,XXX.XX** | **平均XX%** |
        ```

        ### 🧮 通用计算展示要求
        **如果涉及计算，必须包含以下部分**：

        #### 1. 🎯 核心结论
        - 直接回答用户问题，数字要加粗突出

        #### 2. 📊 详细数据来源
        - 展示用于计算的所有原始数据
        - 如果是多天数据，必须显示每日明细表格
        - 如果是多项目数据，必须显示项目列表

        #### 3. 🧮 计算过程透明化
        ```
        **计算公式：**
        具体计算公式 = 数据1 + 数据2 + ... + 数据N

        **计算步骤：**
        步骤1: 数据汇总 = 具体数值1 + 具体数值2 + ... = 中间结果
        步骤2: 最终计算 = 中间结果 ÷ 数据点数量 = 最终结果

        **数据验证：**
        ✅ 数据完整性: X/Y 数据点完整
        ✅ 数据源可靠性: 所有数据来自官方API
        ✅ 计算准确性: 精确到小数点后2位
        ```

        #### 4. 📋 数据来源说明
        - 数据获取方法: [提取方法]
        - 数据源数量: [X个独立数据源]
        - 时间覆盖: [具体时间范围]
        - 数据更新: [最新数据状态]

        ### 💡 智能适配原则
        - **数据量少（1-3条）** → 直接在正文中展示，配合简单表格
        - **数据量中等（4-20条）** → 使用完整表格展示，包含汇总行
        - **数据量大（20+条）** → 完整表格+重点数据突出+统计摘要
        - **包含时间序列** → 按时间顺序排列，标注工作日/周末
        - **包含分类数据** → 按重要性或逻辑顺序排列

        ### 🔢 数值展示标准
        - **金额格式**: 使用千分位逗号，保留2位小数 (12,345.67)
        - **百分比格式**: 保留1-2位小数 (8.5%, 15.67%)
        - **大数字单位**: 超过万的使用万元单位，超过千万的使用千万单位
        - **表格对齐**: 数字右对齐，文字左对齐
        - **突出显示**: 重要数字使用**粗体**标记

        **🎯 核心要求：**
        1. **优先使用direct_answer中的计算结果** - 这是已经完成的精确计算
        2. **如果检测到多数据，必须展示完整的数据明细表格**
        3. **充分利用key_insights进行业务分析**
        4. **不要重新计算** - 所有计算已完成，直接展示结果
        5. **格式要专业详细** - 参考成功报告的格式标准
        6. **表格要完整** - 不能省略任何数据行，必须有合计/统计行

        请基于以上规则，智能判断数据类型并生成包含完整详细数据的专业报告！
        """

        # 🆕 根据查询类型添加特定格式要求
        if calculation_results.get('needs_calculation'):
            base_prompt += self._get_calculation_specific_requirements(calculation_results)

        if query_type_result and query_type_result.type in [QueryType.AGGREGATION, QueryType.COMPARISON]:
            base_prompt += self._get_complex_analysis_requirements()

        return base_prompt

    def _get_calculation_specific_requirements(self, calculation_results: Dict[str, Any]) -> str:
        """获取计算类查询的特定格式要求"""
        return """

        **🧮 计算类查询特殊要求**：

        1. **数据源追溯**：
           - 明确列出所有使用的数据点
           - 标注每个数据的获取时间和来源
           - 展示原始数据→处理数据→最终结果的完整链条

        2. **计算过程完全透明**：
           - 每一步计算都要显示公式
           - 每一步都要显示具体数值代入
           - 中间结果和最终结果都要标注

        3. **数据质量保证**：
           - 数据完整性检查（X/Y天数据完整）
           - 异常值检测结果
           - 计算精度说明

        4. **结果验证**：
           - 交叉验证计算
           - 合理性检查
           - 置信度评估
        """

    def _get_complex_analysis_requirements(self) -> str:
        """获取复杂分析的格式要求"""
        return """

        **📊 复杂分析特殊要求**：

        1. **多维度数据展示**：
           - 时间维度分析
           - 对比维度分析
           - 趋势维度分析

        2. **可视化数据表格**：
           - 使用markdown表格展示关键数据
           - 数据要有清晰的分类和标签
           - 包含汇总行和平均值

        3. **洞察深度要求**：
           - 不仅要有数据，还要有分析
           - 要指出数据背后的业务含义
           - 要给出可操作的建议
        """

    def _summarize_calculation_results(self, calculation_results: Dict[str, Any]) -> Dict[str, Any]:
        """总结计算结果"""
        if not calculation_results.get('success'):
            return {'has_calculation': False}

        return {
            'has_calculation': True,
            'calculation_type': calculation_results.get('calculation_type'),
            'success': True,
            'confidence': calculation_results.get('confidence', 0.0)
        }

    def _load_base_templates(self) -> Dict[str, str]:
        """加载基础模板"""
        return {
            'quick_judgment': "快速判断基础模板",
            'intent_analysis': "意图分析基础模板",
            'date_recognition': "日期识别基础模板",
            'response_generation': "响应生成基础模板"
        }

    def _load_enhancement_templates(self) -> Dict[QueryType, Dict[str, str]]:
        """加载增强模板"""
        return {
            QueryType.REINVESTMENT: {
                'quick_judgment': """
                **复投查询特别判断**：
                - 涉及复投计算的查询需要复杂处理
                - 需要到期产品数据 + 复投比例计算
                - 通常不是快速响应类型
                """,

                'intent_analysis': """
                **复投分析特别要求**：
                1. 必须获取产品到期数据（使用get_product_end_interval）
                2. 需要系统余额数据（使用get_system_data）
                3. 计算类型设置为：reinvestment_analysis
                4. 提取复投比例参数
                5. 计算剩余可提现资金

                复投计算公式：
                - 复投金额 = 到期总金额 × 复投比例
                - 剩余资金 = 到期总金额 - 复投金额
                - 最终余额 = 当前余额 + 剩余资金
                """,

                'response_format': """
                **复投分析专业格式**：
                📊 **复投计算详细分析**

                🔍 **基础数据**：
                - 查询期间：X月X日至X月X日
                - 到期产品总金额：¥XXX
                - 设定复投比例：XX%
                - 当前系统余额：¥XXX

                🧮 **计算过程**：
                1. 到期总金额：¥XXX
                2. 复投金额 = ¥XXX × XX% = ¥XXX
                3. 提现金额 = ¥XXX × XX% = ¥XXX  
                4. 最终可用余额 = ¥XXX + ¥XXX = ¥XXX

                💡 **业务建议**：
                - 现金流影响分析
                - 复投收益预期
                - 风险提示
                """
            },

            QueryType.PREDICTION: {
                'intent_analysis': """
                **预测分析特别要求**：
                1. 需要历史数据支撑（get_daily_data多个日期）
                2. 计算类型：cash_runway 或 trend_prediction
                3. 分析资金消耗速度和趋势
                4. 考虑业务季节性和波动性
                """,

                'response_format': """
                **预测分析专业格式**：
                📈 **资金跑道预测分析**

                📊 **历史数据基础**：
                - 分析周期：最近X天
                - 平均日出金：¥XXX
                - 平均日入金：¥XXX
                - 净流出速度：¥XXX/天

                🔮 **预测结果**：
                - 按当前趋势：还能维持X天
                - 乐观情景：还能维持X天
                - 悲观情景：还能维持X天

                ⚠️ **风险提示**：
                - 预测假设条件
                - 关键风险因素
                - 建议监控指标
                """
            },

            QueryType.COMPARISON: {
                'date_recognition': """
                **比较查询日期识别增强**：
                如果是比较查询，返回comparison格式：
                {{
                    "period_type": "week",
                    "period_value": "comparison",
                    "comparison_periods": {{
                        "current": {{
                            "period": "本周",
                            "start_date": "20250526", 
                            "end_date": "20250601"
                        }},
                        "previous": {{
                            "period": "上周",
                            "start_date": "20250519",
                            "end_date": "20250525"
                        }}
                    }}
                }}
                """,

                'response_format': """
                **对比分析专业格式**：
                📊 **周度对比分析报告**

                📈 **每日明细数据**：
                显示本周和上周的每日数据明细

                🧮 **计算过程**：
                展示完整的变化率计算步骤

                📋 **对比结果**：
                - 绝对变化：±¥XXX
                - 相对变化：±XX%
                - 变化趋势：上升/下降/稳定
                """
            }
        }

    def _get_quick_judgment_enhancement(self, query_type: QueryType) -> str:
        """获取快速判断增强"""
        return self.enhancement_templates.get(query_type, {}).get('quick_judgment', '')

    def _get_intent_analysis_enhancement(self, query_type_result: QueryTypeResult) -> str:
        """获取意图分析增强"""
        enhancement = self.enhancement_templates.get(query_type_result.type, {}).get('intent_analysis', '')

        # 添加特殊要求参数
        if query_type_result.special_requirements:
            enhancement += f"\n\n**检测到的特殊参数**：\n"
            for key, value in query_type_result.special_requirements.items():
                enhancement += f"- {key}: {value}\n"

        return enhancement

    def _get_comparison_date_enhancement(self) -> str:
        """获取比较查询的日期识别增强"""
        return self.enhancement_templates.get(QueryType.COMPARISON, {}).get('date_recognition', '')

    def _get_historical_date_enhancement(self) -> str:
        """获取历史回顾的日期识别增强"""
        return """
        **历史回顾日期增强**：
        需要识别时间范围，如"最近30天"、"上个月"等
        返回足够的历史数据范围支持分析
        """

    def _get_reinvestment_date_enhancement(self) -> str:
        """获取复投查询的日期识别增强"""
        return """
        **复投查询日期增强**：
        重点识别到期时间范围，确保获取准确的产品到期数据
        """

    def _get_response_enhancement(self, query_type: QueryType) -> str:
        """获取响应生成增强"""
        return self.enhancement_templates.get(query_type, {}).get('response_format', '')

    def _build_quick_decision_context(self, quick_decision: Dict[str, Any]) -> str:
        """构建快速决策上下文"""
        if not quick_decision:
            return ""

        context = "**快速判断结果**：\n"
        context += f"该查询被判断为{'快速响应' if quick_decision.get('is_quick_response') else '复杂查询'}\n"
        context += f"原因：{quick_decision.get('reason', '未知')}\n"

        return context

    def _summarize_calculation_results(self, calculation_results: Dict[str, Any]) -> Dict[str, Any]:
        """总结计算结果"""
        if not calculation_results.get('success'):
            return {'has_calculation': False}

        return {
            'has_calculation': True,
            'calculation_type': calculation_results.get('calculation_type'),
            'success': True,
            'confidence': calculation_results.get('confidence', 0.0)
        }


# 工厂函数
def create_prompt_manager() -> PromptManager:
    """创建提示词管理器实例"""
    return PromptManager()
