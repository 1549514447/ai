# utils/helpers/validation_utils.py
"""
🛡️ AI驱动的智能数据验证器
专为金融AI分析系统设计，提供全面的数据验证和安全检查

核心特点:
- AI驱动的业务逻辑验证
- 多层次数据完整性检查
- 智能异常检测和修复建议
- 金融数据特定的验证规则
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, date
import re
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"  # 基础验证
    STANDARD = "standard"  # 标准验证
    STRICT = "strict"  # 严格验证
    AI_ENHANCED = "ai_enhanced"  # AI增强验证


class ValidationSeverity(Enum):
    """验证问题严重程度"""
    INFO = "info"  # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"  # 错误
    CRITICAL = "critical"  # 严重


@dataclass
class ValidationIssue:
    """验证问题数据类"""
    field: str  # 字段名
    issue_type: str  # 问题类型
    severity: ValidationSeverity  # 严重程度
    message: str  # 问题描述
    current_value: Any  # 当前值
    suggested_value: Any = None  # 建议值
    fix_suggestion: str = ""  # 修复建议


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool  # 是否通过验证
    overall_score: float  # 总体评分 (0-1)
    issues: List[ValidationIssue]  # 问题列表
    warnings_count: int  # 警告数量
    errors_count: int  # 错误数量
    validation_level: ValidationLevel  # 验证级别
    validation_timestamp: str  # 验证时间
    metadata: Dict[str, Any]  # 额外信息


class ValidationUtils:
    """
    🛡️ AI驱动的智能数据验证器

    功能特点:
    1. 多层次验证策略 (基础→标准→严格→AI增强)
    2. 金融业务逻辑验证
    3. 智能异常检测和修复建议
    4. API响应数据验证
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化验证工具

        Args:
            claude_client: Claude客户端，用于业务逻辑验证
            gpt_client: GPT客户端，用于数据一致性检查
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client

        # 金融数据合理性范围
        self.financial_ranges = {
            "balance_min": 0,
            "balance_max": 1000000000,  # 10亿
            "user_count_min": 0,
            "user_count_max": 10000000,  # 1千万
            "daily_amount_min": 0,
            "daily_amount_max": 50000000,  # 5千万/天
            "rate_min": -1.0,  # -100%
            "rate_max": 10.0  # 1000%
        }

        logger.info("ValidationUtils initialized with AI capabilities")

    # ============= 核心验证方法 =============

    async def validate_data(self, data: Any, data_type: str,
                            validation_level: ValidationLevel = ValidationLevel.STANDARD,
                            context: Dict[str, Any] = None) -> ValidationResult:
        """
        通用数据验证入口

        Args:
            data: 待验证数据
            data_type: 数据类型 (api_response/financial_data/query_data等)
            validation_level: 验证级别
            context: 验证上下文

        Returns:
            ValidationResult: 验证结果
        """
        try:
            logger.info(f"🔍 开始数据验证: 类型={data_type}, 级别={validation_level.value}")

            issues = []

            # 基础验证 (所有级别都执行)
            basic_issues = self._basic_validation(data, data_type)
            issues.extend(basic_issues)

            # 标准验证
            if validation_level.value in ["standard", "strict", "ai_enhanced"]:
                standard_issues = self._standard_validation(data, data_type, context)
                issues.extend(standard_issues)

            # 严格验证
            if validation_level.value in ["strict", "ai_enhanced"]:
                strict_issues = await self._strict_validation(data, data_type, context)
                issues.extend(strict_issues)

            # AI增强验证
            if validation_level == ValidationLevel.AI_ENHANCED:
                ai_issues = await self._ai_enhanced_validation(data, data_type, context)
                issues.extend(ai_issues)

            # 生成验证结果
            return self._create_validation_result(issues, validation_level)

        except Exception as e:
            logger.error(f"❌ 数据验证失败: {str(e)}")
            return self._create_error_validation_result(str(e), validation_level)

    # ============= 基础验证层 =============

    def _basic_validation(self, data: Any, data_type: str) -> List[ValidationIssue]:
        """基础数据验证"""
        issues = []

        # 空值检查
        if data is None:
            issues.append(ValidationIssue(
                field="root",
                issue_type="null_data",
                severity=ValidationSeverity.CRITICAL,
                message="数据为空",
                current_value=None,
                fix_suggestion="检查数据源是否正常"
            ))
            return issues

        # 数据类型特定验证
        if data_type == "api_response":
            issues.extend(self._validate_api_response_basic(data))
        elif data_type == "financial_data":
            issues.extend(self._validate_financial_data_basic(data))
        elif data_type == "query_data":
            issues.extend(self._validate_query_data_basic(data))
        elif data_type == "date_data":
            issues.extend(self._validate_date_data_basic(data))

        return issues

    def _validate_api_response_basic(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """API响应基础验证"""
        issues = []

        # 检查必需字段
        required_fields = ["result", "status"]
        for field in required_fields:
            if field not in data:
                issues.append(ValidationIssue(
                    field=field,
                    issue_type="missing_field",
                    severity=ValidationSeverity.ERROR,
                    message=f"缺少必需字段: {field}",
                    current_value=None,
                    fix_suggestion=f"确保API返回包含{field}字段"
                ))

        # 检查result字段类型
        if "result" in data and not isinstance(data["result"], bool):
            issues.append(ValidationIssue(
                field="result",
                issue_type="invalid_type",
                severity=ValidationSeverity.ERROR,
                message="result字段应为布尔类型",
                current_value=data["result"],
                suggested_value=bool(data["result"]),
                fix_suggestion="检查API返回格式"
            ))

        # 检查status字段
        if "status" in data and not isinstance(data["status"], (int, str)):
            issues.append(ValidationIssue(
                field="status",
                issue_type="invalid_type",
                severity=ValidationSeverity.WARNING,
                message="status字段类型异常",
                current_value=data["status"],
                fix_suggestion="status应为数字或字符串"
            ))

        # 检查成功响应是否包含data字段
        if data.get("result") is True and "data" not in data:
            issues.append(ValidationIssue(
                field="data",
                issue_type="missing_data",
                severity=ValidationSeverity.ERROR,
                message="成功响应缺少data字段",
                current_value=None,
                fix_suggestion="成功的API响应应包含data字段"
            ))

        return issues

    def _validate_financial_data_basic(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """金融数据基础验证"""
        issues = []

        # 数值字段验证
        numeric_fields = ["总余额", "总入金", "总出金", "总投资金额", "总奖励发放"]

        for field in numeric_fields:
            if field in data:
                value = data[field]

                # 检查是否为数值
                try:
                    numeric_value = float(value) if isinstance(value, str) else value

                    # 检查是否为负数 (余额类不应为负)
                    if field in ["总余额", "总投资金额"] and numeric_value < 0:
                        issues.append(ValidationIssue(
                            field=field,
                            issue_type="negative_value",
                            severity=ValidationSeverity.ERROR,
                            message=f"{field}不应为负数",
                            current_value=numeric_value,
                            suggested_value=abs(numeric_value),
                            fix_suggestion="检查数据计算逻辑"
                        ))

                    # 检查异常大的数值
                    if numeric_value > self.financial_ranges["balance_max"]:
                        issues.append(ValidationIssue(
                            field=field,
                            issue_type="excessive_value",
                            severity=ValidationSeverity.WARNING,
                            message=f"{field}数值异常大: {numeric_value}",
                            current_value=numeric_value,
                            fix_suggestion="验证数据准确性"
                        ))

                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        field=field,
                        issue_type="invalid_numeric",
                        severity=ValidationSeverity.ERROR,
                        message=f"{field}不是有效数值: {value}",
                        current_value=value,
                        fix_suggestion="确保数值字段为数字类型"
                    ))

        return issues

    def _validate_query_data_basic(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """查询数据基础验证"""
        issues = []

        # 检查查询字符串
        if "query" in data:
            query = data["query"]
            if not isinstance(query, str):
                issues.append(ValidationIssue(
                    field="query",
                    issue_type="invalid_type",
                    severity=ValidationSeverity.ERROR,
                    message="查询字段应为字符串",
                    current_value=query,
                    fix_suggestion="确保查询参数为字符串类型"
                ))
            elif len(query.strip()) == 0:
                issues.append(ValidationIssue(
                    field="query",
                    issue_type="empty_query",
                    severity=ValidationSeverity.WARNING,
                    message="查询字符串为空",
                    current_value=query,
                    fix_suggestion="提供有效的查询内容"
                ))
            elif len(query) > 1000:
                issues.append(ValidationIssue(
                    field="query",
                    issue_type="excessive_length",
                    severity=ValidationSeverity.WARNING,
                    message=f"查询字符串过长: {len(query)}字符",
                    current_value=len(query),
                    fix_suggestion="简化查询内容"
                ))

        return issues

    def _validate_date_data_basic(self, data: Any) -> List[ValidationIssue]:
        """日期数据基础验证"""
        issues = []

        if isinstance(data, str):
            # 验证日期格式
            date_patterns = [
                (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d', 'YYYY-MM-DD'),
                (r'^\d{8}$', '%Y%m%d', 'YYYYMMDD'),
                (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d', 'YYYY/MM/DD')
            ]

            valid_format = False
            for pattern, date_format, format_name in date_patterns:
                if re.match(pattern, data):
                    try:
                        datetime.strptime(data, date_format)
                        valid_format = True
                        break
                    except ValueError:
                        continue

            if not valid_format:
                issues.append(ValidationIssue(
                    field="date",
                    issue_type="invalid_date_format",
                    severity=ValidationSeverity.ERROR,
                    message=f"无效的日期格式: {data}",
                    current_value=data,
                    fix_suggestion="使用YYYY-MM-DD或YYYYMMDD格式"
                ))

        return issues

    # ============= 标准验证层 =============

    def _standard_validation(self, data: Any, data_type: str,
                             context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """标准数据验证"""
        issues = []

        if data_type == "financial_data":
            issues.extend(self._validate_financial_logic(data))
        elif data_type == "api_response":
            issues.extend(self._validate_api_consistency(data))

        return issues

    def _validate_financial_logic(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """金融业务逻辑验证"""
        issues = []

        try:
            # 提取数值
            total_balance = float(data.get("总余额", 0))
            total_inflow = float(data.get("总入金", 0))
            total_outflow = float(data.get("总出金", 0))
            total_investment = float(data.get("总投资金额", 0))

            # 逻辑1: 总余额应该合理 (不能远超入金)
            if total_inflow > 0:
                balance_ratio = total_balance / total_inflow
                if balance_ratio > 5:  # 余额超过入金5倍可能有问题
                    issues.append(ValidationIssue(
                        field="总余额",
                        issue_type="balance_logic_warning",
                        severity=ValidationSeverity.WARNING,
                        message=f"总余额是入金的{balance_ratio:.1f}倍，可能存在逻辑问题",
                        current_value=total_balance,
                        fix_suggestion="检查资金计算逻辑"
                    ))

            # 逻辑2: 出金不应超过入金太多
            if total_inflow > 0:
                outflow_ratio = total_outflow / total_inflow
                if outflow_ratio > 1.2:  # 出金超过入金20%需要关注
                    issues.append(ValidationIssue(
                        field="总出金",
                        issue_type="outflow_excess",
                        severity=ValidationSeverity.WARNING,
                        message=f"出金比例过高: {outflow_ratio * 100:.1f}%",
                        current_value=total_outflow,
                        fix_suggestion="关注流动性风险"
                    ))

            # 逻辑3: 投资金额与余额的关系
            if total_balance > 0 and total_investment > 0:
                investment_ratio = total_investment / total_balance
                if investment_ratio > 10:  # 投资额超过余额10倍异常
                    issues.append(ValidationIssue(
                        field="总投资金额",
                        issue_type="investment_logic_error",
                        severity=ValidationSeverity.ERROR,
                        message=f"投资金额异常: 是余额的{investment_ratio:.1f}倍",
                        current_value=total_investment,
                        fix_suggestion="检查投资金额计算方式"
                    ))

        except (ValueError, TypeError) as e:
            issues.append(ValidationIssue(
                field="financial_logic",
                issue_type="calculation_error",
                severity=ValidationSeverity.ERROR,
                message=f"金融逻辑验证计算错误: {str(e)}",
                current_value=None,
                fix_suggestion="检查数据类型和格式"
            ))

        return issues

    def _validate_api_consistency(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """API数据一致性验证"""
        issues = []

        # 检查result与status的一致性
        result = data.get("result")
        status = data.get("status")

        if result is True and status != 0:
            issues.append(ValidationIssue(
                field="status",
                issue_type="result_status_mismatch",
                severity=ValidationSeverity.WARNING,
                message=f"result为true但status为{status}，可能不一致",
                current_value=status,
                suggested_value=0,
                fix_suggestion="检查API响应逻辑"
            ))

        # 检查数据结构完整性
        if result is True and "data" in data:
            data_content = data["data"]
            if isinstance(data_content, dict) and len(data_content) == 0:
                issues.append(ValidationIssue(
                    field="data",
                    issue_type="empty_data_object",
                    severity=ValidationSeverity.WARNING,
                    message="成功响应的data字段为空对象",
                    current_value=data_content,
                    fix_suggestion="确认是否应该返回数据"
                ))

        return issues

    # ============= 严格验证层 =============

    async def _strict_validation(self, data: Any, data_type: str,
                                 context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """严格数据验证"""
        issues = []

        if data_type == "financial_data":
            issues.extend(await self._validate_financial_anomalies(data, context))

        return issues

    async def _validate_financial_anomalies(self, data: Dict[str, Any],
                                            context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """金融数据异常检测"""
        issues = []

        try:
            # 获取历史数据进行对比 (如果有)
            historical_data = context.get("historical_data", []) if context else []

            current_balance = float(data.get("总余额", 0))
            current_users = data.get("用户统计", {}).get("总用户数", 0)

            # 异常检测1: 余额突然大幅变化
            if historical_data and len(historical_data) > 0:
                last_balance = float(historical_data[-1].get("总余额", 0))
                if last_balance > 0:
                    change_ratio = abs(current_balance - last_balance) / last_balance
                    if change_ratio > 0.5:  # 50%以上变化
                        issues.append(ValidationIssue(
                            field="总余额",
                            issue_type="balance_anomaly",
                            severity=ValidationSeverity.WARNING,
                            message=f"余额变化异常: {change_ratio * 100:.1f}%",
                            current_value=current_balance,
                            fix_suggestion="检查是否有重大资金变动"
                        ))

            # 异常检测2: 用户数与资金不匹配
            if current_users > 0 and current_balance > 0:
                avg_balance_per_user = current_balance / current_users
                if avg_balance_per_user > 100000:  # 人均10万以上
                    issues.append(ValidationIssue(
                        field="user_balance_ratio",
                        issue_type="user_balance_anomaly",
                        severity=ValidationSeverity.INFO,
                        message=f"人均余额较高: ¥{avg_balance_per_user:.0f}",
                        current_value=avg_balance_per_user,
                        fix_suggestion="关注高净值用户集中度"
                    ))
                elif avg_balance_per_user < 100:  # 人均100以下
                    issues.append(ValidationIssue(
                        field="user_balance_ratio",
                        issue_type="low_user_balance",
                        severity=ValidationSeverity.INFO,
                        message=f"人均余额较低: ¥{avg_balance_per_user:.0f}",
                        current_value=avg_balance_per_user,
                        fix_suggestion="分析用户活跃度和留存"
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                field="anomaly_detection",
                issue_type="detection_error",
                severity=ValidationSeverity.WARNING,
                message=f"异常检测失败: {str(e)}",
                current_value=None,
                fix_suggestion="检查输入数据格式"
            ))

        return issues

    # ============= AI增强验证层 =============

    async def _ai_enhanced_validation(self, data: Any, data_type: str,
                                      context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """AI增强验证"""
        issues = []

        # Claude业务逻辑验证
        if self.claude_client:
            claude_issues = await self._claude_business_validation(data, data_type, context)
            issues.extend(claude_issues)

        # GPT数据一致性验证
        if self.gpt_client:
            gpt_issues = await self._gpt_consistency_validation(data, data_type, context)
            issues.extend(gpt_issues)

        return issues

    async def _claude_business_validation(self, data: Any, data_type: str,
                                          context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Claude业务逻辑验证"""

        if not self.claude_client:
            return []

        try:
            validation_prompt = f"""
作为一位资深的金融数据分析师，请验证以下数据的业务逻辑合理性：

数据类型: {data_type}
数据内容: {json.dumps(data, ensure_ascii=False, indent=2)}
上下文信息: {json.dumps(context or {}, ensure_ascii=False, indent=2)}

请从以下角度分析：
1. 数据之间的逻辑关系是否合理
2. 是否存在明显的业务异常
3. 数据规模是否在正常范围内
4. 是否有潜在的风险信号

返回JSON格式的验证结果：
{{
    "overall_assessment": "valid/warning/invalid",
    "business_logic_score": 0.0-1.0,
    "identified_issues": [
        {{
            "field": "字段名",
            "issue_type": "问题类型",
            "severity": "info/warning/error/critical",
            "message": "问题描述",
            "business_impact": "业务影响说明",
            "recommendation": "建议措施"
        }}
    ],
    "positive_indicators": ["积极指标列表"],
    "risk_factors": ["风险因素列表"]
}}
"""

            result = await self.claude_client.analyze_complex_query(validation_prompt, {
                "data": data,
                "data_type": data_type,
                "context": context
            })

            if result.get("success"):
                return self._process_claude_validation_result(result["analysis"])
            else:
                logger.warning("Claude业务验证失败")
                return []

        except Exception as e:
            logger.error(f"Claude业务验证异常: {str(e)}")
            return []

    async def _gpt_consistency_validation(self, data: Any, data_type: str,
                                          context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """GPT数据一致性验证"""

        if not self.gpt_client:
            return []

        try:
            validation_prompt = f"""
验证以下数据的数学逻辑和数值一致性：

数据: {json.dumps(data, ensure_ascii=False)}
类型: {data_type}

检查要点：
1. 数值计算是否正确
2. 百分比和比率是否合理
3. 累计数据是否一致
4. 时间序列数据是否连续

返回验证结果，重点关注数值准确性和逻辑一致性。
"""

            result = await self.gpt_client.validate_and_verify(data, validation_prompt)

            if result.get("success"):
                return self._process_gpt_validation_result(result["validation"])
            else:
                logger.warning("GPT一致性验证失败")
                return []

        except Exception as e:
            logger.error(f"GPT一致性验证异常: {str(e)}")
            return []

    def _process_claude_validation_result(self, claude_result: Dict[str, Any]) -> List[ValidationIssue]:
        """处理Claude验证结果"""
        issues = []

        try:
            identified_issues = claude_result.get("identified_issues", [])

            for issue_data in identified_issues:
                severity_map = {
                    "info": ValidationSeverity.INFO,
                    "warning": ValidationSeverity.WARNING,
                    "error": ValidationSeverity.ERROR,
                    "critical": ValidationSeverity.CRITICAL
                }

                severity = severity_map.get(issue_data.get("severity", "warning"), ValidationSeverity.WARNING)

                issues.append(ValidationIssue(
                    field=issue_data.get("field", "unknown"),
                    issue_type=f"claude_{issue_data.get('issue_type', 'business_logic')}",
                    severity=severity,
                    message=issue_data.get("message", ""),
                    current_value=None,
                    fix_suggestion=issue_data.get("recommendation", "")
                ))

        except Exception as e:
            logger.error(f"Claude结果处理失败: {str(e)}")

        return issues

    def _process_gpt_validation_result(self, gpt_result: Dict[str, Any]) -> List[ValidationIssue]:
        """处理GPT验证结果"""
        issues = []

        try:
            gpt_issues = gpt_result.get("issues", [])

            for issue_data in gpt_issues:
                severity_map = {
                    "low": ValidationSeverity.INFO,
                    "medium": ValidationSeverity.WARNING,
                    "high": ValidationSeverity.ERROR
                }

                severity = severity_map.get(issue_data.get("severity", "medium"), ValidationSeverity.WARNING)

                issues.append(ValidationIssue(
                    field="data_consistency",
                    issue_type=f"gpt_{issue_data.get('type', 'consistency')}",
                    severity=severity,
                    message=issue_data.get("description", ""),
                    current_value=None,
                    fix_suggestion=issue_data.get("suggestion", "")
                ))

        except Exception as e:
            logger.error(f"GPT结果处理失败: {str(e)}")

        return issues

    # ============= 结果生成和工具方法 =============

    def _create_validation_result(self, issues: List[ValidationIssue],
                                  validation_level: ValidationLevel) -> ValidationResult:
        """创建验证结果"""

        # 统计各种严重程度的问题
        errors_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warnings_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])

        # 计算总体评分
        total_issues = len(issues)
        if total_issues == 0:
            overall_score = 1.0
        else:
            # 按严重程度加权扣分
            penalty = (critical_count * 0.5 + errors_count * 0.3 + warnings_count * 0.1)
            overall_score = max(0.0, 1.0 - min(penalty / 2, 1.0))

        # 判断是否通过验证
        is_valid = critical_count == 0 and errors_count == 0

        return ValidationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            issues=issues,
            warnings_count=warnings_count,
            errors_count=errors_count + critical_count,
            validation_level=validation_level,
            validation_timestamp=datetime.now().isoformat(),
            metadata={
                "total_issues": total_issues,
                "critical_issues": critical_count,
                "validation_summary": f"{total_issues}个问题 ({critical_count}严重, {errors_count}错误, {warnings_count}警告)"
            }
        )

    def _create_error_validation_result(self, error_msg: str,
                                        validation_level: ValidationLevel) -> ValidationResult:
        """创建错误验证结果"""
        error_issue = ValidationIssue(
            field="validation_system",
            issue_type="validation_error",
            severity=ValidationSeverity.CRITICAL,
            message=f"验证系统错误: {error_msg}",
            current_value=None,
            fix_suggestion="检查验证系统配置"
        )

        return ValidationResult(
            is_valid=False,
            overall_score=0.0,
            issues=[error_issue],
            warnings_count=0,
            errors_count=1,
            validation_level=validation_level,
            validation_timestamp=datetime.now().isoformat(),
            metadata={"validation_error": error_msg}
        )

    # ============= 便捷验证方法 =============

    async def validate_api_response(self, response: Dict[str, Any],
                                    expected_fields: List[str] = None) -> ValidationResult:
        """验证API响应"""
        context = {"expected_fields": expected_fields} if expected_fields else None
        return await self.validate_data(response, "api_response", ValidationLevel.STANDARD, context)

    async def validate_financial_data(self, data: Dict[str, Any],
                                      historical_context: List[Dict] = None) -> ValidationResult:
        """验证金融数据"""
        context = {"historical_data": historical_context} if historical_context else None
        return await self.validate_data(data, "financial_data", ValidationLevel.AI_ENHANCED, context)

    def validate_date_format(self, date_str: str, required_format: str = "YYYY-MM-DD") -> bool:
        """验证日期格式"""
        format_map = {
            "YYYY-MM-DD": r'^\d{4}-\d{2}-\d{2}$',
            "YYYYMMDD": r'^\d{8}$',
            "YYYY/MM/DD": r'^\d{4}/\d{2}/\d{2}$'
        }

        pattern = format_map.get(required_format)
        if not pattern:
            return False

        return bool(re.match(pattern, date_str))

    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """获取验证摘要"""
        return {
            "status": "通过" if result.is_valid else "未通过",
            "score": f"{result.overall_score * 100:.1f}分",
            "level": result.validation_level.value,
            "issues": {
                "total": len(result.issues),
                "errors": result.errors_count,
                "warnings": result.warnings_count
            },
            "timestamp": result.validation_timestamp,
            "top_issues": [
                {
                    "field": issue.field,
                    "severity": issue.severity.value,
                    "message": issue.message
                }
                for issue in
                sorted(result.issues, key=lambda x: ["critical", "error", "warning", "info"].index(x.severity.value))[
                :3]
            ]
        }


# ============= 工厂函数 =============

def create_validation_utils(claude_client=None, gpt_client=None) -> ValidationUtils:
    """
    创建验证工具实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        ValidationUtils: 验证工具实例
    """
    return ValidationUtils(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建验证工具
    validator = create_validation_utils()

    print("=== 数据验证工具测试 ===")

    # 1. API响应验证
    api_response = {
        "result": True,
        "status": 0,
        "data": {
            "总余额": "8223695.0731",
            "总用户数": 13723
        }
    }

    api_validation = await validator.validate_api_response(api_response)
    print(f"API验证: {'通过' if api_validation.is_valid else '未通过'}")
    print(f"评分: {api_validation.overall_score * 100:.1f}分")

    # 2. 金融数据验证
    financial_data = {
        "总余额": "8223695.0731",
        "总入金": "17227143.9231",
        "总出金": "9003448.85",
        "总投资金额": "30772686.00"
    }

    financial_validation = await validator.validate_financial_data(financial_data)
    print(f"金融数据验证: {'通过' if financial_validation.is_valid else '未通过'}")
    print(f"问题数量: {len(financial_validation.issues)}")

    # 3. 日期格式验证
    date_valid = validator.validate_date_format("2024-06-01", "YYYY-MM-DD")
    print(f"日期格式验证: {'有效' if date_valid else '无效'}")

    # 4. 验证摘要
    summary = validator.get_validation_summary(financial_validation)
    print(f"验证摘要: {summary['status']}, {summary['score']}")


if __name__ == "__main__":
    asyncio.run(main())