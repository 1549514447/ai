# utils/data_transformers/time_series_builder.py
"""
📈 AI辅助的时间序列数据构建器
专为金融AI分析系统设计，将API原始数据转换为分析友好的时间序列格式

核心特点:
- 智能数据对齐和补全
- 多维度时间序列构建
- 异常值检测和处理
- AI驱动的数据质量优化
- 为预测和趋势分析优化的数据结构
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from dataclasses import dataclass
from enum import Enum
import json

# 导入我们的工具类
from utils.helpers.date_utils import DateUtils, create_date_utils
from utils.calculators.financial_calculator import FinancialCalculator, create_financial_calculator

logger = logging.getLogger(__name__)


class TimeSeriesType(Enum):
    """时间序列类型"""
    DAILY = "daily"  # 每日数据
    WEEKLY = "weekly"  # 周度数据
    MONTHLY = "monthly"  # 月度数据
    CUMULATIVE = "cumulative"  # 累计数据
    MOVING_AVERAGE = "moving_avg"  # 移动平均


class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 95%以上完整性
    GOOD = "good"  # 80-95%完整性
    FAIR = "fair"  # 60-80%完整性
    POOR = "poor"  # 60%以下完整性


@dataclass
class TimeSeriesMetadata:
    """时间序列元数据"""
    series_type: TimeSeriesType
    date_range: Tuple[str, str]  # (start_date, end_date)
    total_points: int  # 总数据点数
    missing_points: int  # 缺失数据点数
    data_quality: DataQuality  # 数据质量
    interpolated_points: int  # 插值数据点数
    outliers_detected: int  # 检测到的异常值数量
    confidence_score: float  # 数据置信度 (0-1)


@dataclass
class TimeSeriesPoint:
    """时间序列数据点"""
    date: str  # 日期 (YYYY-MM-DD)
    value: float  # 数值
    is_interpolated: bool = False  # 是否为插值
    is_outlier: bool = False  # 是否为异常值
    confidence: float = 1.0  # 数据置信度
    metadata: Dict[str, Any] = None  # 额外元数据


class TimeSeriesBuilder:
    """
    📈 AI辅助的时间序列数据构建器

    功能特点:
    1. 智能数据对齐和补全
    2. 多维度时间序列构建
    3. AI驱动的异常值检测
    4. 自动数据质量评估
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化时间序列构建器

        Args:
            claude_client: Claude客户端，用于数据模式分析
            gpt_client: GPT客户端，用于数值计算验证
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.date_utils = create_date_utils(claude_client)
        self.financial_calculator = create_financial_calculator(gpt_client)

        # 配置参数
        self.config = {
            "outlier_threshold": 2.5,  # 异常值阈值 (标准差倍数)
            "interpolation_method": "linear",  # 插值方法
            "min_data_quality": 0.6,  # 最低数据质量要求
            "max_gap_days": 7,  # 最大允许数据间隔(天)
            "enable_ai_validation": True  # 启用AI验证
        }

        logger.info("TimeSeriesBuilder initialized with AI capabilities")

    # ============= 核心时间序列构建方法 =============

    async def build_daily_time_series(self, raw_data: List[Dict[str, Any]],
                                      metric_field: str,
                                      date_field: str = "日期") -> Dict[str, Any]:
        """
        构建每日时间序列

        Args:
            raw_data: 原始数据列表
            metric_field: 指标字段名 (如"入金", "出金", "注册人数")
            date_field: 日期字段名

        Returns:
            Dict[str, Any]: 时间序列结果
        """
        try:
            logger.info(f"📈 构建每日时间序列: {metric_field}")

            if not raw_data:
                return self._create_empty_series_result(metric_field)

            # 🔍 数据预处理和验证
            cleaned_data = await self._preprocess_raw_data(raw_data, metric_field, date_field)

            if not cleaned_data:
                logger.warning("数据预处理后为空")
                return self._create_empty_series_result(metric_field)

            # 🔧 构建基础时间序列
            base_series = self._build_base_series(cleaned_data, metric_field, date_field)

            # 📅 补全缺失日期
            complete_series = await self._fill_missing_dates(base_series)

            # 🚨 异常值检测和处理
            processed_series = await self._detect_and_handle_outliers(complete_series, metric_field)

            # 📊 数据质量评估
            metadata = self._calculate_series_metadata(processed_series, TimeSeriesType.DAILY)

            # 🧮 统计指标计算
            statistics = await self._calculate_series_statistics(processed_series, metric_field)

            # 🧠 AI数据验证 (可选)
            ai_validation = {}
            if self.config["enable_ai_validation"] and self.claude_client:
                ai_validation = await self._ai_validate_time_series(processed_series, metric_field)

            return {
                "success": True,
                "metric": metric_field,
                "series_type": TimeSeriesType.DAILY.value,
                "time_series": processed_series,
                "metadata": metadata,
                "statistics": statistics,
                "ai_validation": ai_validation,
                "total_points": len(processed_series),
                "date_range": self._get_date_range(processed_series),
                "build_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 时间序列构建失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metric": metric_field,
                "series_type": TimeSeriesType.DAILY.value
            }

    async def build_multi_metric_series(self, raw_data: List[Dict[str, Any]],
                                        metrics: List[str],
                                        date_field: str = "日期") -> Dict[str, Any]:
        """
        构建多指标时间序列

        Args:
            raw_data: 原始数据列表
            metrics: 指标列表 (如["入金", "出金", "注册人数"])
            date_field: 日期字段名

        Returns:
            Dict[str, Any]: 多指标时间序列结果
        """
        try:
            logger.info(f"📊 构建多指标时间序列: {metrics}")

            if not raw_data or not metrics:
                return {"success": False, "error": "数据或指标列表为空"}

            # 并行构建各个指标的时间序列
            series_tasks = [
                self.build_daily_time_series(raw_data, metric, date_field)
                for metric in metrics
            ]

            series_results = await asyncio.gather(*series_tasks, return_exceptions=True)

            # 整理多指标结果
            multi_series = {}
            successful_metrics = []
            failed_metrics = []

            for i, result in enumerate(series_results):
                metric = metrics[i]

                if isinstance(result, Exception):
                    logger.error(f"指标 {metric} 构建失败: {str(result)}")
                    failed_metrics.append({"metric": metric, "error": str(result)})
                elif result.get("success"):
                    multi_series[metric] = result
                    successful_metrics.append(metric)
                else:
                    failed_metrics.append({"metric": metric, "error": result.get("error", "Unknown error")})

            if not successful_metrics:
                return {
                    "success": False,
                    "error": "所有指标构建都失败",
                    "failed_metrics": failed_metrics
                }

            # 🔗 构建对齐的时间序列矩阵
            aligned_series = await self._align_multi_series(multi_series)

            # 📈 计算指标间相关性
            correlations = await self._calculate_correlations(aligned_series, successful_metrics)

            # 📊 综合统计信息
            overall_stats = self._calculate_multi_series_stats(multi_series)

            return {
                "success": True,
                "metrics": successful_metrics,
                "failed_metrics": failed_metrics,
                "individual_series": multi_series,
                "aligned_series": aligned_series,
                "correlations": correlations,
                "overall_statistics": overall_stats,
                "build_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 多指标时间序列构建失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metrics": metrics
            }

    async def build_aggregated_series(self, daily_series: List[TimeSeriesPoint],
                                      aggregation_type: str = "weekly") -> Dict[str, Any]:
        """
        构建聚合时间序列 (周度/月度)

        Args:
            daily_series: 每日时间序列数据
            aggregation_type: 聚合类型 ("weekly", "monthly")

        Returns:
            Dict[str, Any]: 聚合时间序列结果
        """
        try:
            logger.info(f"📅 构建{aggregation_type}聚合时间序列")

            if not daily_series:
                return {"success": False, "error": "输入序列为空"}

            # 🗓️ 根据聚合类型分组数据
            if aggregation_type == "weekly":
                grouped_data = self._group_by_week(daily_series)
                series_type = TimeSeriesType.WEEKLY
            elif aggregation_type == "monthly":
                grouped_data = self._group_by_month(daily_series)
                series_type = TimeSeriesType.MONTHLY
            else:
                return {"success": False, "error": f"不支持的聚合类型: {aggregation_type}"}

            # 🧮 计算聚合值
            aggregated_points = []
            for period, points in grouped_data.items():
                if points:
                    # 计算平均值、总和等统计量
                    values = [p.value for p in points if not p.is_interpolated]

                    if values:
                        agg_point = TimeSeriesPoint(
                            date=period,
                            value=sum(values),  # 默认使用总和
                            is_interpolated=False,
                            confidence=sum(p.confidence for p in points) / len(points),
                            metadata={
                                "period_type": aggregation_type,
                                "data_points_count": len(values),
                                "avg_value": sum(values) / len(values),
                                "max_value": max(values),
                                "min_value": min(values),
                                "interpolated_points": sum(1 for p in points if p.is_interpolated)
                            }
                        )
                        aggregated_points.append(agg_point)

            # 📊 计算聚合元数据
            metadata = self._calculate_series_metadata(aggregated_points, series_type)

            return {
                "success": True,
                "aggregation_type": aggregation_type,
                "series_type": series_type.value,
                "aggregated_series": aggregated_points,
                "metadata": metadata,
                "original_points": len(daily_series),
                "aggregated_points": len(aggregated_points),
                "build_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 聚合时间序列构建失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "aggregation_type": aggregation_type
            }

    # ============= 数据预处理方法 =============

    async def _preprocess_raw_data(self, raw_data: List[Dict[str, Any]],
                                   metric_field: str, date_field: str) -> List[Dict[str, Any]]:
        """预处理原始数据"""

        cleaned_data = []

        for item in raw_data:
            try:
                # 检查必需字段
                if date_field not in item or metric_field not in item:
                    continue

                # 日期格式转换和验证
                date_value = item[date_field]
                if isinstance(date_value, str):
                    # 尝试转换API格式 (YYYYMMDD) 到标准格式
                    if len(date_value) == 8 and date_value.isdigit():
                        date_value = self.date_utils.api_format_to_date(date_value)

                    # 验证日期格式
                    if not self.date_utils.validate_date_format(date_value):
                        logger.warning(f"无效日期格式: {date_value}")
                        continue

                # 数值格式转换和验证
                metric_value = item[metric_field]
                try:
                    if isinstance(metric_value, str):
                        # 移除可能的货币符号和逗号
                        metric_value = metric_value.replace('¥', '').replace(',', '').strip()

                    numeric_value = float(metric_value)

                    # 基础合理性检查
                    if numeric_value < 0 and metric_field in ["入金", "出金", "注册人数", "总余额"]:
                        logger.warning(f"负数值可能异常: {metric_field}={numeric_value}")

                except (ValueError, TypeError):
                    logger.warning(f"无效数值: {metric_field}={metric_value}")
                    continue

                # 添加到清理后的数据
                cleaned_item = item.copy()
                cleaned_item[date_field] = date_value
                cleaned_item[metric_field] = numeric_value
                cleaned_data.append(cleaned_item)

            except Exception as e:
                logger.warning(f"数据项预处理失败: {str(e)}")
                continue

        # 按日期排序
        cleaned_data.sort(key=lambda x: x[date_field])

        logger.info(f"数据预处理完成: {len(raw_data)} -> {len(cleaned_data)} 项")
        return cleaned_data

    def _build_base_series(self, cleaned_data: List[Dict[str, Any]],
                           metric_field: str, date_field: str) -> List[TimeSeriesPoint]:
        """构建基础时间序列"""

        series_points = []

        for item in cleaned_data:
            point = TimeSeriesPoint(
                date=item[date_field],
                value=float(item[metric_field]),
                is_interpolated=False,
                confidence=1.0,
                metadata={
                    "source": "raw_data",
                    "original_item": item
                }
            )
            series_points.append(point)

        return series_points

    async def _fill_missing_dates(self, series: List[TimeSeriesPoint]) -> List[TimeSeriesPoint]:
        """补全缺失日期"""

        if len(series) < 2:
            return series

        # 获取日期范围
        start_date = datetime.strptime(series[0].date, "%Y-%m-%d")
        end_date = datetime.strptime(series[-1].date, "%Y-%m-%d")

        # 生成完整日期列表
        complete_dates = []
        current_date = start_date
        while current_date <= end_date:
            complete_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        # 创建日期到数据点的映射
        existing_data = {point.date: point for point in series}

        # 补全序列
        complete_series = []
        for date_str in complete_dates:
            if date_str in existing_data:
                complete_series.append(existing_data[date_str])
            else:
                # 插值处理缺失数据
                interpolated_value = self._interpolate_missing_value(
                    date_str, complete_series, series
                )

                missing_point = TimeSeriesPoint(
                    date=date_str,
                    value=interpolated_value,
                    is_interpolated=True,
                    confidence=0.7,  # 插值数据置信度较低
                    metadata={
                        "source": "interpolation",
                        "method": self.config["interpolation_method"]
                    }
                )
                complete_series.append(missing_point)

        logger.info(f"日期补全: {len(series)} -> {len(complete_series)} 点")
        return complete_series

    def _interpolate_missing_value(self, missing_date: str,
                                   current_series: List[TimeSeriesPoint],
                                   original_series: List[TimeSeriesPoint]) -> float:
        """插值计算缺失值"""

        if not current_series:
            # 如果是第一个点，使用原始序列的第一个值
            return original_series[0].value if original_series else 0.0

        if self.config["interpolation_method"] == "linear":
            # 线性插值
            last_point = current_series[-1]

            # 查找下一个已知数据点
            missing_dt = datetime.strptime(missing_date, "%Y-%m-%d")
            next_point = None

            for point in original_series:
                point_dt = datetime.strptime(point.date, "%Y-%m-%d")
                if point_dt > missing_dt:
                    next_point = point
                    break

            if next_point:
                # 线性插值计算
                last_dt = datetime.strptime(last_point.date, "%Y-%m-%d")
                next_dt = datetime.strptime(next_point.date, "%Y-%m-%d")

                total_days = (next_dt - last_dt).days
                elapsed_days = (missing_dt - last_dt).days

                if total_days > 0:
                    ratio = elapsed_days / total_days
                    interpolated = last_point.value + (next_point.value - last_point.value) * ratio
                    return interpolated

            # 如果找不到下一个点，使用前一个值
            return last_point.value

        elif self.config["interpolation_method"] == "forward_fill":
            # 前向填充
            return current_series[-1].value if current_series else 0.0

        else:
            # 默认使用0
            return 0.0

    async def _detect_and_handle_outliers(self, series: List[TimeSeriesPoint],
                                          metric_field: str) -> List[TimeSeriesPoint]:
        """异常值检测和处理"""

        if len(series) < 3:
            return series

        # 提取非插值的数值用于统计
        real_values = [p.value for p in series if not p.is_interpolated]

        if len(real_values) < 3:
            return series

        # 计算统计量
        mean_val = sum(real_values) / len(real_values)
        squared_diffs = [(x - mean_val) ** 2 for x in real_values]
        std_val = (sum(squared_diffs) / len(squared_diffs)) ** 0.5

        # 异常值阈值
        threshold = self.config["outlier_threshold"]
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val

        # 标记异常值
        processed_series = []
        outliers_count = 0

        for point in series:
            new_point = TimeSeriesPoint(
                date=point.date,
                value=point.value,
                is_interpolated=point.is_interpolated,
                is_outlier=False,
                confidence=point.confidence,
                metadata=point.metadata.copy() if point.metadata else {}
            )

            # 检测异常值 (只对非插值数据检测)
            if not point.is_interpolated and (point.value < lower_bound or point.value > upper_bound):
                new_point.is_outlier = True
                new_point.confidence = min(new_point.confidence, 0.5)  # 降低置信度
                outliers_count += 1

                if new_point.metadata:
                    new_point.metadata["outlier_info"] = {
                        "deviation": abs(point.value - mean_val) / std_val if std_val > 0 else 0,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    }

            processed_series.append(new_point)

        logger.info(f"异常值检测完成: {outliers_count}/{len(series)} 个异常值")

        # 🧠 AI验证异常值 (可选)
        if outliers_count > 0 and self.claude_client:
            await self._ai_validate_outliers(processed_series, metric_field)

        return processed_series

    async def _ai_validate_outliers(self, series: List[TimeSeriesPoint], metric_field: str):
        """AI验证异常值"""

        try:
            outliers = [p for p in series if p.is_outlier]

            if not outliers:
                return

            outlier_summary = [
                {
                    "date": p.date,
                    "value": p.value,
                    "deviation": p.metadata.get("outlier_info", {}).get("deviation", 0)
                }
                for p in outliers[:5]  # 只分析前5个异常值
            ]

            validation_prompt = f"""
分析以下金融指标的异常值是否合理：

指标: {metric_field}
异常值列表: {json.dumps(outlier_summary, ensure_ascii=False)}

请判断：
1. 这些异常值是否可能是真实的业务事件（如大额交易、营销活动等）
2. 是否存在数据错误的可能性
3. 建议的处理方式

返回简洁的分析结果。
"""

            result = await self.claude_client.analyze_complex_query(validation_prompt, {
                "metric": metric_field,
                "outliers": outlier_summary
            })

            if result.get("success"):
                logger.info(f"AI异常值验证完成: {metric_field}")
                # 可以根据AI建议调整异常值的置信度

        except Exception as e:
            logger.warning(f"AI异常值验证失败: {str(e)}")

    # ============= 统计计算方法 =============

    def _calculate_series_metadata(self, series: List[TimeSeriesPoint],
                                   series_type: TimeSeriesType) -> TimeSeriesMetadata:
        """计算时间序列元数据"""

        if not series:
            return TimeSeriesMetadata(
                series_type=series_type,
                date_range=("", ""),
                total_points=0,
                missing_points=0,
                data_quality=DataQuality.POOR,
                interpolated_points=0,
                outliers_detected=0,
                confidence_score=0.0
            )

        total_points = len(series)
        interpolated_points = sum(1 for p in series if p.is_interpolated)
        outliers_detected = sum(1 for p in series if p.is_outlier)
        missing_points = interpolated_points  # 插值点即为原缺失点

        # 计算数据质量
        completeness_ratio = (total_points - missing_points) / total_points if total_points > 0 else 0

        if completeness_ratio >= 0.95:
            data_quality = DataQuality.EXCELLENT
        elif completeness_ratio >= 0.80:
            data_quality = DataQuality.GOOD
        elif completeness_ratio >= 0.60:
            data_quality = DataQuality.FAIR
        else:
            data_quality = DataQuality.POOR

        # 计算置信度
        avg_confidence = sum(p.confidence for p in series) / total_points if total_points > 0 else 0
        quality_penalty = outliers_detected * 0.05  # 异常值降低置信度
        confidence_score = max(0.0, avg_confidence - quality_penalty)

        return TimeSeriesMetadata(
            series_type=series_type,
            date_range=(series[0].date, series[-1].date),
            total_points=total_points,
            missing_points=missing_points,
            data_quality=data_quality,
            interpolated_points=interpolated_points,
            outliers_detected=outliers_detected,
            confidence_score=confidence_score
        )

    async def _calculate_series_statistics(self, series: List[TimeSeriesPoint],
                                           metric_field: str) -> Dict[str, Any]:
        """计算时间序列统计信息"""

        if not series:
            return {}

        values = [p.value for p in series]
        real_values = [p.value for p in series if not p.is_interpolated]

        # 基础统计
        stats = {
            "total_points": len(values),
            "real_data_points": len(real_values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "sum": sum(values)
        }

        # 计算标准差和方差
        if len(values) > 1:
            mean_val = stats["mean"]
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            stats["variance"] = variance
            stats["std_dev"] = variance ** 0.5
            stats["coefficient_of_variation"] = stats["std_dev"] / mean_val if mean_val != 0 else 0

        # 🧮 使用金融计算器计算增长率
        if len(real_values) >= 2 and self.financial_calculator:
            growth_analysis = self.financial_calculator.calculate_growth_rate(real_values, "compound")
            stats["growth_analysis"] = {
                "growth_rate": growth_analysis.growth_rate,
                "trend_direction": growth_analysis.trend_direction,
                "volatility": growth_analysis.volatility,
                "confidence_level": growth_analysis.confidence_level
            }

        # 计算移动平均
        if len(values) >= 7:
            ma_7 = self._calculate_moving_average(values, 7)
            stats["moving_averages"] = {
                "ma_7": ma_7[-1] if ma_7 else None
            }

            if len(values) >= 30:
                ma_30 = self._calculate_moving_average(values, 30)
                stats["moving_averages"]["ma_30"] = ma_30[-1] if ma_30 else None

        # 数据质量统计
        stats["data_quality"] = {
            "interpolated_ratio": sum(1 for p in series if p.is_interpolated) / len(series),
            "outlier_ratio": sum(1 for p in series if p.is_outlier) / len(series),
            "avg_confidence": sum(p.confidence for p in series) / len(series)
        }

        return stats

    def _calculate_moving_average(self, values: List[float], window: int) -> List[float]:
        """计算移动平均"""
        if len(values) < window:
            return []

        ma_values = []
        for i in range(window - 1, len(values)):
            window_values = values[i - window + 1:i + 1]
            ma_values.append(sum(window_values) / window)

        return ma_values

    # ============= 多指标时间序列处理 =============

    async def _align_multi_series(self, multi_series: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """对齐多指标时间序列"""

        if not multi_series:
            return {}

        # 找到共同的日期范围
        all_dates = set()
        for metric, series_data in multi_series.items():
            series_points = series_data.get("time_series", [])
            for point in series_points:
                all_dates.add(point.date)

        sorted_dates = sorted(list(all_dates))

        # 构建对齐的矩阵
        aligned_data = {
            "dates": sorted_dates,
            "metrics": {}
        }

        for metric, series_data in multi_series.items():
            series_points = series_data.get("time_series", [])
            point_dict = {p.date: p for p in series_points}

            metric_values = []
            for date in sorted_dates:
                if date in point_dict:
                    metric_values.append(point_dict[date].value)
                else:
                    metric_values.append(None)  # 缺失值

            aligned_data["metrics"][metric] = metric_values

        return aligned_data

    async def _calculate_correlations(self, aligned_series: Dict[str, Any],
                                      metrics: List[str]) -> Dict[str, Any]:
        """计算指标间相关性"""

        correlations = {}

        if len(metrics) < 2:
            return correlations

        # 计算两两相关性
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics[i + 1:], i + 1):
                values1 = aligned_series["metrics"].get(metric1, [])
                values2 = aligned_series["metrics"].get(metric2, [])

                # 过滤None值
                valid_pairs = [
                    (v1, v2) for v1, v2 in zip(values1, values2)
                    if v1 is not None and v2 is not None
                ]

                if len(valid_pairs) >= 3:  # 至少需要3个数据点
                    correlation = self._calculate_pearson_correlation(valid_pairs)
                    correlations[f"{metric1}_vs_{metric2}"] = {
                        "correlation": correlation,
                        "data_points": len(valid_pairs),
                        "strength": self._interpret_correlation(correlation)
                    }

        return correlations

    def _calculate_pearson_correlation(self, pairs: List[Tuple[float, float]]) -> float:
        """计算皮尔逊相关系数"""

        if len(pairs) < 2:
            return 0.0

        n = len(pairs)
        sum_x = sum(x for x, y in pairs)
        sum_y = sum(y for x, y in pairs)
        sum_xy = sum(x * y for x, y in pairs)
        sum_x2 = sum(x * x for x, y in pairs)
        sum_y2 = sum(y * y for x, y in pairs)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _interpret_correlation(self, correlation: float) -> str:
        """解释相关性强度"""
        abs_corr = abs(correlation)

        if abs_corr >= 0.8:
            return "很强"
        elif abs_corr >= 0.6:
            return "强"
        elif abs_corr >= 0.4:
            return "中等"
        elif abs_corr >= 0.2:
            return "弱"
        else:
            return "很弱"

    # ============= 聚合时间序列处理 =============

    def _group_by_week(self, series: List[TimeSeriesPoint]) -> Dict[str, List[TimeSeriesPoint]]:
        """按周分组"""
        grouped = {}

        for point in series:
            date_obj = datetime.strptime(point.date, "%Y-%m-%d")
            # 获取周一的日期作为周标识
            monday = date_obj - timedelta(days=date_obj.weekday())
            week_key = monday.strftime("%Y-W%U")

            if week_key not in grouped:
                grouped[week_key] = []
            grouped[week_key].append(point)

        return grouped

    def _group_by_month(self, series: List[TimeSeriesPoint]) -> Dict[str, List[TimeSeriesPoint]]:
        """按月分组"""
        grouped = {}

        for point in series:
            date_obj = datetime.strptime(point.date, "%Y-%m-%d")
            month_key = date_obj.strftime("%Y-%m")

            if month_key not in grouped:
                grouped[month_key] = []
            grouped[month_key].append(point)

        return grouped

    # ============= AI增强验证 =============

    async def _ai_validate_time_series(self, series: List[TimeSeriesPoint],
                                       metric_field: str) -> Dict[str, Any]:
        """AI验证时间序列质量"""

        if not self.claude_client:
            return {"validation_performed": False, "reason": "no_claude_client"}

        try:
            # 准备验证数据
            sample_points = series[:10] + series[-10:]  # 取开始和结束的数据点
            validation_data = [
                {
                    "date": p.date,
                    "value": p.value,
                    "is_interpolated": p.is_interpolated,
                    "is_outlier": p.is_outlier,
                    "confidence": p.confidence
                }
                for p in sample_points
            ]

            validation_prompt = f"""
分析以下时间序列数据的质量和合理性：

指标: {metric_field}
总数据点: {len(series)}
采样数据: {json.dumps(validation_data, ensure_ascii=False)}

请评估：
1. 数据趋势是否合理
2. 数值范围是否正常
3. 是否存在明显的数据质量问题
4. 整体可信度评分 (0-1)

返回简洁的验证结果。
"""

            result = await self.claude_client.analyze_complex_query(validation_prompt, {
                "metric": metric_field,
                "series_length": len(series),
                "sample_data": validation_data
            })

            if result.get("success"):
                return {
                    "validation_performed": True,
                    "ai_assessment": result.get("analysis", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"validation_performed": False, "error": result.get("error", "Unknown error")}

        except Exception as e:
            logger.error(f"AI验证失败: {str(e)}")
            return {"validation_performed": False, "error": str(e)}

    # ============= 工具方法 =============

    def _create_empty_series_result(self, metric_field: str) -> Dict[str, Any]:
        """创建空序列结果"""
        return {
            "success": False,
            "error": "无有效数据",
            "metric": metric_field,
            "series_type": TimeSeriesType.DAILY.value,
            "time_series": [],
            "metadata": None
        }

    def _get_date_range(self, series: List[TimeSeriesPoint]) -> Tuple[str, str]:
        """获取序列日期范围"""
        if not series:
            return "", ""
        return series[0].date, series[-1].date

    def _calculate_multi_series_stats(self, multi_series: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算多序列统计信息"""

        total_metrics = len(multi_series)
        successful_series = sum(1 for s in multi_series.values() if s.get("success", False))

        avg_quality = 0
        total_points = 0

        for series_data in multi_series.values():
            if series_data.get("success"):
                metadata = series_data.get("metadata")
                if metadata:
                    avg_quality += metadata.confidence_score
                    total_points += metadata.total_points

        avg_quality = avg_quality / successful_series if successful_series > 0 else 0

        return {
            "total_metrics": total_metrics,
            "successful_metrics": successful_series,
            "success_rate": successful_series / total_metrics if total_metrics > 0 else 0,
            "average_quality_score": avg_quality,
            "total_data_points": total_points
        }


# ============= 工厂函数 =============

def create_time_series_builder(claude_client=None, gpt_client=None) -> TimeSeriesBuilder:
    """
    创建时间序列构建器实例

    Args:
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        TimeSeriesBuilder: 时间序列构建器实例
    """
    return TimeSeriesBuilder(claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""

    # 创建时间序列构建器
    builder = create_time_series_builder()

    print("=== 时间序列构建器测试 ===")

    # 模拟原始数据
    sample_data = [
        {"日期": "20240601", "入金": "180000.50", "注册人数": 25},
        {"日期": "20240602", "入金": "165000.30", "注册人数": 30},
        {"日期": "20240603", "入金": "195000.00", "注册人数": 28},
        # 缺失 20240604 的数据
        {"日期": "20240605", "入金": "210000.80", "注册人数": 35},
        {"日期": "20240606", "入金": "185000.20", "注册人数": 22},
    ]

    # 1. 构建单指标时间序列
    inflow_series = await builder.build_daily_time_series(sample_data, "入金")
    print(f"入金时间序列: {'成功' if inflow_series.get('success') else '失败'}")
    if inflow_series.get("success"):
        print(f"数据点数: {inflow_series['total_points']}")
        print(f"质量评分: {inflow_series['metadata'].confidence_score:.2f}")

    # 2. 构建多指标时间序列
    multi_series = await builder.build_multi_metric_series(sample_data, ["入金", "注册人数"])
    print(f"多指标序列: {'成功' if multi_series.get('success') else '失败'}")
    if multi_series.get("success"):
        print(f"成功指标: {multi_series['successful_metrics']}")
        print(f"相关性: {len(multi_series.get('correlations', {}))} 组")

    # 3. 构建聚合序列 (如果有足够数据)
    if inflow_series.get("success"):
        time_series = inflow_series["time_series"]
        if len(time_series) >= 7:
            weekly_series = await builder.build_aggregated_series(time_series, "weekly")
            print(f"周度聚合: {'成功' if weekly_series.get('success') else '失败'}")


if __name__ == "__main__":
    asyncio.run(main())