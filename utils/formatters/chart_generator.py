# utils/formatters/chart_generator.py - 完整增强版
"""
📊 智能图表生成工具 - AI驱动版本

专业的金融图表生成工具，集成AI智能分析：
- AI驱动的图表类型推荐
- 智能数据预处理和分析
- 多种图表类型支持
- 金融专业配色方案
- 自动添加标题、图例和注释
"""

import os
import json
import base64
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta
import io
import math
import random
import re
import traceback
import asyncio

# 图表库
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np

# 交互式图表支持
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 导入格式化工具
from utils.formatters.financial_formatter import FinancialFormatter, create_financial_formatter

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """图表类型枚举"""
    LINE = "line"  # 折线图
    BAR = "bar"  # 柱状图
    PIE = "pie"  # 饼图
    SCATTER = "scatter"  # 散点图
    AREA = "area"  # 面积图
    HEATMAP = "heatmap"  # 热力图
    RADAR = "radar"  # 雷达图
    CANDLESTICK = "candlestick"  # K线图
    BOXPLOT = "boxplot"  # 箱线图
    HISTOGRAM = "histogram"  # 直方图
    BUBBLE = "bubble"  # 气泡图
    COMBO = "combo"  # 组合图表
    SANKEY = "sankey"  # 桑基图
    DONUT = "donut"  # 环形图


class ChartTheme(Enum):
    """图表主题枚举"""
    DEFAULT = "default"  # 默认主题
    FINANCIAL = "financial"  # 金融主题
    MODERN = "modern"  # 现代主题
    MINIMAL = "minimal"  # 极简主题
    DARK = "dark"  # 暗色主题
    COLORFUL = "colorful"  # 多彩主题


class OutputFormat(Enum):
    """输出格式枚举"""
    PNG = "png"  # PNG图片
    SVG = "svg"  # SVG矢量图
    PDF = "pdf"  # PDF文档
    HTML = "html"  # HTML网页
    JSON = "json"  # JSON数据
    BASE64 = "base64"  # Base64编码


class ChartGenerator:
    """
    📊 智能图表生成器 - AI驱动版本

    自动生成专业金融图表，支持AI智能分析和推荐
    """

    def __init__(self, theme: ChartTheme = ChartTheme.FINANCIAL,
                 interactive: bool = True,
                 formatter: FinancialFormatter = None,
                 claude_client=None):
        """
        初始化图表生成器

        Args:
            theme: 图表主题
            interactive: 是否生成交互式图表
            formatter: 金融数据格式化工具
            claude_client: AI客户端，用于智能分析
        """
        self.theme = theme
        self.interactive = interactive and PLOTLY_AVAILABLE
        self.formatter = formatter or create_financial_formatter()
        self.claude_client = claude_client  # 🆕 AI客户端

        # 设置图表样式
        self._setup_chart_style()

        # 图表配色方案
        self.color_palettes = {
            ChartTheme.DEFAULT: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                 '#bcbd22', '#17becf'],
            ChartTheme.FINANCIAL: ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00',
                                   '#999999'],
            ChartTheme.MODERN: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#7f8c8d'],
            ChartTheme.MINIMAL: ['#555555', '#777777', '#999999', '#bbbbbb', '#dddddd', '#333333', '#aaaaaa',
                                 '#cccccc'],
            ChartTheme.DARK: ['#c1e7ff', '#ffcf9e', '#a5ffd6', '#ffb1b1', '#d8c4ff', '#ffe0c2', '#ffd1ec', '#e0e0e0'],
            ChartTheme.COLORFUL: ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF5', '#F5FF33', '#FF33F5',
                                  '#33F5FF']
        }

        # 图表默认尺寸
        self.default_sizes = {
            ChartType.LINE: (10, 6),
            ChartType.BAR: (10, 6),
            ChartType.PIE: (8, 8),
            ChartType.SCATTER: (10, 8),
            ChartType.AREA: (10, 6),
            ChartType.HEATMAP: (10, 8),
            ChartType.RADAR: (8, 8),
            ChartType.CANDLESTICK: (12, 8),
            ChartType.BOXPLOT: (10, 6),
            ChartType.HISTOGRAM: (10, 6),
            ChartType.BUBBLE: (10, 8),
            ChartType.COMBO: (12, 8),
            ChartType.SANKEY: (12, 8),
            ChartType.DONUT: (8, 8)
        }

    def _setup_chart_style(self):
        """设置图表样式"""
        # Matplotlib样式设置
        if self.theme == ChartTheme.DARK:
            plt.style.use('dark_background')
        elif self.theme == ChartTheme.MINIMAL:
            plt.style.use('seaborn-whitegrid')
        elif self.theme == ChartTheme.FINANCIAL:
            plt.style.use('ggplot')
        else:
            plt.style.use('default')

        # 设置中文字体支持（如果有）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            logger.warning("无法设置中文字体，可能导致中文显示问题")

    def _get_color_palette(self, num_colors: int = None) -> List[str]:
        """获取当前主题的配色方案"""
        palette = self.color_palettes.get(self.theme, self.color_palettes[ChartTheme.DEFAULT])
        if num_colors and num_colors > len(palette):
            # 如果需要更多颜色，使用循环
            return palette * (num_colors // len(palette) + 1)
        return palette

    # =================== 🆕 智能分析核心方法 ===================

    async def intelligent_chart_generation(self, extracted_data: Dict[str, Any],
                                           user_query: str = "",
                                           auto_select: bool = True) -> Dict[str, Any]:
        """
        🤖 智能图表生成 - 主入口方法

        Args:
            extracted_data: 从编排器传来的提取数据
            user_query: 用户原始查询
            auto_select: 是否自动选择图表类型

        Returns:
            Dict[str, Any]: 生成的图表结果
        """

        try:
            logger.info("🎨 开始智能图表生成流程...")

            # 步骤1: 准备多维度图表数据
            chart_data = self._prepare_intelligent_chart_data(extracted_data)

            if not self._has_valid_chart_data(chart_data):
                return {
                    'success': False,
                    'error': '没有找到适合生成图表的数据',
                    'chart_data_available': list(chart_data.keys()),
                    'extracted_data_keys': list(extracted_data.keys())
                }

            # 步骤2: AI智能推荐图表类型
            recommendation = await self._ai_recommend_chart_type(user_query, chart_data, extracted_data)

            # 步骤3: 生成推荐的图表
            generated_charts = []

            if recommendation.get('success'):
                primary_chart = await self._generate_primary_chart(recommendation, chart_data)
                if primary_chart.get('success'):
                    generated_charts.append(primary_chart)

                # 生成辅助图表（如果推荐）
                if recommendation.get('generate_secondary', False):
                    secondary_chart = await self._generate_secondary_chart(recommendation, chart_data)
                    if secondary_chart.get('success'):
                        generated_charts.append(secondary_chart)
            else:
                # 降级：使用规则生成图表
                fallback_chart = self._generate_fallback_chart(chart_data, user_query)
                if fallback_chart.get('success'):
                    generated_charts.append(fallback_chart)

            result = {
                'success': len(generated_charts) > 0,
                'generated_charts': generated_charts,
                'chart_count': len(generated_charts),
                'recommendation_used': recommendation.get('success', False),
                'ai_reasoning': recommendation.get('reasoning', '使用默认规则'),
                'data_analysis': chart_data.get('metadata', {}),
                'generation_method': 'ai_intelligent' if recommendation.get('success') else 'rule_based'
            }

            logger.info(f"✅ 智能图表生成完成: {len(generated_charts)} 个图表")
            return result

        except Exception as e:
            logger.error(f"❌ 智能图表生成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_method': 'failed'
            }

    def _prepare_intelligent_chart_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备多维度图表数据 - 修复数据类型处理问题"""

        chart_data = {
            'time_series': None,
            'comparison': None,
            'distribution': None,
            'financial_breakdown': None,
            'reinvestment_flow': None,
            'portfolio_composition': None,
            'trend_analysis': None,
            'auto': None,
            'metadata': {}
        }

        try:
            logger.info("📊 开始准备多维度图表数据...")

            # =================== 1. 复投计算专用数据（修复版） ===================
            detailed_calculations = extracted_data.get('detailed_calculations', {})
            if detailed_calculations:
                logger.info("💰 处理复投计算数据...")

                # 🔧 修复：复投分配饼图数据 - 确保标签正确
                if '复投分配' in detailed_calculations:
                    reinvest_data = detailed_calculations['复投分配']
                    withdrawal_amount = self._safe_float_conversion(reinvest_data.get('withdrawal_amount', 0))
                    reinvestment_amount = self._safe_float_conversion(reinvest_data.get('reinvestment_amount', 0))

                    if withdrawal_amount > 0 or reinvestment_amount > 0:
                        chart_data['distribution'] = pd.DataFrame([
                            {'类型': '提现金额', '金额': withdrawal_amount, '比例': 0.5},
                            {'类型': '复投金额', '金额': reinvestment_amount, '比例': 0.5}
                        ])

                        # 复投流向数据（桑基图用）
                        chart_data['reinvestment_flow'] = pd.DataFrame([
                            {'来源': '到期产品', '去向': '提现', '金额': withdrawal_amount},
                            {'来源': '到期产品', '去向': '复投', '金额': reinvestment_amount}
                        ])

                        chart_data['metadata']['reinvestment'] = {
                            'total_amount': withdrawal_amount + reinvestment_amount,
                            'withdrawal_rate': withdrawal_amount / (withdrawal_amount + reinvestment_amount) if (
                                                                                                                            withdrawal_amount + reinvestment_amount) > 0 else 0,
                            'reinvestment_rate': reinvestment_amount / (withdrawal_amount + reinvestment_amount) if (
                                                                                                                                withdrawal_amount + reinvestment_amount) > 0 else 0
                        }

                # 🔧 修复：到期金额结构分析 - 安全处理interest_earned字段
                if '到期金额' in detailed_calculations:
                    expiry_data = detailed_calculations['到期金额']
                    total_amount = self._safe_float_conversion(expiry_data.get('total_expiry_amount', 0))

                    # 🔧 修复：安全处理interest_earned字段
                    interest_earned_raw = expiry_data.get('interest_earned', 0)
                    logger.info(
                        f"🔍 [DEBUG] interest_earned原始值: {interest_earned_raw} (类型: {type(interest_earned_raw)})")

                    # 检查是否为有效的数字值
                    if self._is_valid_number(interest_earned_raw):
                        interest = self._safe_float_conversion(interest_earned_raw)
                        logger.info(f"✅ 使用实际利息数据: {interest}")
                    else:
                        # 如果不是数字，使用估算值
                        if total_amount > 0:
                            interest = total_amount * 0.1  # 假设10%的收益率
                            logger.info(f"⚠️ interest_earned为非数字值，使用估算利息: {interest} (总额的10%)")
                        else:
                            interest = 0
                            logger.warning("⚠️ 无法获取利息数据，设为0")

                    principal = total_amount - interest if total_amount > interest else total_amount

                    if total_amount > 0:
                        chart_data['financial_breakdown'] = pd.DataFrame([
                            {'项目': '本金回收', '金额': principal, '类型': '本金'},
                            {'项目': '利息收益', '金额': interest, '类型': '收益'}
                        ])
                        logger.info(f"✅ 金融分解数据: 本金={principal}, 利息={interest}")

            # =================== 2. 时间序列数据处理 ===================
            formatted_details = extracted_data.get('formatted_details', {})
            if 'daily_data_list' in formatted_details and formatted_details['daily_data_list']:
                logger.info("📈 处理时间序列数据...")
                daily_data = formatted_details['daily_data_list']

                # 构建时间序列DataFrame
                time_series_data = []
                for item in daily_data:
                    time_series_data.append({
                        '日期': item.get('date_formatted', item.get('date', '')),
                        '日期_原始': item.get('date', ''),
                        '净流入': item.get('net_flow', 0),
                        '入金': item.get('inflow', 0),
                        '出金': item.get('outflow', 0),
                        '注册人数': item.get('registrations', 0),
                        '购买数量': item.get('purchases', 0),
                        '持仓人数': item.get('holdings', 0)
                    })

                if time_series_data:
                    df_time = pd.DataFrame(time_series_data)
                    # 计算累计净流入
                    df_time['累计净流入'] = df_time['净流入'].cumsum()

                    chart_data['time_series'] = df_time
                    chart_data['metadata']['time_range'] = {
                        'start_date': df_time['日期_原始'].iloc[0] if len(df_time) > 0 else None,
                        'end_date': df_time['日期_原始'].iloc[-1] if len(df_time) > 0 else None,
                        'data_points': len(df_time)
                    }

            # =================== 3. 对比分析数据 ===================
            comparison_analysis = extracted_data.get('comparison_analysis', {})
            if comparison_analysis:
                logger.info("📊 处理对比分析数据...")

                comparison_data = []
                for metric, analysis in comparison_analysis.items():
                    if isinstance(analysis, dict):
                        comparison_data.append({
                            '指标': metric,
                            '当前值': analysis.get('current_value', 0),
                            '对比值': analysis.get('baseline_value', 0),
                            '变化金额': analysis.get('absolute_change', 0),
                            '变化率': analysis.get('percentage_change', 0),
                            '变化方向': analysis.get('change_direction', '持平')
                        })

                if comparison_data:
                    chart_data['comparison'] = pd.DataFrame(comparison_data)

            # =================== 4. 基础指标分布 ===================
            extracted_metrics = extracted_data.get('extracted_metrics', {})
            if extracted_metrics and not chart_data['distribution']:  # 如果还没有分布数据
                logger.info("📈 处理基础指标分布...")

                # 选择适合饼图的指标
                suitable_metrics = {}
                for key, value in extracted_metrics.items():
                    if isinstance(value, (int, float)) and value > 0:
                        if any(word in key for word in ['金额', '余额', '入金', '出金', '投资']):
                            suitable_metrics[key] = value

                if len(suitable_metrics) >= 2:
                    distribution_data = [
                        {'指标': key, '数值': value}
                        for key, value in suitable_metrics.items()
                    ]
                    chart_data['distribution'] = pd.DataFrame(distribution_data)

            # =================== 5. 投资组合构成分析 ===================
            raw_api_results = extracted_data.get('raw_api_results', {})
            if raw_api_results:
                logger.info("💼 处理投资组合数据...")

                # 查找产品数据
                for key, result in raw_api_results.items():
                    if 'product' in key.lower() and result.get('success'):
                        product_data = result.get('data', {})
                        if '产品列表' in product_data:
                            products = product_data['产品列表']
                            if products:
                                portfolio_data = []
                                for product in products[:10]:  # 最多显示10个产品
                                    portfolio_data.append({
                                        '产品名称': product.get('产品名称', '未知'),
                                        '投资金额': float(product.get('总投资金额', 0)),
                                        '持有数量': int(product.get('当前持有数', 0)),
                                        '收益率': float(product.get('预期年化收益率', 0))
                                    })

                                if portfolio_data:
                                    chart_data['portfolio_composition'] = pd.DataFrame(portfolio_data)
                            break

            # =================== 6. 趋势分析数据 ===================
            if chart_data['time_series'] is not None and len(chart_data['time_series']) > 3:
                logger.info("📈 生成趋势分析数据...")
                df = chart_data['time_series']

                # 计算移动平均
                if len(df) >= 3:
                    df_trend = df.copy()
                    df_trend['净流入_3日均线'] = df_trend['净流入'].rolling(window=3, min_periods=1).mean()
                    if len(df) >= 7:
                        df_trend['净流入_7日均线'] = df_trend['净流入'].rolling(window=7, min_periods=1).mean()

                    chart_data['trend_analysis'] = df_trend

            # =================== 7. 智能数据选择 ===================
            # 根据数据丰富程度选择最佳展示数据
            data_priority = [
                ('reinvestment_flow', chart_data['reinvestment_flow']),  # 复投流向（最具业务意义）
                ('financial_breakdown', chart_data['financial_breakdown']),  # 金融分解
                ('comparison', chart_data['comparison']),  # 对比分析
                ('trend_analysis', chart_data['trend_analysis']),  # 趋势分析
                ('time_series', chart_data['time_series']),  # 时间序列
                ('portfolio_composition', chart_data['portfolio_composition']),  # 投资组合
                ('distribution', chart_data['distribution'])  # 基础分布
            ]

            # 选择第一个非空的数据作为自动数据
            for data_type, data_df in data_priority:
                if data_df is not None and len(data_df) > 0:
                    chart_data['auto'] = data_df
                    chart_data['metadata']['auto_selected_type'] = data_type
                    chart_data['metadata']['auto_selected_reason'] = f"选择{data_type}因为数据最丰富且业务相关性最高"
                    logger.info(f"🎯 自动选择图表数据类型: {data_type}")
                    break

            # =================== 8. 数据质量评估 ===================
            chart_data['metadata']['data_quality'] = {
                'has_time_series': chart_data['time_series'] is not None,
                'has_comparison': chart_data['comparison'] is not None,
                'has_distribution': chart_data['distribution'] is not None,
                'has_financial_breakdown': chart_data['financial_breakdown'] is not None,
                'has_reinvestment_data': chart_data['reinvestment_flow'] is not None,
                'total_data_types': sum(
                    1 for k, v in chart_data.items() if k != 'metadata' and isinstance(v, pd.DataFrame)),
                'recommended_chart_types': []
            }

            # 基于数据特征推荐图表类型
            if chart_data['reinvestment_flow'] is not None:
                chart_data['metadata']['data_quality']['recommended_chart_types'].extend(['sankey', 'pie', 'donut'])
            if chart_data['time_series'] is not None:
                chart_data['metadata']['data_quality']['recommended_chart_types'].extend(['line', 'area'])
            if chart_data['comparison'] is not None:
                chart_data['metadata']['data_quality']['recommended_chart_types'].extend(['bar', 'radar'])
            if chart_data['distribution'] is not None:
                chart_data['metadata']['data_quality']['recommended_chart_types'].extend(['pie', 'treemap'])

            logger.info(f"✅ 图表数据准备完成: {chart_data['metadata']['data_quality']['total_data_types']} 种数据类型")
            return chart_data

        except Exception as e:
            logger.error(f"❌ 准备图表数据时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")

            # 返回基础的空数据结构，避免完全失败
            return {
                'time_series': None,
                'comparison': None,
                'distribution': None,
                'auto': None,
                'metadata': {'error': str(e)}
            }

    def _safe_float_conversion(self, value: Any) -> float:
        """安全的浮点数转换"""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # 尝试清理字符串中的非数字字符
                cleaned = re.sub(r'[^\d.-]', '', str(value))
                if cleaned:
                    return float(cleaned)
                else:
                    return 0.0
            return 0.0
        except (ValueError, TypeError):
            logger.warning(f"⚠️ 无法转换为浮点数: {value} (类型: {type(value)})")
            return 0.0

    def _is_valid_number(self, value: Any) -> bool:
        """检查值是否为有效的数字"""
        try:
            if value is None:
                return False
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str):
                # 检查字符串是否主要包含数字
                cleaned = re.sub(r'[^\d.-]', '', str(value))
                if cleaned and len(cleaned) >= len(str(value)) * 0.5:  # 至少一半是数字字符
                    float(cleaned)
                    return True
            return False
        except (ValueError, TypeError):
            return False

    def _has_valid_chart_data(self, chart_data: Dict[str, Any]) -> bool:
        """检查是否有有效的图表数据 - 修复DataFrame检查"""
        for key, data in chart_data.items():
            if key != 'metadata':
                # 🔧 修复：正确检查DataFrame
                if isinstance(data, pd.DataFrame) and not data.empty:
                    return True
                # 🔧 也支持其他数据类型
                elif data is not None and not isinstance(data, pd.DataFrame):
                    return True
        return False


    async def _ai_recommend_chart_type(self, user_query: str,
                                       chart_data: Dict[str, Any],
                                       extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🤖 AI智能推荐图表类型

        Args:
            user_query: 用户查询
            chart_data: 准备好的图表数据
            extracted_data: 原始提取数据

        Returns:
            Dict[str, Any]: AI推荐结果
        """

        if not self.claude_client:
            logger.warning("Claude客户端不可用，跳过AI图表推荐")
            return {'success': False, 'reason': 'claude_unavailable'}

        try:
            # 分析可用数据类型
            available_data_types = [k for k, v in chart_data.items()
                                    if k != 'metadata' and v is not None and isinstance(v, pd.DataFrame) and len(v) > 0]

            if not available_data_types:
                return {'success': False, 'reason': 'no_data_available'}

            # 构建AI分析prompt
            chart_analysis_prompt = f"""
            作为专业的数据可视化专家，请为金融查询推荐最佳图表类型。

            **用户查询**: "{user_query}"

            **可用数据类型**: {available_data_types}

            **数据详情**:
            {self._format_data_for_ai_analysis(chart_data)}

            **图表推荐规则**:
            1. **复投/分配查询** → 饼图(pie) > 环形图(donut) > 桑基图(sankey)
            2. **趋势/时间查询** → 折线图(line) > 面积图(area) > 柱状图(bar)
            3. **对比查询** → 柱状图(bar) > 雷达图(radar) > 热力图(heatmap)
            4. **分布查询** → 饼图(pie) > 柱状图(bar)
            5. **金融结构查询** → 饼图(pie) > 柱状图(bar)

            请返回JSON格式推荐:
            {{
                "primary_recommendation": {{
                    "chart_type": "pie",
                    "data_source": "distribution",
                    "confidence": 0.9,
                    "title": "资金分配图",
                    "reasoning": "基于复投查询，推荐饼图展示资金分配"
                }},
                "secondary_recommendation": {{
                    "chart_type": "bar",
                    "data_source": "financial_breakdown",
                    "confidence": 0.7,
                    "title": "收益构成图"
                }},
                "generate_secondary": false,
                "chart_config": {{
                    "show_percentage": true,
                    "show_values": true,
                    "interactive": true
                }}
            }}

            图表类型: pie, bar, line, scatter, area, heatmap, radar, donut, sankey
            """

            result = await self.claude_client.generate_text(chart_analysis_prompt, max_tokens=1500)

            if result.get('success'):
                response_text = result.get('text', '{}')

                import json
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    ai_recommendation = json.loads(json_match.group())

                    primary = ai_recommendation.get('primary_recommendation', {})
                    chart_type = primary.get('chart_type')

                    logger.info(f"🤖 AI推荐图表: {chart_type} (置信度: {primary.get('confidence', 0)})")

                    return {
                        'success': True,
                        'primary_chart_type': chart_type,
                        'primary_data_source': primary.get('data_source'),
                        'primary_title': primary.get('title', '数据分析图'),
                        'reasoning': primary.get('reasoning', 'AI智能推荐'),
                        'secondary_recommendation': ai_recommendation.get('secondary_recommendation'),
                        'generate_secondary': ai_recommendation.get('generate_secondary', False),
                        'chart_config': ai_recommendation.get('chart_config', {}),
                        'confidence': primary.get('confidence', 0.8)
                    }

            return {'success': False, 'reason': 'parse_failed'}

        except Exception as e:
            logger.error(f"AI图表推荐异常: {e}")
            return {'success': False, 'reason': str(e)}

    def _format_data_for_ai_analysis(self, chart_data: Dict[str, Any]) -> str:
        """格式化数据信息供AI分析"""
        info_lines = []

        for data_type, data_df in chart_data.items():
            if data_type == 'metadata' or data_df is None:
                continue

            try:
                if isinstance(data_df, pd.DataFrame) and len(data_df) > 0:
                    columns = list(data_df.columns)
                    sample_data = data_df.head(2).to_dict('records') if len(data_df) > 0 else []
                    info_lines.append(f"- {data_type}: {len(data_df)}行, 字段: {columns}")
                    if sample_data:
                        info_lines.append(f"  示例: {sample_data[0]}")
                else:
                    info_lines.append(f"- {data_type}: 无有效数据")
            except Exception as e:
                info_lines.append(f"- {data_type}: 数据格式错误")

        return '\n'.join(info_lines) if info_lines else "无可用数据"

    async def _generate_primary_chart(self, recommendation: Dict[str, Any],
                                      chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成主图表 - 修复DataFrame or操作符问题"""
        try:
            chart_type = recommendation.get('primary_chart_type', 'pie')
            data_source = recommendation.get('primary_data_source', 'auto')
            title = recommendation.get('primary_title', '数据分析图')
            config = recommendation.get('chart_config', {})

            # 🔧 修复：分步获取数据，避免or操作符
            data_df = chart_data.get(data_source)
            if data_df is None or (isinstance(data_df, pd.DataFrame) and data_df.empty):
                data_df = chart_data.get('auto')

            # 🔧 修复：最终检查数据有效性
            if data_df is None or (isinstance(data_df, pd.DataFrame) and data_df.empty):
                return {'success': False, 'error': f'数据源 {data_source} 无效或为空'}

            logger.info(f"🎯 生成主图表: type={chart_type}, title={title}, data_source={data_source}")
            logger.info(
                f"🔍 [DEBUG] 数据形状: {data_df.shape if isinstance(data_df, pd.DataFrame) else 'non-dataframe'}")

            # 根据图表类型生成
            if chart_type == 'pie' or chart_type == 'donut':
                return self._generate_pie_chart_from_df(data_df, title, chart_type == 'donut', config)
            elif chart_type == 'bar':
                return self._generate_bar_chart_from_df(data_df, title, config)
            elif chart_type == 'line':
                return self._generate_line_chart_from_df(data_df, title, config)
            elif chart_type == 'area':
                return self._generate_area_chart_from_df(data_df, title, config)
            else:
                # 默认使用饼图
                return self._generate_pie_chart_from_df(data_df, title, False, config)

        except Exception as e:
            logger.error(f"生成主图表失败: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    async def _generate_secondary_chart(self, recommendation: Dict[str, Any],
                                        chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成辅助图表 - 修复DataFrame or操作符问题"""
        try:
            secondary = recommendation.get('secondary_recommendation', {})
            if not secondary:
                return {'success': False, 'error': '无辅助图表推荐'}

            chart_type = secondary.get('chart_type', 'bar')
            data_source = secondary.get('data_source', 'auto')
            title = secondary.get('title', '辅助分析图')

            # 🔧 修复：分步获取数据，避免or操作符
            data_df = chart_data.get(data_source)
            if data_df is None or (isinstance(data_df, pd.DataFrame) and data_df.empty):
                data_df = chart_data.get('auto')

            # 🔧 修复：最终检查数据有效性
            if data_df is None or (isinstance(data_df, pd.DataFrame) and data_df.empty):
                return {'success': False, 'error': f'辅助图表数据源 {data_source} 无效或为空'}

            logger.info(f"🎯 生成辅助图表: type={chart_type}, title={title}")

            # 生成辅助图表
            if chart_type == 'bar':
                return self._generate_bar_chart_from_df(data_df, title, {})
            elif chart_type == 'line':
                return self._generate_line_chart_from_df(data_df, title, {})
            else:
                return self._generate_bar_chart_from_df(data_df, title, {})

        except Exception as e:
            logger.error(f"生成辅助图表失败: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def _generate_fallback_chart(self, chart_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """生成降级图表 - 避免所有DataFrame布尔判断问题"""
        try:
            logger.info("📊 使用规则生成降级图表...")

            data_df = None
            chart_title = "数据分析图"

            # 🔧 修复：按优先级逐一检查，避免复杂的布尔判断
            # 优先级1: 复投分配数据
            distribution_df = chart_data.get('distribution')
            if distribution_df is not None and isinstance(distribution_df, pd.DataFrame) and not distribution_df.empty:
                data_df = distribution_df
                chart_title = "资金分配分析"
                logger.info("🎯 选择复投分配数据生成饼图")

            # 优先级2: 金融分解数据
            elif True:  # 使用elif继续检查
                financial_breakdown_df = chart_data.get('financial_breakdown')
                if financial_breakdown_df is not None and isinstance(financial_breakdown_df,
                                                                     pd.DataFrame) and not financial_breakdown_df.empty:
                    data_df = financial_breakdown_df
                    chart_title = "收益结构分析"
                    logger.info("🎯 选择金融分解数据生成饼图")

                # 优先级3: 复投流向数据
                elif True:  # 继续检查
                    reinvestment_flow_df = chart_data.get('reinvestment_flow')
                    if reinvestment_flow_df is not None and isinstance(reinvestment_flow_df,
                                                                       pd.DataFrame) and not reinvestment_flow_df.empty:
                        data_df = reinvestment_flow_df
                        chart_title = "资金流向分析"
                        logger.info("🎯 选择复投流向数据生成图表")

                    # 优先级4: 自动选择数据
                    elif True:  # 继续检查
                        auto_df = chart_data.get('auto')
                        if auto_df is not None and isinstance(auto_df, pd.DataFrame) and not auto_df.empty:
                            data_df = auto_df
                            chart_title = "自动数据分析"
                            logger.info("🎯 使用自动选择数据")

                        # 优先级5: 遍历查找任何可用数据
                        else:
                            for key, df in chart_data.items():
                                if (key != 'metadata' and
                                        df is not None and
                                        isinstance(df, pd.DataFrame) and
                                        not df.empty):
                                    data_df = df
                                    chart_title = f"{key}分析图"
                                    logger.info(f"🎯 降级选择 {key} 数据")
                                    break

            # 🔧 修复：最终检查数据有效性
            if data_df is None:
                return {'success': False, 'error': '无可用数据生成图表'}

            if isinstance(data_df, pd.DataFrame) and data_df.empty:
                return {'success': False, 'error': '选中的数据为空'}

            logger.info(f"✅ 最终选择数据: shape={data_df.shape}, columns={list(data_df.columns)}")

            # 基于查询关键词选择图表类型
            query_lower = user_query.lower()
            if any(word in query_lower for word in ['复投', '提现', '分配', '占比']):
                return self._generate_pie_chart_from_df(data_df, chart_title, False, {})
            elif any(word in query_lower for word in ['趋势', '变化', '时间']):
                return self._generate_line_chart_from_df(data_df, chart_title, {})
            elif any(word in query_lower for word in ['对比', '比较']):
                return self._generate_bar_chart_from_df(data_df, chart_title, {})
            else:
                # 默认：根据数据结构智能选择
                if len(data_df.columns) >= 2:
                    # 检查是否有类型/标签列
                    label_cols = [col for col in data_df.columns if data_df[col].dtype == 'object']
                    if label_cols:
                        return self._generate_pie_chart_from_df(data_df, chart_title, False, {})
                    else:
                        return self._generate_bar_chart_from_df(data_df, chart_title, {})
                else:
                    return self._generate_pie_chart_from_df(data_df, chart_title, False, {})

        except Exception as e:
            logger.error(f"生成降级图表失败: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    # 🔧 安全的数据获取工具函数
    def _safe_get_dataframe(self, chart_data: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
        """安全获取DataFrame，避免布尔判断问题"""
        data = chart_data.get(key)
        if data is None:
            return None
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data
        return None

    # 在 chart_generator.py 中修复 _generate_pie_chart_from_df 方法
    def _generate_pie_chart_from_df(self, data_df: pd.DataFrame, title: str,
                                    is_donut: bool = False, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """从DataFrame生成饼图 - 修复版"""
        try:
            config = config or {}

            # 🔧 智能选择合适的列
            label_col = None
            value_col = None

            # 查找标签列和数值列
            for col in data_df.columns:
                if data_df[col].dtype == 'object' or data_df[col].dtype.name == 'category':
                    if label_col is None:
                        label_col = col
                elif pd.api.types.is_numeric_dtype(data_df[col]):
                    if value_col is None:
                        value_col = col

            if label_col is None or value_col is None:
                if len(data_df.columns) >= 2:
                    label_col = data_df.columns[0]
                    value_col = data_df.columns[1]
                else:
                    return {'success': False, 'error': '数据格式不适合生成饼图'}

            # 🔧 确保标签唯一性
            unique_labels = []
            unique_values = []

            for _, row in data_df.iterrows():
                label = str(row[label_col])
                value = float(row[value_col])

                # 检查是否已存在相同标签
                if label in unique_labels:
                    # 如果存在，合并数值或修改标签
                    existing_index = unique_labels.index(label)
                    unique_values[existing_index] += value
                else:
                    unique_labels.append(label)
                    unique_values.append(value)

            logger.info(f"🔍 [DEBUG] 饼图数据: labels={unique_labels}, values={unique_values}")

            # 生成饼图
            result = self.generate_pie_chart(
                data=pd.DataFrame({label_col: unique_labels, value_col: unique_values}),
                title=title,
                label_column=label_col,
                value_column=value_col,
                donut=is_donut,
                show_percentage=config.get('show_percentage', True),
                config={'output_format': OutputFormat.BASE64}
            )

            if result.get('error'):
                return {'success': False, 'error': result['error']}

            return {
                'success': True,
                'chart_type': 'donut' if is_donut else 'pie',
                'title': title,
                'image_data': result.get('image_data'),
                'data_summary': {
                    'labels': unique_labels,
                    'values': unique_values
                }
            }

        except Exception as e:
            logger.error(f"生成饼图失败: {e}")
            return {'success': False, 'error': str(e)}

    # 修复剩余的DataFrame布尔判断问题

    def _generate_bar_chart_from_df(self, data_df: pd.DataFrame, title: str,
                                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """从DataFrame生成柱状图 - 修复布尔判断问题"""
        try:
            config = config or {}

            # 🔧 修复：确保数据有效性检查
            if data_df is None:
                return {'success': False, 'error': '数据为空，无法生成柱状图'}

            if isinstance(data_df, pd.DataFrame) and data_df.empty:
                return {'success': False, 'error': '数据DataFrame为空，无法生成柱状图'}

            # 智能选择列
            if len(data_df.columns) < 2:
                return {'success': False, 'error': '数据列数不足，无法生成柱状图'}

            x_col = data_df.columns[0]
            y_cols = [col for col in data_df.columns[1:] if pd.api.types.is_numeric_dtype(data_df[col])]

            if not y_cols:
                return {'success': False, 'error': '没有找到数值列'}

            result = self.generate_bar_chart(
                data=data_df,
                title=title,
                x_column=x_col,
                y_columns=y_cols[:3],  # 最多3个系列
                config={'output_format': OutputFormat.BASE64}
            )

            if result.get('error'):
                return {'success': False, 'error': result['error']}

            return {
                'success': True,
                'chart_type': 'bar',
                'title': title,
                'image_data': result.get('image_data'),
                'data_summary': {
                    'categories': data_df[x_col].tolist(),
                    'series': {col: data_df[col].tolist() for col in y_cols[:3]}
                }
            }

        except Exception as e:
            logger.error(f"生成柱状图失败: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def _generate_line_chart_from_df(self, data_df: pd.DataFrame, title: str,
                                     config: Dict[str, Any] = None) -> Dict[str, Any]:
        """从DataFrame生成折线图 - 修复布尔判断问题"""
        try:
            config = config or {}

            # 🔧 修复：确保数据有效性检查
            if data_df is None:
                return {'success': False, 'error': '数据为空，无法生成折线图'}

            if isinstance(data_df, pd.DataFrame) and data_df.empty:
                return {'success': False, 'error': '数据DataFrame为空，无法生成折线图'}

            # 智能选择列
            if len(data_df.columns) < 2:
                return {'success': False, 'error': '数据列数不足，无法生成折线图'}

            x_col = data_df.columns[0]
            y_cols = [col for col in data_df.columns[1:] if pd.api.types.is_numeric_dtype(data_df[col])]

            if not y_cols:
                return {'success': False, 'error': '没有找到数值列'}

            result = self.generate_line_chart(
                data=data_df,
                title=title,
                x_column=x_col,
                y_columns=y_cols[:3],  # 最多3个系列
                show_markers=True,
                config={'output_format': OutputFormat.BASE64}
            )

            if result.get('error'):
                return {'success': False, 'error': result['error']}

            return {
                'success': True,
                'chart_type': 'line',
                'title': title,
                'image_data': result.get('image_data'),
                'data_summary': {
                    'x_axis': data_df[x_col].tolist(),
                    'series': {col: data_df[col].tolist() for col in y_cols[:3]}
                }
            }

        except Exception as e:
            logger.error(f"生成折线图失败: {e}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def _generate_area_chart_from_df(self, data_df: pd.DataFrame, title: str,
                                     config: Dict[str, Any] = None) -> Dict[str, Any]:
        """从DataFrame生成面积图"""
        try:
            # 面积图本质上是带填充的折线图
            line_result = self._generate_line_chart_from_df(data_df, title, config)
            if line_result.get('success'):
                line_result['chart_type'] = 'area'
            return line_result

        except Exception as e:
            logger.error(f"生成面积图失败: {e}")
            return {'success': False, 'error': str(e)}

    # =================== 原有方法保持不变 ===================

    def _prepare_data(self, data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """准备图表数据"""
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # 处理不同的字典格式
            if 'x' in data and 'y' in data:
                df = pd.DataFrame({
                    'x': data['x'],
                    'y': data['y']
                })
                if 'categories' in data:
                    df['category'] = data['categories']
                return df

            elif 'data' in data:
                if isinstance(data['data'], list):
                    if all(isinstance(item, dict) for item in data['data']):
                        return pd.DataFrame(data['data'])
                    else:
                        return pd.DataFrame({'value': data['data']})

            elif 'labels' in data and 'values' in data:
                return pd.DataFrame({
                    'label': data['labels'],
                    'value': data['values']
                })

            else:
                return pd.DataFrame(data).reset_index().rename(columns={'index': 'category'})

        logger.warning(f"无法识别的数据格式: {type(data)}")
        return pd.DataFrame()

    def _detect_chart_type(self, data: pd.DataFrame, columns: List[str] = None) -> ChartType:
        """智能检测适合的图表类型"""
        if data.empty:
            return ChartType.BAR

        if not columns:
            columns = data.select_dtypes(include=['number']).columns.tolist()

        if len(columns) == 0:
            return ChartType.BAR

        # 检查是否有时间列
        date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]'
                     or (data[col].dtype == 'object' and pd.to_datetime(data[col], errors='coerce').notna().all())]

        num_rows = len(data)
        category_cols = [col for col in data.columns if col not in columns and col not in date_cols]
        num_categories = 0
        if category_cols and len(category_cols) > 0:
            num_categories = data[category_cols[0]].nunique()

        # 决策逻辑
        if date_cols and len(date_cols) > 0:
            if num_rows > 20:
                return ChartType.LINE
            else:
                return ChartType.BAR
        elif num_categories > 0:
            if num_categories <= 8 and len(columns) == 1:
                return ChartType.PIE
            elif num_categories > 10:
                return ChartType.LINE
            else:
                return ChartType.BAR
        elif len(columns) >= 2:
            corr = data[columns].corr().abs().iloc[0, 1] if len(columns) >= 2 else 0
            if 0.3 <= corr <= 0.9:
                return ChartType.SCATTER
            else:
                return ChartType.COMBO
        else:
            if num_rows > 50:
                return ChartType.HISTOGRAM
            else:
                return ChartType.BAR

    def _add_chart_annotations(self, fig, ax, title: str = None, subtitle: str = None,
                               x_label: str = None, y_label: str = None,
                               data: pd.DataFrame = None, chart_type: ChartType = None):
        """添加图表注释"""
        if title:
            ax.set_title(title, fontsize=14, pad=20)
            if subtitle:
                ax.text(0.5, 1.05, subtitle, transform=ax.transAxes,
                        ha='center', va='center', fontsize=10,
                        color='gray', style='italic')

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        if chart_type == ChartType.LINE and data is not None:
            for column in data.select_dtypes(include=['number']).columns:
                if column in data.columns and len(data) > 1:
                    start_val = data[column].iloc[0]
                    end_val = data[column].iloc[-1]
                    ax.annotate(f'{start_val:.2f}', xy=(0, start_val),
                                xytext=(-15, 0), textcoords='offset points',
                                ha='right', va='center', fontsize=8)
                    ax.annotate(f'{end_val:.2f}', xy=(len(data) - 1, end_val),
                                xytext=(15, 0), textcoords='offset points',
                                ha='left', va='center', fontsize=8)

        ax.grid(True, linestyle='--', alpha=0.7)

        if chart_type not in [ChartType.PIE, ChartType.HISTOGRAM]:
            ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.7)

        fig.tight_layout()

    def generate_line_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                            title: str = "趋势分析",
                            x_label: str = None,
                            y_label: str = None,
                            x_column: str = None,
                            y_columns: List[str] = None,
                            show_markers: bool = True,
                            show_area: bool = False,
                            config: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成折线图"""
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        if x_column is None:
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]'
                         or (df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all())]
            if date_cols:
                x_column = date_cols[0]
                df[x_column] = pd.to_datetime(df[x_column])
            else:
                x_column = df.columns[0]

        if y_columns is None:
            y_columns = df.select_dtypes(include=['number']).columns.tolist()
            if x_column in y_columns:
                y_columns.remove(x_column)

        if not y_columns:
            return {"error": "没有找到有效的Y轴数据列"}

        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.LINE])
        dpi = config.get('dpi', 100)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        colors = self._get_color_palette(len(y_columns))

        for i, y_col in enumerate(y_columns):
            if y_col in df.columns:
                marker = 'o' if show_markers else None
                if show_area:
                    ax.fill_between(df[x_column], df[y_col], alpha=0.2, color=colors[i % len(colors)])
                ax.plot(df[x_column], df[y_col], marker=marker, label=y_col,
                        color=colors[i % len(colors)], linewidth=2, markersize=4)

        if df[x_column].dtype == 'datetime64[ns]':
            date_format = '%Y-%m-%d' if (df[x_column].max() - df[x_column].min()).days > 365 else '%m-%d'
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(date_format))
            plt.xticks(rotation=45)

        self._add_chart_annotations(
            fig=fig,
            ax=ax,
            title=title,
            subtitle=None,
            x_label=x_label or x_column,
            y_label=y_label or "数值",
            data=df,
            chart_type=ChartType.LINE
        )

        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)

        return {
            "type": "line",
            "title": title,
            "data": {
                "x": df[x_column].tolist(),
                "y": {col: df[col].tolist() for col in y_columns if col in df.columns}
            },
            "image_data": img_data
        }

    def generate_bar_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                           title: str = "对比分析",
                           x_label: str = None,
                           y_label: str = None,
                           x_column: str = None,
                           y_columns: List[str] = None,
                           stacked: bool = False,
                           horizontal: bool = False,
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成柱状图"""
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        if x_column is None:
            x_column = df.columns[0]

        if y_columns is None:
            y_columns = df.select_dtypes(include=['number']).columns.tolist()
            if x_column in y_columns:
                y_columns.remove(x_column)

        if not y_columns:
            return {"error": "没有找到有效的Y轴数据列"}

        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.BAR])
        dpi = config.get('dpi', 100)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        colors = self._get_color_palette(len(y_columns))
        width = 0.8 / len(y_columns) if not stacked else 0.8

        for i, y_col in enumerate(y_columns):
            if y_col in df.columns:
                x_pos = range(len(df))
                if not stacked and len(y_columns) > 1:
                    x_pos = [x + width * i for x in range(len(df))]

                if horizontal:
                    if stacked:
                        bottom = df[y_columns[:i]].sum(axis=1) if i > 0 else 0
                        ax.barh(x_pos, df[y_col], height=width, left=bottom,
                                label=y_col, color=colors[i % len(colors)])
                    else:
                        ax.barh(x_pos, df[y_col], height=width,
                                label=y_col, color=colors[i % len(colors)])
                else:
                    if stacked:
                        bottom = df[y_columns[:i]].sum(axis=1) if i > 0 else 0
                        ax.bar(x_pos, df[y_col], width=width, bottom=bottom,
                               label=y_col, color=colors[i % len(colors)])
                    else:
                        ax.bar(x_pos, df[y_col], width=width,
                               label=y_col, color=colors[i % len(colors)])

        if horizontal:
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df[x_column])
        else:
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x_column], rotation=45 if len(df) > 5 else 0)

        if len(df) <= 10 and len(y_columns) <= 3:
            for i, y_col in enumerate(y_columns):
                if y_col in df.columns:
                    x_pos = range(len(df))
                    if not stacked and len(y_columns) > 1:
                        x_pos = [x + width * i for x in range(len(df))]

                    for j, value in enumerate(df[y_col]):
                        if horizontal:
                            ax.text(value + 0.1, x_pos[j], f'{value:.1f}',
                                    va='center', ha='left', fontsize=8)
                        else:
                            ax.text(x_pos[j], value + 0.1, f'{value:.1f}',
                                    ha='center', va='bottom', fontsize=8)

        self._add_chart_annotations(
            fig=fig,
            ax=ax,
            title=title,
            subtitle=None,
            x_label=x_label or x_column if not horizontal else y_label or "数值",
            y_label=y_label or "数值" if not horizontal else x_label or x_column,
            data=df,
            chart_type=ChartType.BAR
        )

        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)

        return {
            "type": "bar",
            "title": title,
            "data": {
                "categories": df[x_column].tolist(),
                "series": {col: df[col].tolist() for col in y_columns if col in df.columns}
            },
            "stacked": stacked,
            "horizontal": horizontal,
            "image_data": img_data
        }

    def generate_pie_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                           title: str = "占比分析",
                           label_column: str = None,
                           value_column: str = None,
                           show_percentage: bool = True,
                           donut: bool = False,
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成饼图 - 修复matplotlib线程警告"""

        # 🔧 修复matplotlib线程问题
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端

        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        if label_column is None:
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                label_column = non_numeric_cols[0]
            else:
                label_column = df.columns[0]

        if value_column is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                value_column = numeric_cols[0]
            else:
                value_column = 'count'
                df[value_column] = 1

        if label_column == value_column:
            return {"error": "标签列和数值列不能相同"}

        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.PIE])
        dpi = config.get('dpi', 100)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        colors = self._get_color_palette(len(df))
        total = df[value_column].sum()

        if donut:
            wedges, texts, autotexts = ax.pie(
                df[value_column],
                labels=None,
                autopct='%1.1f%%' if show_percentage else None,
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.5)
            )
        else:
            wedges, texts, autotexts = ax.pie(
                df[value_column],
                labels=None,
                autopct='%1.1f%%' if show_percentage else None,
                startangle=90,
                colors=colors
            )

        if show_percentage:
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')

        legend_labels = [f"{row[label_column]} ({row[value_column]:.1f}, {row[value_column] / total * 100:.1f}%)"
                         for _, row in df.iterrows()]
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax.set_title(title, fontsize=14, pad=20)
        ax.axis('equal')

        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)

        return {
            "type": "pie",
            "title": title,
            "data": {
                "labels": df[label_column].tolist(),
                "values": df[value_column].tolist(),
                "percentages": (df[value_column] / total * 100).tolist()
            },
            "donut": donut,
            "image_data": img_data
        }

    def generate_scatter_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                               title: str = "相关性分析",
                               x_label: str = None,
                               y_label: str = None,
                               x_column: str = None,
                               y_column: str = None,
                               category_column: str = None,
                               show_trend_line: bool = True,
                               show_correlation: bool = True,
                               config: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成散点图"""
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return {"error": "散点图需要至少两个数值列"}

        if x_column is None:
            x_column = numeric_cols[0]

        if y_column is None:
            for col in numeric_cols:
                if col != x_column:
                    y_column = col
                    break

        if y_column is None:
            return {"error": "没有找到合适的Y轴数据列"}

        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.SCATTER])
        dpi = config.get('dpi', 100)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if category_column and category_column in df.columns:
            categories = df[category_column].unique()
            colors = self._get_color_palette(len(categories))

            for i, category in enumerate(categories):
                category_data = df[df[category_column] == category]
                ax.scatter(category_data[x_column], category_data[y_column],
                           label=category, color=colors[i % len(colors)],
                           alpha=0.7, s=50)

                if show_trend_line and len(category_data) >= 2:
                    try:
                        z = np.polyfit(category_data[x_column], category_data[y_column], 1)
                        p = np.poly1d(z)
                        ax.plot(category_data[x_column], p(category_data[x_column]),
                                '--', color=colors[i % len(colors)], alpha=0.5)
                    except:
                        logger.warning(f"无法为分类 {category} 添加趋势线")
        else:
            ax.scatter(df[x_column], df[y_column], alpha=0.7, s=50,
                       color=self._get_color_palette()[0])

            if show_trend_line and len(df) >= 2:
                try:
                    z = np.polyfit(df[x_column], df[y_column], 1)
                    p = np.poly1d(z)
                    ax.plot(df[x_column], p(df[x_column]), '--',
                            color='red', alpha=0.5)

                    eq_text = f'y = {z[0]:.2f}x + {z[1]:.2f}'
                    ax.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=9, ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                except:
                    logger.warning("无法添加趋势线")

        corr = None
        if show_correlation and not category_column:
            try:
                corr = df[[x_column, y_column]].corr().iloc[0, 1]
                corr_text = f'相关系数: {corr:.2f}'
                ax.annotate(corr_text, xy=(0.05, 0.85), xycoords='axes fraction',
                            fontsize=9, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.3))
            except:
                logger.warning("无法计算相关系数")

        self._add_chart_annotations(
            fig=fig,
            ax=ax,
            title=title,
            subtitle=None,
            x_label=x_label or x_column,
            y_label=y_label or y_column,
            data=df,
            chart_type=ChartType.SCATTER
        )

        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)

        return {
            "type": "scatter",
            "title": title,
            "data": {
                "x": df[x_column].tolist(),
                "y": df[y_column].tolist(),
                "categories": df[category_column].tolist() if category_column in df.columns else None
            },
            "correlation": corr if show_correlation and not category_column else None,
            "image_data": img_data
        }

    def generate_heatmap(self, data: Union[Dict[str, Any], pd.DataFrame],
                         title: str = "热力图分析",
                         x_label: str = None,
                         y_label: str = None,
                         value_column: str = None,
                         x_column: str = None,
                         y_column: str = None,
                         config: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成热力图"""
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.HEATMAP])
        dpi = config.get('dpi', 100)

        if len(df.columns) < 3 and value_column is None:
            pivot_data = df
        else:
            if not x_column:
                x_column = df.columns[0]
            if not y_column:
                for col in df.columns:
                    if col != x_column:
                        y_column = col
                        break
            if not value_column:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    for col in numeric_cols:
                        if col != x_column and col != y_column:
                            value_column = col
                            break
                    if not value_column:
                        value_column = numeric_cols[0]
                else:
                    return {"error": "没有找到合适的数值列"}

            try:
                pivot_data = df.pivot(index=y_column, columns=x_column, values=value_column)
            except:
                return {"error": "无法创建热力图所需的数据透视表"}

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.heatmap(pivot_data, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap='YlOrRd')

        ax.set_title(title, fontsize=14, pad=20)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        plt.tight_layout()

        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)

        return {
            "type": "heatmap",
            "title": title,
            "data": pivot_data.to_dict(),
            "image_data": img_data
        }

    def generate_radar_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                             title: str = "多维度分析",
                             categories: List[str] = None,
                             series_column: str = None,
                             config: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成雷达图"""
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.RADAR])
        dpi = config.get('dpi', 100)

        if categories is None:
            categories = df.select_dtypes(include=['number']).columns.tolist()

        if not categories:
            return {"error": "没有找到合适的雷达图维度"}

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        colors = self._get_color_palette()

        if series_column and series_column in df.columns:
            series = df[series_column].unique()

            for i, s in enumerate(series):
                series_data = df[df[series_column] == s]
                values = [series_data[cat].iloc[0] if cat in series_data else 0 for cat in categories]
                values += values[:1]

                ax.plot(angles, values, 'o-', linewidth=2, label=s, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        else:
            for i, row in df.iterrows():
                values = [row[cat] if cat in row else 0 for cat in categories]
                values += values[:1]

                label = f"系列 {i + 1}" if i < 5 else None
                ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

                if i >= 9:
                    break

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        ax.set_title(title, fontsize=14, pad=20)

        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()

        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)

        return {
            "type": "radar",
            "title": title,
            "data": {
                "categories": categories,
                "series": df.to_dict('records')
            },
            "image_data": img_data
        }

    def _fig_to_image(self, fig: Figure, output_format: OutputFormat = OutputFormat.PNG) -> Dict[str, Any]:
        """将matplotlib图表转换为图像数据"""
        result = {}

        if output_format == OutputFormat.BASE64:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
            buf.seek(0)
            result["base64"] = base64.b64encode(buf.read()).decode('utf-8')
            result["format"] = "png"

        elif output_format == OutputFormat.SVG:
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            result["svg"] = buf.read().decode('utf-8')
            result["format"] = "svg"

        elif output_format == OutputFormat.HTML:
            if PLOTLY_AVAILABLE and hasattr(fig, 'to_html'):
                result["html"] = fig.to_html(include_plotlyjs='cdn')
            else:
                buf = io.BytesIO()
                fig.savefig(buf, format='svg', bbox_inches='tight')
                buf.seek(0)
                svg_data = buf.read().decode('utf-8')
                result["html"] = f"<div>{svg_data}</div>"
            result["format"] = "html"

        else:  # PNG或其他格式
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
            buf.seek(0)
            result["binary"] = buf.read()
            result["format"] = "png"

        return result

    def generate_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                       chart_type: ChartType = None,
                       title: str = "数据分析图表",
                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """智能生成图表"""
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}

        if chart_type is None:
            chart_type = self._detect_chart_type(df)

        if chart_type == ChartType.LINE:
            return self.generate_line_chart(df, title=title, config=config)
        elif chart_type == ChartType.BAR:
            return self.generate_bar_chart(df, title=title, config=config)
        elif chart_type == ChartType.PIE:
            return self.generate_pie_chart(df, title=title, config=config)
        elif chart_type == ChartType.SCATTER:
            return self.generate_scatter_chart(df, title=title, config=config)
        elif chart_type == ChartType.HEATMAP:
            return self.generate_heatmap(df, title=title, config=config)
        elif chart_type == ChartType.RADAR:
            return self.generate_radar_chart(df, title=title, config=config)
        else:
            return self.generate_bar_chart(df, title=title, config=config)

    def save_chart(self, chart_result: Dict[str, Any], filename: str,
                   output_format: OutputFormat = OutputFormat.PNG) -> str:
        """保存图表到文件"""
        if "error" in chart_result:
            logger.error(f"无法保存图表: {chart_result['error']}")
            return None

        if not filename.endswith(f".{output_format.value}"):
            filename = f"{filename}.{output_format.value}"

        img_data = chart_result.get("image_data", {})

        try:
            if output_format == OutputFormat.BASE64:
                with open(filename, 'w') as f:
                    f.write(img_data.get("base64", ""))
            elif output_format == OutputFormat.SVG:
                with open(filename, 'w') as f:
                    f.write(img_data.get("svg", ""))
            elif output_format == OutputFormat.HTML:
                with open(filename, 'w') as f:
                    f.write(img_data.get("html", ""))
            else:
                with open(filename, 'wb') as f:
                    f.write(img_data.get("binary", b""))

            logger.info(f"图表已保存到: {filename}")
            return filename

        except Exception as e:
            logger.error(f"保存图表时出错: {str(e)}")
            return None


# 工厂函数
def create_chart_generator(theme: ChartTheme = ChartTheme.FINANCIAL,
                           interactive: bool = True,
                           formatter: FinancialFormatter = None,
                           claude_client=None) -> ChartGenerator:
    """创建智能图表生成器"""
    return ChartGenerator(theme, interactive, formatter, claude_client)