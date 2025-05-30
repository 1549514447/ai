# utils/formatters/chart_generator.py
"""
📊 智能图表生成工具

专业的金融图表生成工具，支持多种常见图表类型：
- 折线图（趋势分析）
- 柱状图（对比分析）
- 饼图（占比分析）
- 散点图（相关性分析）
- 热力图（密度分析）
- 雷达图（多维度分析）
- 组合图表（复合分析）

核心特点:
- 智能数据分析与图表推荐
- 金融专业配色方案
- 自动添加标题、图例和注释
- 支持交互式和静态图表
- 多种导出格式（PNG, SVG, HTML等）
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
    LINE = "line"                  # 折线图
    BAR = "bar"                    # 柱状图
    PIE = "pie"                    # 饼图
    SCATTER = "scatter"            # 散点图
    AREA = "area"                  # 面积图
    HEATMAP = "heatmap"            # 热力图
    RADAR = "radar"                # 雷达图
    CANDLESTICK = "candlestick"    # K线图
    BOXPLOT = "boxplot"            # 箱线图
    HISTOGRAM = "histogram"        # 直方图
    BUBBLE = "bubble"              # 气泡图
    COMBO = "combo"                # 组合图表


class ChartTheme(Enum):
    """图表主题枚举"""
    DEFAULT = "default"            # 默认主题
    FINANCIAL = "financial"        # 金融主题
    MODERN = "modern"              # 现代主题
    MINIMAL = "minimal"            # 极简主题
    DARK = "dark"                  # 暗色主题
    COLORFUL = "colorful"          # 多彩主题


class OutputFormat(Enum):
    """输出格式枚举"""
    PNG = "png"                    # PNG图片
    SVG = "svg"                    # SVG矢量图
    PDF = "pdf"                    # PDF文档
    HTML = "html"                  # HTML网页
    JSON = "json"                  # JSON数据
    BASE64 = "base64"              # Base64编码


class ChartGenerator:
    """
    📊 智能图表生成器
    
    自动生成专业金融图表，支持多种图表类型和主题
    """
    
    def __init__(self, theme: ChartTheme = ChartTheme.FINANCIAL, 
                interactive: bool = True, 
                formatter: FinancialFormatter = None):
        """
        初始化图表生成器
        
        Args:
            theme: 图表主题
            interactive: 是否生成交互式图表
            formatter: 金融数据格式化工具
        """
        self.theme = theme
        self.interactive = interactive and PLOTLY_AVAILABLE
        self.formatter = formatter or create_financial_formatter()
        
        # 设置图表样式
        self._setup_chart_style()
        
        # 图表配色方案
        self.color_palettes = {
            ChartTheme.DEFAULT: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            ChartTheme.FINANCIAL: ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00', '#999999'],
            ChartTheme.MODERN: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#7f8c8d'],
            ChartTheme.MINIMAL: ['#555555', '#777777', '#999999', '#bbbbbb', '#dddddd', '#333333', '#aaaaaa', '#cccccc'],
            ChartTheme.DARK: ['#c1e7ff', '#ffcf9e', '#a5ffd6', '#ffb1b1', '#d8c4ff', '#ffe0c2', '#ffd1ec', '#e0e0e0'],
            ChartTheme.COLORFUL: ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF5', '#F5FF33', '#FF33F5', '#33F5FF']
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
            ChartType.COMBO: (12, 8)
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
    
    def _prepare_data(self, data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        准备图表数据
        
        Args:
            data: 输入数据，可以是字典、DataFrame或字典列表
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        
        if isinstance(data, dict):
            # 处理不同的字典格式
            if 'x' in data and 'y' in data:
                # {x: [...], y: [...]} 格式
                df = pd.DataFrame({
                    'x': data['x'],
                    'y': data['y']
                })
                if 'categories' in data:
                    df['category'] = data['categories']
                return df
            
            elif 'data' in data:
                # {data: [...]} 格式
                if isinstance(data['data'], list):
                    if all(isinstance(item, dict) for item in data['data']):
                        return pd.DataFrame(data['data'])
                    else:
                        # 可能是简单数组
                        return pd.DataFrame({'value': data['data']})
            
            elif 'labels' in data and 'values' in data:
                # {labels: [...], values: [...]} 格式 (饼图)
                return pd.DataFrame({
                    'label': data['labels'],
                    'value': data['values']
                })
            
            else:
                # 尝试将字典的键作为索引
                return pd.DataFrame(data).reset_index().rename(columns={'index': 'category'})
        
        # 如果无法识别格式，返回空DataFrame
        logger.warning(f"无法识别的数据格式: {type(data)}")
        return pd.DataFrame()
    
    def _detect_chart_type(self, data: pd.DataFrame, columns: List[str] = None) -> ChartType:
        """
        智能检测适合的图表类型
        
        Args:
            data: 数据DataFrame
            columns: 要分析的列名列表
            
        Returns:
            ChartType: 推荐的图表类型
        """
        if data.empty:
            return ChartType.BAR  # 默认
            
        # 使用所有数值列
        if not columns:
            columns = data.select_dtypes(include=['number']).columns.tolist()
            
        if len(columns) == 0:
            return ChartType.BAR  # 没有数值列，默认柱状图
            
        # 检查是否有时间列
        date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]' 
                    or (data[col].dtype == 'object' and pd.to_datetime(data[col], errors='coerce').notna().all())]
        
        # 检查数据点数量
        num_rows = len(data)
        
        # 检查类别数量
        category_cols = [col for col in data.columns if col not in columns and col not in date_cols]
        num_categories = 0
        if category_cols and len(category_cols) > 0:
            num_categories = data[category_cols[0]].nunique()
        
        # 决策逻辑
        if date_cols and len(date_cols) > 0:
            # 有时间列，可能是时间序列
            if num_rows > 20:
                return ChartType.LINE  # 数据点多，用折线图
            else:
                return ChartType.BAR  # 数据点少，用柱状图
                
        elif num_categories > 0:
            # 有类别列
            if num_categories <= 8 and len(columns) == 1:
                # 类别少且只有一个数值列，考虑饼图
                return ChartType.PIE
            elif num_categories > 10:
                # 类别多，用折线图
                return ChartType.LINE
            else:
                # 类别适中，用柱状图
                return ChartType.BAR
                
        elif len(columns) >= 2:
            # 多个数值列，考虑散点图或组合图
            # 检查相关性
            corr = data[columns].corr().abs().iloc[0, 1]
            if 0.3 <= corr <= 0.9:
                return ChartType.SCATTER  # 中等相关性，用散点图
            else:
                return ChartType.COMBO  # 低或高相关性，用组合图
        
        else:
            # 单个数值列，无明显类别
            if num_rows > 50:
                return ChartType.HISTOGRAM  # 数据点多，用直方图
            else:
                return ChartType.BAR  # 数据点少，用柱状图
    
    def _add_chart_annotations(self, fig, ax, title: str = None, subtitle: str = None, 
                             x_label: str = None, y_label: str = None, 
                             data: pd.DataFrame = None, chart_type: ChartType = None):
        """添加图表注释"""
        # 设置标题和副标题
        if title:
            ax.set_title(title, fontsize=14, pad=20)
            if subtitle:
                ax.text(0.5, 1.05, subtitle, transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       color='gray', style='italic')
        
        # 设置轴标签
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
            
        # 根据图表类型添加特定注释
        if chart_type == ChartType.LINE and data is not None:
            # 为折线图添加起点和终点标记
            for column in data.select_dtypes(include=['number']).columns:
                if column in data.columns and len(data) > 1:
                    start_val = data[column].iloc[0]
                    end_val = data[column].iloc[-1]
                    ax.annotate(f'{start_val:.2f}', xy=(0, start_val), 
                               xytext=(-15, 0), textcoords='offset points',
                               ha='right', va='center', fontsize=8)
                    ax.annotate(f'{end_val:.2f}', xy=(len(data)-1, end_val), 
                               xytext=(15, 0), textcoords='offset points',
                               ha='left', va='center', fontsize=8)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        if chart_type not in [ChartType.PIE, ChartType.HISTOGRAM]:
            ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.7)
            
        # 调整布局
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
        """
        生成折线图
        
        Args:
            data: 图表数据
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            x_column: X轴数据列名
            y_columns: Y轴数据列名列表
            show_markers: 是否显示数据点标记
            show_area: 是否显示面积
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
            
        # 确定X轴和Y轴
        if x_column is None:
            # 尝试找到日期列或第一列作为X轴
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' 
                        or (df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all())]
            if date_cols:
                x_column = date_cols[0]
                # 转换为日期类型
                df[x_column] = pd.to_datetime(df[x_column])
            else:
                x_column = df.columns[0]
        
        # 如果没有指定Y轴列，使用所有数值列
        if y_columns is None:
            y_columns = df.select_dtypes(include=['number']).columns.tolist()
            # 如果X轴也是数值列，从Y轴列表中移除
            if x_column in y_columns:
                y_columns.remove(x_column)
        
        # 如果没有Y轴列，返回错误
        if not y_columns:
            return {"error": "没有找到有效的Y轴数据列"}
            
        # 配置
        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.LINE])
        dpi = config.get('dpi', 100)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 获取颜色
        colors = self._get_color_palette(len(y_columns))
        
        # 绘制折线
        for i, y_col in enumerate(y_columns):
            if y_col in df.columns:
                marker = 'o' if show_markers else None
                if show_area:
                    ax.fill_between(df[x_column], df[y_col], alpha=0.2, color=colors[i % len(colors)])
                ax.plot(df[x_column], df[y_col], marker=marker, label=y_col, 
                       color=colors[i % len(colors)], linewidth=2, markersize=4)
        
        # 设置X轴格式
        if df[x_column].dtype == 'datetime64[ns]':
            # 日期格式化
            date_format = '%Y-%m-%d' if (df[x_column].max() - df[x_column].min()).days > 365 else '%m-%d'
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(date_format))
            plt.xticks(rotation=45)
        
        # 添加注释
        self._add_chart_annotations(fig, ax, title, 
                                  x_label or x_column, 
                                  y_label or "数值", 
                                  df, ChartType.LINE)
        
        # 转换为图像
        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)
        
        # 返回结果
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
        """
        生成柱状图
        
        Args:
            data: 图表数据
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            x_column: X轴数据列名
            y_columns: Y轴数据列名列表
            stacked: 是否堆叠显示
            horizontal: 是否水平显示
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
            
        # 确定X轴和Y轴
        if x_column is None:
            # 使用第一列作为X轴
            x_column = df.columns[0]
        
        # 如果没有指定Y轴列，使用所有数值列
        if y_columns is None:
            y_columns = df.select_dtypes(include=['number']).columns.tolist()
            # 如果X轴也是数值列，从Y轴列表中移除
            if x_column in y_columns:
                y_columns.remove(x_column)
        
        # 如果没有Y轴列，返回错误
        if not y_columns:
            return {"error": "没有找到有效的Y轴数据列"}
            
        # 配置
        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.BAR])
        dpi = config.get('dpi', 100)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 获取颜色
        colors = self._get_color_palette(len(y_columns))
        
        # 计算柱宽
        width = 0.8 / len(y_columns) if not stacked else 0.8
        
        # 绘制柱状图
        for i, y_col in enumerate(y_columns):
            if y_col in df.columns:
                x_pos = range(len(df))
                if not stacked and len(y_columns) > 1:
                    # 多系列非堆叠，需要错开位置
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
        
        # 设置刻度标签
        if horizontal:
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df[x_column])
        else:
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x_column], rotation=45 if len(df) > 5 else 0)
        
        # 添加数值标签
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
        
        # 添加注释
        self._add_chart_annotations(fig, ax, title, 
                                  y_label or "数值" if horizontal else x_label or x_column, 
                                  x_label or x_column if horizontal else y_label or "数值", 
                                  df, ChartType.BAR)
        
        # 转换为图像
        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)
        
        # 返回结果
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
        """
        生成饼图
        
        Args:
            data: 图表数据
            title: 图表标题
            label_column: 标签列名
            value_column: 数值列名
            show_percentage: 是否显示百分比
            donut: 是否为环形图
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
            
        # 确定标签列和数值列
        if label_column is None:
            # 使用第一个非数值列作为标签列
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                label_column = non_numeric_cols[0]
            else:
                label_column = df.columns[0]
        
        if value_column is None:
            # 使用第一个数值列作为数值列
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                value_column = numeric_cols[0]
            else:
                # 如果没有数值列，使用计数
                value_column = 'count'
                df[value_column] = 1
        
        # 如果标签列和数值列相同，返回错误
        if label_column == value_column:
            return {"error": "标签列和数值列不能相同"}
            
        # 配置
        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.PIE])
        dpi = config.get('dpi', 100)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 获取颜色
        colors = self._get_color_palette(len(df))
        
        # 计算总和
        total = df[value_column].sum()
        
        # 绘制饼图
        if donut:
            # 环形图
            wedges, texts, autotexts = ax.pie(
                df[value_column], 
                labels=None,  # 不在图上显示标签
                autopct='%1.1f%%' if show_percentage else None,
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.5)  # 环形宽度
            )
        else:
            # 普通饼图
            wedges, texts, autotexts = ax.pie(
                df[value_column], 
                labels=None,  # 不在图上显示标签
                autopct='%1.1f%%' if show_percentage else None,
                startangle=90,
                colors=colors
            )
        
        # 设置标签格式
        if show_percentage:
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
        
        # 添加图例
        legend_labels = [f"{row[label_column]} ({row[value_column]:.1f}, {row[value_column]/total*100:.1f}%)" 
                        for _, row in df.iterrows()]
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 设置标题
        ax.set_title(title, fontsize=14, pad=20)
        
        # 设置为圆形
        ax.axis('equal')
        
        # 转换为图像
        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)
        
        # 返回结果
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
        """
        生成散点图
        
        Args:
            data: 图表数据
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            x_column: X轴数据列名
            y_column: Y轴数据列名
            category_column: 分类列名，用于区分不同系列
            show_trend_line: 是否显示趋势线
            show_correlation: 是否显示相关系数
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
            
        # 确定X轴和Y轴列
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return {"error": "散点图需要至少两个数值列"}
            
        if x_column is None:
            x_column = numeric_cols[0]
        
        if y_column is None:
            # 选择与x_column不同的第一个数值列
            for col in numeric_cols:
                if col != x_column:
                    y_column = col
                    break
        
        # 如果没有找到合适的Y轴列，返回错误
        if y_column is None:
            return {"error": "没有找到合适的Y轴数据列"}
            
        # 配置
        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.SCATTER])
        dpi = config.get('dpi', 100)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 处理分类
        if category_column and category_column in df.columns:
            # 按分类绘制不同颜色的散点
            categories = df[category_column].unique()
            colors = self._get_color_palette(len(categories))
            
            for i, category in enumerate(categories):
                category_data = df[df[category_column] == category]
                ax.scatter(category_data[x_column], category_data[y_column], 
                        label=category, color=colors[i % len(colors)], 
                        alpha=0.7, s=50)
                
                # 为每个分类添加趋势线
                if show_trend_line and len(category_data) >= 2:
                    try:
                        z = np.polyfit(category_data[x_column], category_data[y_column], 1)
                        p = np.poly1d(z)
                        ax.plot(category_data[x_column], p(category_data[x_column]), 
                            '--', color=colors[i % len(colors)], alpha=0.5)
                    except:
                        logger.warning(f"无法为分类 {category} 添加趋势线")
        else:
            # 单系列散点图
            ax.scatter(df[x_column], df[y_column], alpha=0.7, s=50, 
                    color=self._get_color_palette()[0])
            
            # 添加趋势线
            if show_trend_line and len(df) >= 2:
                try:
                    z = np.polyfit(df[x_column], df[y_column], 1)
                    p = np.poly1d(z)
                    ax.plot(df[x_column], p(df[x_column]), '--', 
                        color='red', alpha=0.5)
                    
                    # 显示趋势线方程
                    eq_text = f'y = {z[0]:.2f}x + {z[1]:.2f}'
                    ax.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=9, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                except:
                    logger.warning("无法添加趋势线")
        
        # 显示相关系数
        if show_correlation and not category_column:
            try:
                corr = df[[x_column, y_column]].corr().iloc[0, 1]
                corr_text = f'相关系数: {corr:.2f}'
                ax.annotate(corr_text, xy=(0.05, 0.85), xycoords='axes fraction',
                        fontsize=9, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.3))
            except:
                logger.warning("无法计算相关系数")
        
        # 添加注释
        self._add_chart_annotations(fig, ax, title, 
                                x_label or x_column, 
                                y_label or y_column, 
                                df, ChartType.SCATTER)
        
        # 转换为图像
        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)
        
        # 返回结果
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
        """
        生成热力图
        
        Args:
            data: 图表数据
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            value_column: 数值列名
            x_column: X轴分类列名
            y_column: Y轴分类列名
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
        
        # 配置
        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.HEATMAP])
        dpi = config.get('dpi', 100)
        
        # 确定列
        if len(df.columns) < 3 and value_column is None:
            # 数据可能已经是矩阵形式
            pivot_data = df
        else:
            # 需要透视表转换
            if not x_column:
                x_column = df.columns[0]
            if not y_column:
                # 找到一个与x_column不同的列
                for col in df.columns:
                    if col != x_column:
                        y_column = col
                        break
            if not value_column:
                # 找到一个数值列
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
            
            # 创建透视表
            try:
                pivot_data = df.pivot(index=y_column, columns=x_column, values=value_column)
            except:
                return {"error": "无法创建热力图所需的数据透视表"}
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 绘制热力图
        sns.heatmap(pivot_data, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap='YlOrRd')
        
        # 设置标题和标签
        ax.set_title(title, fontsize=14, pad=20)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        
        # 调整布局
        plt.tight_layout()
        
        # 转换为图像
        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)
        
        # 返回结果
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
        """
        生成雷达图
        
        Args:
            data: 图表数据
            title: 图表标题
            categories: 雷达图的维度类别
            series_column: 系列列名
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
        
        # 配置
        config = config or {}
        figsize = config.get('figsize', self.default_sizes[ChartType.RADAR])
        dpi = config.get('dpi', 100)
        
        # 确定类别和系列
        if categories is None:
            # 如果没有指定类别，使用所有数值列作为类别
            categories = df.select_dtypes(include=['number']).columns.tolist()
        
        if not categories:
            return {"error": "没有找到合适的雷达图维度"}
        
        # 创建图表
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        # 获取颜色
        colors = self._get_color_palette()
        
        # 绘制雷达图
        if series_column and series_column in df.columns:
            # 多系列雷达图
            series = df[series_column].unique()
            
            for i, s in enumerate(series):
                series_data = df[df[series_column] == s]
                values = [series_data[cat].iloc[0] if cat in series_data else 0 for cat in categories]
                values += values[:1]  # 闭合雷达图
                
                ax.plot(angles, values, 'o-', linewidth=2, label=s, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        else:
            # 单系列雷达图
            for i, row in df.iterrows():
                values = [row[cat] if cat in row else 0 for cat in categories]
                values += values[:1]  # 闭合雷达图
                
                label = f"系列 {i+1}" if i < 5 else None  # 最多显示5个系列的图例
                ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
                
                if i >= 9:  # 最多显示10个系列
                    break
        
        # 设置刻度和标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # 设置标题
        ax.set_title(title, fontsize=14, pad=20)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 调整布局
        plt.tight_layout()
        
        # 转换为图像
        img_data = self._fig_to_image(fig, config.get('output_format', OutputFormat.PNG))
        plt.close(fig)
        
        # 返回结果
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
            # 转换为Base64编码
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=fig.dpi)
            buf.seek(0)
            result["base64"] = base64.b64encode(buf.read()).decode('utf-8')
            result["format"] = "png"
        
        elif output_format == OutputFormat.SVG:
            # 转换为SVG
            buf = io.BytesIO()
            fig.savefig(buf, format='svg')
            buf.seek(0)
            result["svg"] = buf.read().decode('utf-8')
            result["format"] = "svg"
        
        elif output_format == OutputFormat.HTML:
            # 如果有Plotly，使用Plotly转换为HTML
            if PLOTLY_AVAILABLE and hasattr(fig, 'to_html'):
                result["html"] = fig.to_html(include_plotlyjs='cdn')
            else:
                # 否则使用SVG嵌入HTML
                buf = io.BytesIO()
                fig.savefig(buf, format='svg')
                buf.seek(0)
                svg_data = buf.read().decode('utf-8')
                result["html"] = f"<div>{svg_data}</div>"
            result["format"] = "html"
        
        else:  # PNG或其他格式
            # 转换为PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=fig.dpi)
            buf.seek(0)
            result["binary"] = buf.read()
            result["format"] = "png"
        
        return result

    def generate_chart(self, data: Union[Dict[str, Any], pd.DataFrame], 
                    chart_type: ChartType = None, 
                    title: str = "数据分析图表",
                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        智能生成图表
        
        根据数据特征自动选择合适的图表类型，或使用指定的图表类型
        
        Args:
            data: 图表数据
            chart_type: 图表类型，如果为None则自动检测
            title: 图表标题
            config: 额外配置
            
        Returns:
            Dict[str, Any]: 图表结果
        """
        # 准备数据
        df = self._prepare_data(data)
        if df.empty:
            return {"error": "无有效数据"}
        
        # 自动检测图表类型
        if chart_type is None:
            chart_type = self._detect_chart_type(df)
        
        # 根据图表类型调用相应的生成方法
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
            # 默认使用柱状图
            return self.generate_bar_chart(df, title=title, config=config)

    def save_chart(self, chart_result: Dict[str, Any], filename: str, 
                output_format: OutputFormat = OutputFormat.PNG) -> str:
        """
        保存图表到文件
        
        Args:
            chart_result: 图表结果（由generate_*方法返回）
            filename: 文件名
            output_format: 输出格式
            
        Returns:
            str: 保存的文件路径
        """
        if "error" in chart_result:
            logger.error(f"无法保存图表: {chart_result['error']}")
            return None
        
        # 确保文件扩展名与格式匹配
        if not filename.endswith(f".{output_format.value}"):
            filename = f"{filename}.{output_format.value}"
        
        # 获取图像数据
        img_data = chart_result.get("image_data", {})
        
        try:
            if output_format == OutputFormat.BASE64:
                # 保存Base64编码
                with open(filename, 'w') as f:
                    f.write(img_data.get("base64", ""))
            
            elif output_format == OutputFormat.SVG:
                # 保存SVG
                with open(filename, 'w') as f:
                    f.write(img_data.get("svg", ""))
            
            elif output_format == OutputFormat.HTML:
                # 保存HTML
                with open(filename, 'w') as f:
                    f.write(img_data.get("html", ""))
            
            else:  # PNG或其他二进制格式
                # 保存二进制数据
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
                         formatter: FinancialFormatter = None) -> ChartGenerator:
    """
    创建图表生成器
    
    Args:
        theme: 图表主题
        interactive: 是否生成交互式图表
        formatter: 金融数据格式化工具
        
    Returns:
        ChartGenerator: 图表生成器实例
    """
    return ChartGenerator(theme, interactive, formatter)


# ============= 使用示例 =============

def main():
    """使用示例"""
    # 创建图表生成器
    generator = create_chart_generator()
    
    # 示例1：生成折线图
    time_data = pd.DataFrame({
        '日期': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        '收入': [1200, 1300, 1450, 1800, 2100, 2400, 2300, 2500, 2800, 3100, 3400, 3800],
        '支出': [1000, 1100, 1200, 1300, 1500, 1700, 1600, 1800, 2000, 2200, 2400, 2600]
    })
    
    line_chart = generator.generate_line_chart(
        time_data,
        title="2023年月度收支趋势",
        x_column='日期',
        y_columns=['收入', '支出'],
        show_markers=True,
        show_area=True
    )
    
    # 保存图表
    generator.save_chart(line_chart, "monthly_trend.png")
    
    # 示例2：生成饼图
    category_data = pd.DataFrame({
        '类别': ['产品A', '产品B', '产品C', '产品D', '其他'],
        '销售额': [4500, 2500, 1800, 1200, 800]
    })
    
    pie_chart = generator.generate_pie_chart(
        category_data,
        title="产品销售占比",
        label_column='类别',
        value_column='销售额',
        donut=True
    )
    
    generator.save_chart(pie_chart, "sales_distribution.png")
    
    # 示例3：智能图表生成
    performance_data = pd.DataFrame({
        '季度': ['Q1', 'Q2', 'Q3', 'Q4'],
        '实际': [85, 88, 92, 96],
        '目标': [80, 85, 90, 95],
        '完成率': [1.06, 1.04, 1.02, 1.01]
    })
    
    # 自动检测合适的图表类型
    auto_chart = generator.generate_chart(
        performance_data,
        title="季度绩效分析"
    )
    
    generator.save_chart(auto_chart, "performance_analysis.png")


if __name__ == "__main__":
    main()