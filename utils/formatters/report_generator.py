# utils/formatters/report_generator.py
"""
📊 专业报告生成工具

自动生成结构化金融分析报告，支持多种输出格式：
- Markdown格式（轻量级文本格式）
- HTML格式（网页展示）
- PDF格式（专业打印文档）

核心特点:
- 模板驱动的报告生成
- 支持图表和表格嵌入
- 自动格式化数据展示
- 多种主题和样式
- 支持自定义和扩展
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import markdown
import jinja2
import pdfkit
from weasyprint import HTML, CSS
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io

# 导入格式化工具
from utils.formatters.financial_formatter import FinancialFormatter, create_financial_formatter
# 导入图表生成器
from utils.formatters.chart_generator import ChartGenerator, ChartType, ChartTheme, create_chart_generator

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式枚举"""
    MARKDOWN = "markdown"  # Markdown格式
    HTML = "html"          # HTML格式
    PDF = "pdf"            # PDF格式


class ReportTheme(Enum):
    """报告主题枚举"""
    DEFAULT = "default"    # 默认主题
    BUSINESS = "business"  # 商务主题
    MODERN = "modern"      # 现代主题
    MINIMAL = "minimal"    # 极简主题
    DARK = "dark"          # 暗色主题


class ReportSection:
    """报告章节类"""
    
    def __init__(self, title: str, content: str = "", level: int = 1):
        """
        初始化报告章节
        
        Args:
            title: 章节标题
            content: 章节内容
            level: 章节级别（1-6）
        """
        self.title = title
        self.content = content
        self.level = max(1, min(level, 6))  # 确保级别在1-6之间
        self.subsections = []
        self.charts = []
        self.tables = []
        self.metrics = []
        
    def add_subsection(self, title: str, content: str = "", level: int = None) -> 'ReportSection':
        """添加子章节"""
        if level is None:
            level = self.level + 1
        subsection = ReportSection(title, content, level)
        self.subsections.append(subsection)
        return subsection
        
    def add_chart(self, chart_data: Dict[str, Any], caption: str = "") -> None:
        """
        添加已生成的图表
        
        Args:
            chart_data: 图表数据（由图表生成器生成）
            caption: 图表标题
        """
        self.charts.append({
            'data': chart_data,
            'caption': caption
        })
    
    def generate_chart(self, data: Union[Dict[str, Any], pd.DataFrame], 
                     chart_type: ChartType = None, 
                     title: str = None,
                     caption: str = None,
                     chart_generator: ChartGenerator = None,
                     config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成并添加图表
        
        Args:
            data: 图表数据
            chart_type: 图表类型，如果为None则自动检测
            title: 图表标题
            caption: 图表说明文字
            chart_generator: 图表生成器实例，如果为None则创建新实例
            config: 图表配置
            
        Returns:
            Dict[str, Any]: 生成的图表数据
        """
        # 使用提供的图表生成器或创建新实例
        generator = chart_generator or create_chart_generator()
        
        # 生成图表
        if title is None:
            title = f"{self.title}图表"
            
        chart_result = generator.generate_chart(
            data=data,
            chart_type=chart_type,
            title=title,
            config=config or {}
        )
        
        # 添加到章节的图表列表
        self.add_chart(chart_result, caption or title)
        
        return chart_result
    
    def generate_line_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                          title: str = None,
                          caption: str = None,
                          x_column: str = None,
                          y_columns: List[str] = None,
                          show_markers: bool = True,
                          show_area: bool = False,
                          chart_generator: ChartGenerator = None) -> Dict[str, Any]:
        """
        生成并添加折线图
        
        Args:
            data: 图表数据
            title: 图表标题
            caption: 图表说明文字
            x_column: X轴数据列名
            y_columns: Y轴数据列名列表
            show_markers: 是否显示数据点标记
            show_area: 是否显示面积
            chart_generator: 图表生成器实例
            
        Returns:
            Dict[str, Any]: 生成的图表数据
        """
        # 使用提供的图表生成器或创建新实例
        generator = chart_generator or create_chart_generator()
        
        # 生成折线图
        if title is None:
            title = f"{self.title}趋势"
            
        chart_result = generator.generate_line_chart(
            data=data,
            title=title,
            x_column=x_column,
            y_columns=y_columns,
            show_markers=show_markers,
            show_area=show_area
        )
        
        # 添加到章节的图表列表
        self.add_chart(chart_result, caption or title)
        
        return chart_result
    
    def generate_bar_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                         title: str = None,
                         caption: str = None,
                         x_column: str = None,
                         y_columns: List[str] = None,
                         stacked: bool = False,
                         horizontal: bool = False,
                         chart_generator: ChartGenerator = None) -> Dict[str, Any]:
        """
        生成并添加柱状图
        
        Args:
            data: 图表数据
            title: 图表标题
            caption: 图表说明文字
            x_column: X轴数据列名
            y_columns: Y轴数据列名列表
            stacked: 是否堆叠显示
            horizontal: 是否水平显示
            chart_generator: 图表生成器实例
            
        Returns:
            Dict[str, Any]: 生成的图表数据
        """
        # 使用提供的图表生成器或创建新实例
        generator = chart_generator or create_chart_generator()
        
        # 生成柱状图
        if title is None:
            title = f"{self.title}对比"
            
        chart_result = generator.generate_bar_chart(
            data=data,
            title=title,
            x_column=x_column,
            y_columns=y_columns,
            stacked=stacked,
            horizontal=horizontal
        )
        
        # 添加到章节的图表列表
        self.add_chart(chart_result, caption or title)
        
        return chart_result
    
    def generate_pie_chart(self, data: Union[Dict[str, Any], pd.DataFrame],
                         title: str = None,
                         caption: str = None,
                         label_column: str = None,
                         value_column: str = None,
                         donut: bool = False,
                         chart_generator: ChartGenerator = None) -> Dict[str, Any]:
        """
        生成并添加饼图
        
        Args:
            data: 图表数据
            title: 图表标题
            caption: 图表说明文字
            label_column: 标签列名
            value_column: 数值列名
            donut: 是否为环形图
            chart_generator: 图表生成器实例
            
        Returns:
            Dict[str, Any]: 生成的图表数据
        """
        # 使用提供的图表生成器或创建新实例
        generator = chart_generator or create_chart_generator()
        
        # 生成饼图
        if title is None:
            title = f"{self.title}占比"
            
        chart_result = generator.generate_pie_chart(
            data=data,
            title=title,
            label_column=label_column,
            value_column=value_column,
            donut=donut
        )
        
        # 添加到章节的图表列表
        self.add_chart(chart_result, caption or title)
        
        return chart_result
        
    def add_table(self, table_data: Union[List[Dict[str, Any]], pd.DataFrame], 
                 headers: List[str] = None, caption: str = "") -> None:
        """添加表格"""
        # 如果是pandas DataFrame，转换为字典列表
        if isinstance(table_data, pd.DataFrame):
            table_data = table_data.to_dict('records')
            
        self.tables.append({
            'data': table_data,
            'headers': headers,
            'caption': caption
        })
        
    def to_markdown(self, formatter: FinancialFormatter = None) -> str:
        """转换为Markdown格式"""
        if formatter is None:
            formatter = create_financial_formatter()
            
        md = f"{'#' * self.level} {self.title}\n\n"
        
        # 添加内容
        if self.content:
            md += f"{self.content}\n\n"
            
        # 添加关键指标
        if self.metrics:
            md += "**关键指标:**\n\n"
            for metric in self.metrics:
                value = metric['value']
                if metric['format_type'] == 'currency':
                    value = formatter.format_currency(value)
                elif metric['format_type'] == 'percentage':
                    value = formatter.format_percentage(value)
                elif metric['format_type'] == 'compact':
                    value = formatter.format_compact_number(value)
                    
                md += f"- **{metric['name']}**: {value}"
                if metric['description']:
                    md += f" - {metric['description']}"
                md += "\n"
            md += "\n"
            
        # 添加表格
        for table in self.tables:
            if table['caption']:
                md += f"**{table['caption']}**\n\n"
                
            if table['data']:
                # 获取表头
                headers = table['headers'] or list(table['data'][0].keys())
                
                # 创建表头行
                md += "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                # 创建数据行
                for row in table['data']:
                    row_values = []
                    for header in headers:
                        value = row.get(header, "")
                        row_values.append(str(value))
                    md += "| " + " | ".join(row_values) + " |\n"
                md += "\n"
        
        # 图表在Markdown中通常是链接或说明
        for chart in self.charts:
            if chart['caption']:
                md += f"**{chart['caption']}**\n\n"
            md += "*[图表数据 - 在HTML/PDF版本中可见]*\n\n"
        
        # 添加子章节
        for subsection in self.subsections:
            md += subsection.to_markdown(formatter)
            
        return md
        
    def to_html_fragment(self, formatter: FinancialFormatter = None) -> str:
        """转换为HTML片段"""
        if formatter is None:
            formatter = create_financial_formatter()
            
        html = f"<h{self.level}>{self.title}</h{self.level}>\n"
        
        # 添加内容
        if self.content:
            html += f"<p>{self.content}</p>\n"
            
        # 添加关键指标
        if self.metrics:
            html += "<div class='metrics-container'>\n"
            html += "<h4>关键指标</h4>\n"
            html += "<ul class='metrics-list'>\n"
            
            for metric in self.metrics:
                value = metric['value']
                if metric['format_type'] == 'currency':
                    value = formatter.format_currency(value)
                elif metric['format_type'] == 'percentage':
                    value = formatter.format_percentage(value)
                elif metric['format_type'] == 'compact':
                    value = formatter.format_compact_number(value)
                    
                html += f"<li><strong>{metric['name']}:</strong> {value}"
                if metric['description']:
                    html += f" - <span class='description'>{metric['description']}</span>"
                html += "</li>\n"
                
            html += "</ul>\n</div>\n"
            
        # 添加表格
        for table in self.tables:
            html += "<div class='table-container'>\n"
            if table['caption']:
                html += f"<h4>{table['caption']}</h4>\n"
                
            html += "<table class='data-table'>\n"
            
            # 添加表头
            if table['data']:
                headers = table['headers'] or list(table['data'][0].keys())
                html += "<thead>\n<tr>\n"
                for header in headers:
                    html += f"<th>{header}</th>\n"
                html += "</tr>\n</thead>\n"
                
                # 添加数据行
                html += "<tbody>\n"
                for row in table['data']:
                    html += "<tr>\n"
                    for header in headers:
                        value = row.get(header, "")
                        html += f"<td>{value}</td>\n"
                    html += "</tr>\n"
                html += "</tbody>\n"
                
            html += "</table>\n</div>\n"
        
        # 添加图表（使用base64编码的图片或外部链接）
        for chart in self.charts:
            html += "<div class='chart-container'>\n"
            if chart['caption']:
                html += f"<h4>{chart['caption']}</h4>\n"
                
            # 如果有图表数据，尝试生成内嵌图表
            if chart['data'].get('image_data'):
                html += f"<img src='data:image/png;base64,{chart['data']['image_data']}' alt='{chart['caption']}' class='chart-image'>\n"
            elif chart['data'].get('image_url'):
                html += f"<img src='{chart['data']['image_url']}' alt='{chart['caption']}' class='chart-image'>\n"
            else:
                html += "<p class='chart-placeholder'>[图表数据 - 需要渲染]</p>\n"
                
            html += "</div>\n"
        
        # 添加子章节
        for subsection in self.subsections:
            html += subsection.to_html_fragment(formatter)
            
        return html


class Report:
    """报告类"""
    
    def __init__(self, title: str, subtitle: str = "", author: str = "", 
                theme: ReportTheme = ReportTheme.DEFAULT):
        """
        初始化报告
        
        Args:
            title: 报告标题
            subtitle: 报告副标题
            author: 作者
            theme: 报告主题
        """
        self.title = title
        self.subtitle = subtitle
        self.author = author
        self.theme = theme
        self.created_at = datetime.now()
        self.sections = []
        self.metadata = {}
        self.formatter = create_financial_formatter()
        
    def add_section(self, title: str, content: str = "") -> ReportSection:
        """添加章节"""
        section = ReportSection(title, content)
        self.sections.append(section)
        return section
        
    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
        
    def set_theme(self, theme: ReportTheme) -> None:
        """设置主题"""
        self.theme = theme
        
    def to_markdown(self) -> str:
        """生成Markdown格式报告"""
        md = f"# {self.title}\n\n"
        
        if self.subtitle:
            md += f"## {self.subtitle}\n\n"
            
        # 添加元数据
        md += f"*生成时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        if self.author:
            md += f"*作者: {self.author}*\n\n"
            
        # 添加自定义元数据
        if self.metadata:
            md += "### 报告信息\n\n"
            for key, value in self.metadata.items():
                md += f"- **{key}**: {value}\n"
            md += "\n"
            
        # 添加章节
        for section in self.sections:
            md += section.to_markdown(self.formatter)
            
        return md
        
    def to_html(self) -> str:
        """生成HTML格式报告"""
        # 加载HTML模板
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        if not os.path.exists(template_dir):
            template_dir = os.path.join(os.getcwd(), 'utils', 'formatters', 'templates')
            if not os.path.exists(template_dir):
                os.makedirs(template_dir)
                
        # 创建简单的默认模板
        default_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .report-header {
                    text-align: center;
                    margin-bottom: 40px;
                }
                .report-title {
                    font-size: 28px;
                    margin-bottom: 10px;
                }
                .report-subtitle {
                    font-size: 20px;
                    color: #666;
                    margin-bottom: 20px;
                }
                .report-meta {
                    font-size: 14px;
                    color: #888;
                }
                .metrics-container {
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .metrics-list {
                    list-style-type: none;
                    padding: 0;
                }
                .metrics-list li {
                    margin-bottom: 8px;
                }
                .table-container {
                    margin: 20px 0;
                    overflow-x: auto;
                }
                .data-table {
                    border-collapse: collapse;
                    width: 100%;
                }
                .data-table th, .data-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .data-table th {
                    background-color: #f2f2f2;
                }
                .data-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .chart-image {
                    max-width: 100%;
                    height: auto;
                }
                h1, h2, h3, h4, h5, h6 {
                    color: #444;
                    margin-top: 25px;
                }
                p {
                    margin-bottom: 15px;
                }
            </style>
        </head>
        <body>
            <div class="report-header">
                <div class="report-title">{{ title }}</div>
                {% if subtitle %}
                <div class="report-subtitle">{{ subtitle }}</div>
                {% endif %}
                <div class="report-meta">
                    生成时间: {{ created_at }}
                    {% if author %}
                    <br>作者: {{ author }}
                    {% endif %}
                </div>
            </div>
            
            {% if metadata %}
            <div class="report-metadata">
                <h3>报告信息</h3>
                <ul>
                {% for key, value in metadata.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <div class="report-content">
                {{ content }}
            </div>
        </body>
        </html>
        """
        
        # 保存默认模板
        default_template_path = os.path.join(template_dir, 'default.html')
        if not os.path.exists(default_template_path):
            with open(default_template_path, 'w', encoding='utf-8') as f:
                f.write(default_template)
        
        # 创建Jinja2环境
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template(f"{self.theme.value}.html" if os.path.exists(os.path.join(template_dir, f"{self.theme.value}.html")) else "default.html")
        
        # 生成HTML内容
        content = ""
        for section in self.sections:
            content += section.to_html_fragment(self.formatter)
            
        # 渲染模板
        html = template.render(
            title=self.title,
            subtitle=self.subtitle,
            author=self.author,
            created_at=self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            metadata=self.metadata,
            content=content
        )
        
        return html
        
    def to_pdf(self, output_path: str) -> str:
        """
        生成PDF格式报告
        
        Args:
            output_path: PDF文件输出路径
            
        Returns:
            str: 输出文件的路径
        """
        html_content = self.to_html()
        
        try:
            # 尝试使用weasyprint生成PDF
            HTML(string=html_content).write_pdf(output_path)
        except Exception as e:
            logger.warning(f"使用weasyprint生成PDF失败: {str(e)}，尝试使用pdfkit...")
            try:
                # 备选方案：使用pdfkit
                pdfkit.from_string(html_content, output_path)
            except Exception as e2:
                logger.error(f"PDF生成失败: {str(e2)}")
                raise Exception(f"无法生成PDF报告: {str(e2)}")
                
        return output_path
        
    def save(self, output_path: str, format: ReportFormat = ReportFormat.HTML) -> str:
        """
        保存报告到文件
        
        Args:
            output_path: 输出文件路径
            format: 报告格式
            
        Returns:
            str: 输出文件的路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if format == ReportFormat.MARKDOWN:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.to_markdown())
        elif format == ReportFormat.HTML:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.to_html())
        elif format == ReportFormat.PDF:
            self.to_pdf(output_path)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
            
        return output_path


class ReportGenerator:
    """
    📊 专业报告生成器
    
    自动生成结构化金融分析报告，支持多种输出格式
    """
    
    def __init__(self, templates_dir: str = None):
        """
        初始化报告生成器
        
        Args:
            templates_dir: 模板目录
        """
        self.templates_dir = templates_dir
        if templates_dir and not os.path.exists(templates_dir):
            os.makedirs(templates_dir, exist_ok=True)
            
        self.formatter = create_financial_formatter()
        self.chart_generator = create_chart_generator()
        
    def create_report(self, title: str, subtitle: str = "", author: str = "",
                     theme: ReportTheme = ReportTheme.DEFAULT) -> Report:
        """
        创建新报告
        
        Args:
            title: 报告标题
            subtitle: 报告副标题
            author: 作者
            theme: 报告主题
            
        Returns:
            Report: 报告对象
        """
        return Report(title, subtitle, author, theme)
        
    def generate_financial_report(self, data: Dict[str, Any], 
                                 title: str = "金融分析报告",
                                 output_format: ReportFormat = ReportFormat.HTML,
                                 output_path: str = None) -> Union[str, Report]:
        """
        生成金融分析报告
        
        Args:
            data: 报告数据
            title: 报告标题
            output_format: 输出格式
            output_path: 输出路径
            
        Returns:
            Union[str, Report]: 报告对象或输出文件路径
        """
        # 创建报告
        report = self.create_report(
            title=title,
            subtitle=data.get('subtitle', f"{datetime.now().strftime('%Y年%m月%d日')}"),
            author=data.get('author', "AI财务分析师")
        )
        
        # 添加元数据
        if 'metadata' in data:
            for key, value in data['metadata'].items():
                report.add_metadata(key, value)
        
        # 添加摘要
        if 'summary' in data:
            summary = report.add_section("摘要", data['summary'].get('content', ''))
            
            # 添加关键指标
            if 'key_metrics' in data['summary']:
                for metric in data['summary']['key_metrics']:
                    summary.add_metric(
                        metric['name'], 
                        metric['value'],
                        metric.get('format_type', 'plain'),
                        metric.get('description', '')
                    )
            
            # 添加摘要图表
            if 'chart_data' in data['summary']:
                chart_data = data['summary']['chart_data']
                if isinstance(chart_data, dict) and 'data' in chart_data:
                    chart_type = chart_data.get('chart_type', 'auto')
                    
                    if chart_type == 'line' or chart_type == 'trend':
                        summary.generate_line_chart(
                            data=chart_data['data'],
                            title=chart_data.get('title', '关键指标趋势'),
                            caption=chart_data.get('caption', ''),
                            chart_generator=self.chart_generator
                        )
                    elif chart_type == 'bar' or chart_type == 'comparison':
                        summary.generate_bar_chart(
                            data=chart_data['data'],
                            title=chart_data.get('title', '关键指标对比'),
                            caption=chart_data.get('caption', ''),
                            chart_generator=self.chart_generator
                        )
                    elif chart_type == 'pie' or chart_type == 'distribution':
                        summary.generate_pie_chart(
                            data=chart_data['data'],
                            title=chart_data.get('title', '关键指标分布'),
                            caption=chart_data.get('caption', ''),
                            chart_generator=self.chart_generator
                        )
                    else:
                        # 自动检测图表类型
                        summary.generate_chart(
                            data=chart_data['data'],
                            title=chart_data.get('title', '关键指标分析'),
                            caption=chart_data.get('caption', ''),
                            chart_generator=self.chart_generator
                        )
        
        # 添加详细分析章节
        if 'analysis_sections' in data:
            for section_data in data['analysis_sections']:
                section = report.add_section(
                    section_data['title'],
                    section_data.get('content', '')
                )
                
                # 添加指标
                if 'metrics' in section_data:
                    for metric in section_data['metrics']:
                        section.add_metric(
                            metric['name'], 
                            metric['value'],
                            metric.get('format_type', 'plain'),
                            metric.get('description', '')
                        )
                
                # 添加表格
                if 'tables' in section_data:
                    for table in section_data['tables']:
                        section.add_table(
                            table['data'],
                            table.get('headers'),
                            table.get('caption', '')
                        )
                
                # 添加图表
                if 'charts' in section_data:
                    for chart in section_data['charts']:
                        if isinstance(chart, dict) and 'data' in chart:
                            chart_type = chart.get('chart_type', 'auto')
                            
                            if chart_type == 'line' or chart_type == 'trend':
                                section.generate_line_chart(
                                    data=chart['data'],
                                    title=chart.get('title'),
                                    caption=chart.get('caption', ''),
                                    chart_generator=self.chart_generator
                                )
                            elif chart_type == 'bar' or chart_type == 'comparison':
                                section.generate_bar_chart(
                                    data=chart['data'],
                                    title=chart.get('title'),
                                    caption=chart.get('caption', ''),
                                    chart_generator=self.chart_generator
                                )
                            elif chart_type == 'pie' or chart_type == 'distribution':
                                section.generate_pie_chart(
                                    data=chart['data'],
                                    title=chart.get('title'),
                                    caption=chart.get('caption', ''),
                                    chart_generator=self.chart_generator
                                )
                            else:
                                # 自动检测图表类型
                                section.generate_chart(
                                    data=chart['data'],
                                    title=chart.get('title'),
                                    caption=chart.get('caption', ''),
                                    chart_generator=self.chart_generator
                                )
                
                # 添加子章节
                if 'subsections' in section_data:
                    for subsection_data in section_data['subsections']:
                        subsection = section.add_subsection(
                            subsection_data['title'],
                            subsection_data.get('content', '')
                        )
                        
                        # 添加子章节图表
                        if 'charts' in subsection_data:
                            for chart in subsection_data['charts']:
                                if isinstance(chart, dict) and 'data' in chart:
                                    chart_type = chart.get('chart_type', 'auto')
                                    
                                    if chart_type == 'line' or chart_type == 'trend':
                                        subsection.generate_line_chart(
                                            data=chart['data'],
                                            title=chart.get('title'),
                                            caption=chart.get('caption', ''),
                                            chart_generator=self.chart_generator
                                        )
                                    elif chart_type == 'bar' or chart_type == 'comparison':
                                        subsection.generate_bar_chart(
                                            data=chart['data'],
                                            title=chart.get('title'),
                                            caption=chart.get('caption', ''),
                                            chart_generator=self.chart_generator
                                        )
                                    elif chart_type == 'pie' or chart_type == 'distribution':
                                        subsection.generate_pie_chart(
                                            data=chart['data'],
                                            title=chart.get('title'),
                                            caption=chart.get('caption', ''),
                                            chart_generator=self.chart_generator
                                        )
                                    else:
                                        # 自动检测图表类型
                                        subsection.generate_chart(
                                            data=chart['data'],
                                            title=chart.get('title'),
                                            caption=chart.get('caption', ''),
                                            chart_generator=self.chart_generator
                                        )
        
        # 添加结论
        if 'conclusion' in data:
            conclusion = report.add_section("结论与建议", data['conclusion'].get('content', ''))
            
            # 添加建议
            if 'recommendations' in data['conclusion']:
                recommendations_content = "\n\n### 建议\n\n"
                for i, rec in enumerate(data['conclusion']['recommendations']):
                    recommendations_content += f"{i+1}. {rec}\n"
                
                conclusion.content += recommendations_content
        
        # 保存报告
        if output_path:
            return report.save(output_path, output_format)
        
        return report
        
    def generate_trend_report(self, trend_data: Dict[str, Any],
                            title: str = "趋势分析报告",
                            period: str = None,
                            output_format: ReportFormat = ReportFormat.HTML,
                            output_path: str = None) -> Union[str, Report]:
        """
        生成趋势分析报告
        
        Args:
            trend_data: 趋势数据
            title: 报告标题
            period: 分析周期
            output_format: 输出格式
            output_path: 输出路径
            
        Returns:
            Union[str, Report]: 报告对象或输出文件路径
        """
        # 创建报告
        subtitle = f"{period} 趋势分析" if period else "趋势分析"
        report = self.create_report(
            title=title,
            subtitle=subtitle,
            author=trend_data.get('author', '系统自动生成'),
            theme=ReportTheme(trend_data.get('theme', 'default'))
        )
        
        # 添加元数据
        report.add_metadata("分析周期", period or "未指定")
        report.add_metadata("数据点数量", trend_data.get('data_points_count', 'N/A'))
        report.add_metadata("分析深度", trend_data.get('analysis_depth', '标准'))
        
        # 添加摘要部分
        if 'summary' in trend_data:
            summary_section = report.add_section("摘要", trend_data['summary'].get('content', ''))
            
            # 添加关键趋势指标
            if 'key_trends' in trend_data['summary']:
                for trend in trend_data['summary']['key_trends']:
                    summary_section.add_metric(
                        name=trend['name'],
                        value=trend['value'],
                        format_type=trend.get('format_type', 'percentage'),
                        description=trend.get('description', '')
                    )
        
        # 添加趋势分析部分
        if 'trends' in trend_data:
            trends_section = report.add_section("趋势分析", trend_data.get('introduction', ''))
            
            # 添加各指标趋势
            for trend in trend_data['trends']:
                trend_subsection = trends_section.add_subsection(
                    trend['title'],
                    trend.get('description', '')
                )
                
                # 添加趋势图表
                if 'chart' in trend:
                    trend_subsection.add_chart(trend['chart'], "趋势变化")
                
                # 添加趋势数据表
                if 'data' in trend:
                    trend_subsection.add_table(
                        trend['data'],
                        trend.get('headers'),
                        "趋势数据"
                    )
                    
                # 添加趋势分析
                if 'analysis' in trend:
                    analysis_text = "\n\n**分析:**\n\n" + trend['analysis']
                    trend_subsection.content += analysis_text
        
        # 添加结论部分
        if 'conclusion' in trend_data:
            conclusion_section = report.add_section("结论与展望", trend_data['conclusion'].get('content', ''))
            
            # 添加预测
            if 'predictions' in trend_data['conclusion']:
                predictions_text = "\n\n**预测:**\n\n"
                for prediction in trend_data['conclusion']['predictions']:
                    predictions_text += f"- {prediction}\n"
                conclusion_section.content += predictions_text
        
        # 保存报告
        if output_path:
            return report.save(output_path, output_format)
        
        return report
        
    def generate_comparison_report(self, comparison_data: Dict[str, Any],
                                 title: str = "对比分析报告",
                                 output_format: ReportFormat = ReportFormat.HTML,
                                 output_path: str = None) -> Union[str, Report]:
        """
        生成对比分析报告
        
        Args:
            comparison_data: 对比数据
            title: 报告标题
            output_format: 输出格式
            output_path: 输出路径
            
        Returns:
            Union[str, Report]: 报告对象或输出文件路径
        """
        # 创建报告
        report = self.create_report(
            title=title,
            subtitle=comparison_data.get('subtitle', '对比分析'),
            author=comparison_data.get('author', '系统自动生成'),
            theme=ReportTheme(comparison_data.get('theme', 'default'))
        )
        
        # 添加元数据
        if 'metadata' in comparison_data:
            for key, value in comparison_data['metadata'].items():
                report.add_metadata(key, value)
        
        # 添加摘要部分
        if 'summary' in comparison_data:
            summary_section = report.add_section("摘要", comparison_data['summary'].get('content', ''))
            
            # 添加关键对比指标
            if 'key_differences' in comparison_data['summary']:
                for diff in comparison_data['summary']['key_differences']:
                    summary_section.add_metric(
                        name=diff['name'],
                        value=diff['value'],
                        format_type=diff.get('format_type', 'auto'),
                        description=diff.get('description', '')
                    )
        
        # 添加对比分析部分
        if 'comparisons' in comparison_data:
            comparison_section = report.add_section("对比分析", comparison_data.get('introduction', ''))
            
            # 添加各维度对比
            for comparison in comparison_data['comparisons']:
                comparison_subsection = comparison_section.add_subsection(
                    comparison['title'],
                    comparison.get('description', '')
                )
                
                # 添加对比图表
                if 'chart' in comparison:
                    comparison_subsection.add_chart(comparison['chart'], "对比图表")
                
                # 添加对比数据表
                if 'data' in comparison:
                    comparison_subsection.add_table(
                        comparison['data'],
                        comparison.get('headers'),
                        "对比数据"
                    )
        
        # 添加结论部分
        if 'conclusion' in comparison_data:
            conclusion_section = report.add_section("结论", comparison_data['conclusion'].get('content', ''))
            
            # 添加建议
            if 'recommendations' in comparison_data['conclusion']:
                recommendations_text = "\n\n**建议:**\n\n"
                for rec in comparison_data['conclusion']['recommendations']:
                    recommendations_text += f"- {rec}\n"
                conclusion_section.content += recommendations_text
        
        # 保存报告
        if output_path:
            return report.save(output_path, output_format)
        
        return report


# ============= 工厂函数 =============

def create_report_generator(templates_dir: str = None) -> ReportGenerator:
    """
    创建报告生成器
    
    Args:
        templates_dir: 模板目录路径
        
    Returns:
        ReportGenerator: 报告生成器实例
    """
    return ReportGenerator(templates_dir)


# ============= 使用示例 =============

def main():
    """使用示例"""
    # 创建报告生成器
    generator = create_report_generator()
    
    # 示例1：创建简单报告
    report = generator.create_report(
        title="每月财务分析报告",
        subtitle="2023年5月",
        author="AI财务分析师"
    )
    
    # 添加摘要章节
    summary = report.add_section("摘要", "本报告分析了2023年5月的财务表现，重点关注收入增长、成本控制和利润率变化。")
    summary.add_metric("总收入", 1234567.89, "currency", "较上月增长8.3%")
    summary.add_metric("利润率", 0.23, "percentage", "较上月提升2.1个百分点")
    summary.add_metric("活跃用户", 45678, "compact", "较上月增加5.4%")
    
    # 添加收入分析章节
    revenue = report.add_section("收入分析", "本月收入结构和增长分析")
    
    # 添加收入表格
    revenue_data = [
        {"产品线": "产品A", "收入": 567890, "占比": 0.46, "同比增长": 0.12},
        {"产品线": "产品B", "收入": 345678, "占比": 0.28, "同比增长": 0.08},
        {"产品线": "产品C", "收入": 234567, "占比": 0.19, "同比增长": -0.05},
        {"产品线": "其他", "收入": 86432, "占比": 0.07, "同比增长": 0.02}
    ]
    revenue.add_table(revenue_data, ["产品线", "收入", "占比", "同比增长"], "收入明细表")
    
    # 添加图表示例
    time_data = pd.DataFrame({
        '月份': ['1月', '2月', '3月', '4月', '5月'],
        '收入': [1200000, 1350000, 1450000, 1600000, 1750000],
        '支出': [950000, 1050000, 1100000, 1200000, 1300000]
    })
    
    # 生成并添加趋势图
    revenue.generate_line_chart(
        data=time_data,
        title="2023年收入趋势",
        caption="月度收入与支出趋势对比",
        x_column='月份',
        y_columns=['收入', '支出'],
        show_area=True
    )
    
    # 生成并添加占比图
    category_data = pd.DataFrame({
        '产品线': ['产品A', '产品B', '产品C', '其他'],
        '销售额': [567890, 345678, 234567, 86432]
    })
    
    revenue.generate_pie_chart(
        data=category_data,
        title="产品销售占比",
        caption="各产品线销售额占比分析",
        label_column='产品线',
        value_column='销售额',
        donut=True
    )
    
    # 添加趋势子章节
    trend = revenue.add_subsection("收入趋势", "近6个月收入趋势分析")
    
    # 添加成本分析章节
    cost = report.add_section("成本分析", "本月成本构成和控制效果分析")
    
    # 生成并添加成本对比图
    cost_data = pd.DataFrame({
        '类别': ['人力成本', '运营成本', '市场费用', '技术投入', '其他费用'],
        '本月': [450000, 350000, 280000, 150000, 70000],
        '上月': [430000, 360000, 250000, 140000, 75000]
    })
    
    cost.generate_bar_chart(
        data=cost_data,
        title="成本构成对比",
        caption="本月与上月各类成本对比",
        x_column='类别',
        y_columns=['本月', '上月']
    )
    
    # 添加结论章节
    conclusion = report.add_section("结论与建议", "基于本月财务表现的结论和下一步建议")
    
    # 生成并保存报告
    # 保存为Markdown
    report.save("monthly_report.md", ReportFormat.MARKDOWN)
    
    # 保存为HTML
    report.save("monthly_report.html", ReportFormat.HTML)
    
    # 示例2：使用预定义模板生成趋势报告
    trend_data = {
        "subtitle": "用户增长趋势分析",
        "author": "AI数据分析师",
        "summary": {
            "content": "本报告分析了过去12个月的用户增长趋势，发现了几个关键模式。",
            "key_trends": [
                {"name": "月均增长率", "value": 0.053, "description": "稳定增长"},
                {"name": "季度波动", "value": 0.12, "description": "Q3增长最快"}
            ]
        },
        "trends": [
            {
                "title": "新用户增长",
                "description": "新注册用户数量变化趋势",
                "analysis": "新用户增长在假期季节明显加速，夏季略有放缓。"
            },
            {
                "title": "用户活跃度",
                "description": "月活跃用户比例变化",
                "analysis": "活跃度整体呈上升趋势，但在2月有明显下降。"
            }
        ],
        "conclusion": {
            "content": "用户增长整体表现良好，但仍有季节性波动。",
            "predictions": [
                "预计下季度增长率将保持在4-6%区间",
                "活跃度有望突破80%大关"
            ]
        }
    }
    
    generator.generate_trend_report(
        trend_data,
        title="用户增长分析",
        period="2023年1月-12月",
        output_format=ReportFormat.HTML,
        output_path="user_growth_report.html"
    )


if __name__ == "__main__":
    main()