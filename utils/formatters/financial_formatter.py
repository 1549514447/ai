# utils/formatters/financial_formatter.py
"""
💰 金融数据格式化工具

专业的金融数据格式化工具，提供多种业务友好的展示格式：
- 货币格式化（支持多币种）
- 百分比格式化（支持多种精度和样式）
- 趋势指示器（箭头、颜色编码等）
- 数值简化（K, M, B等）
- 时间序列格式化
- 同比/环比变化展示
- 表格数据格式化
"""

import locale
import re
from typing import Dict, Any, List, Union, Optional, Tuple
from datetime import datetime, date
import math
from enum import Enum
import json


class TrendDirection(Enum):
    """趋势方向枚举"""
    UP = "up"              # 上升
    DOWN = "down"          # 下降
    STABLE = "stable"      # 稳定
    VOLATILE = "volatile"  # 波动
    UNKNOWN = "unknown"    # 未知


class FormatStyle(Enum):
    """格式化样式枚举"""
    DEFAULT = "default"    # 默认样式
    COMPACT = "compact"    # 紧凑样式
    FULL = "full"          # 完整样式
    BUSINESS = "business"  # 商务样式
    SIMPLE = "simple"      # 简单样式
    TECHNICAL = "technical"  # 技术样式


class FinancialFormatter:
    """
    💰 金融数据格式化工具
    
    提供丰富的金融数据格式化功能，使数据展示更专业、更易读
    """
    
    def __init__(self, locale_str: str = 'zh_CN', default_currency: str = 'CNY'):
        """
        初始化金融格式化工具
        
        Args:
            locale_str: 地区设置，默认中文
            default_currency: 默认货币，默认人民币
        """
        try:
            locale.setlocale(locale.LC_ALL, locale_str)
        except:
            # 如果设置失败，使用系统默认
            locale.setlocale(locale.LC_ALL, '')
            
        self.default_currency = default_currency
        
        # 货币符号映射
        self.currency_symbols = {
            'CNY': '¥',
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'HKD': 'HK$',
            'KRW': '₩',
            'RUB': '₽'
        }
        
        # 趋势指示器映射
        self.trend_indicators = {
            TrendDirection.UP: {
                'symbol': '↑',
                'color': 'red',
                'description': '上升',
                'html': '<span style="color:red">↑</span>'
            },
            TrendDirection.DOWN: {
                'symbol': '↓',
                'color': 'green',
                'description': '下降',
                'html': '<span style="color:green">↓</span>'
            },
            TrendDirection.STABLE: {
                'symbol': '→',
                'color': 'gray',
                'description': '稳定',
                'html': '<span style="color:gray">→</span>'
            },
            TrendDirection.VOLATILE: {
                'symbol': '↕',
                'color': 'orange',
                'description': '波动',
                'html': '<span style="color:orange">↕</span>'
            },
            TrendDirection.UNKNOWN: {
                'symbol': '-',
                'color': 'gray',
                'description': '未知',
                'html': '<span style="color:gray">-</span>'
            }
        }
        
    # ============= 货币格式化 =============
    
    def format_currency(self, amount: Union[float, int], 
                       currency: str = None, 
                       style: FormatStyle = FormatStyle.DEFAULT,
                       decimal_places: int = 2,
                       show_symbol: bool = True) -> str:
        """
        格式化货币金额
        
        Args:
            amount: 金额数值
            currency: 货币代码，默认使用初始化设置的货币
            style: 格式化样式
            decimal_places: 小数位数
            show_symbol: 是否显示货币符号
            
        Returns:
            str: 格式化后的货币字符串
        """
        if amount is None:
            return "-"
            
        currency = currency or self.default_currency
        symbol = self.currency_symbols.get(currency, '')
        
        # 根据样式选择格式化方法
        if style == FormatStyle.COMPACT:
            formatted = self.format_compact_number(amount, decimal_places)
            if show_symbol:
                return f"{symbol}{formatted}"
            return formatted
            
        elif style == FormatStyle.SIMPLE:
            formatted = f"{amount:,.{decimal_places}f}"
            if show_symbol:
                return f"{symbol}{formatted}"
            return formatted
            
        elif style == FormatStyle.BUSINESS:
            if abs(amount) >= 1_000_000_000:
                formatted = f"{amount/1_000_000_000:.{decimal_places}f}B"
            elif abs(amount) >= 1_000_000:
                formatted = f"{amount/1_000_000:.{decimal_places}f}M"
            elif abs(amount) >= 1_000:
                formatted = f"{amount/1_000:.{decimal_places}f}K"
            else:
                formatted = f"{amount:.{decimal_places}f}"
                
            if show_symbol:
                return f"{symbol}{formatted}"
            return formatted
            
        else:  # 默认样式
            formatted = f"{amount:,.{decimal_places}f}"
            if show_symbol:
                return f"{symbol}{formatted}"
            return formatted
    
    # ============= 百分比格式化 =============
    
    def format_percentage(self, value: Union[float, int], 
                         decimal_places: int = 2,
                         show_sign: bool = False,
                         style: FormatStyle = FormatStyle.DEFAULT) -> str:
        """
        格式化百分比
        
        Args:
            value: 百分比值（如0.1234表示12.34%）
            decimal_places: 小数位数
            show_sign: 是否显示正号（负号始终显示）
            style: 格式化样式
            
        Returns:
            str: 格式化后的百分比字符串
        """
        if value is None:
            return "-"
            
        # 将小数转换为百分比值
        percentage = value * 100
        
        # 根据样式选择格式化方法
        if style == FormatStyle.TECHNICAL:
            # 技术分析样式，如+12.34%↑
            sign = "+" if percentage > 0 else ("-" if percentage < 0 else "")
            if not show_sign and sign == "+":
                sign = ""
                
            direction = TrendDirection.UP if percentage > 0 else (
                TrendDirection.DOWN if percentage < 0 else TrendDirection.STABLE)
            indicator = self.trend_indicators[direction]['symbol']
            
            return f"{sign}{abs(percentage):.{decimal_places}f}%{indicator}"
            
        elif style == FormatStyle.COMPACT:
            # 紧凑样式，如12%
            sign = "+" if percentage > 0 and show_sign else ("-" if percentage < 0 else "")
            if decimal_places == 0 or percentage == int(percentage):
                return f"{sign}{abs(percentage):.0f}%"
            return f"{sign}{abs(percentage):.{decimal_places}f}%"
            
        elif style == FormatStyle.BUSINESS:
            # 商务样式，如"增长12.34%"
            if percentage > 0:
                return f"增长{percentage:.{decimal_places}f}%"
            elif percentage < 0:
                return f"下降{abs(percentage):.{decimal_places}f}%"
            else:
                return f"持平0%"
                
        else:  # 默认样式
            sign = "+" if percentage > 0 and show_sign else ("-" if percentage < 0 else "")
            return f"{sign}{abs(percentage):.{decimal_places}f}%"
    
    # ============= 趋势格式化 =============
    
    def format_trend(self, current_value: Union[float, int], 
                    previous_value: Union[float, int],
                    threshold: float = 0.01,
                    format_type: str = 'symbol') -> str:
        """
        格式化趋势指示器
        
        Args:
            current_value: 当前值
            previous_value: 前一个值
            threshold: 变化阈值，小于此值视为稳定
            format_type: 格式化类型，可选symbol(符号)、text(文本)、html(HTML)
            
        Returns:
            str: 格式化后的趋势指示
        """
        if current_value is None or previous_value is None or previous_value == 0:
            return self.trend_indicators[TrendDirection.UNKNOWN][format_type if format_type in ('symbol', 'description', 'html') else 'symbol']
            
        # 计算变化率
        change_rate = (current_value - previous_value) / abs(previous_value)
        
        # 确定趋势方向
        if abs(change_rate) < threshold:
            direction = TrendDirection.STABLE
        elif change_rate > 0:
            direction = TrendDirection.UP
        else:
            direction = TrendDirection.DOWN
            
        # 返回指定格式的趋势指示
        if format_type == 'text' or format_type == 'description':
            return self.trend_indicators[direction]['description']
        elif format_type == 'html':
            return self.trend_indicators[direction]['html']
        elif format_type == 'color':
            return self.trend_indicators[direction]['color']
        else:  # 默认返回符号
            return self.trend_indicators[direction]['symbol']
    
    def format_trend_with_value(self, current_value: Union[float, int], 
                               previous_value: Union[float, int],
                               is_percentage: bool = False,
                               decimal_places: int = 2) -> str:
        """
        格式化带数值的趋势
        
        Args:
            current_value: 当前值
            previous_value: 前一个值
            is_percentage: 是否为百分比值
            decimal_places: 小数位数
            
        Returns:
            str: 格式化后的趋势与数值，如"12.34% ↑"
        """
        if current_value is None or previous_value is None:
            return "-"
            
        # 计算变化
        change = current_value - previous_value
        
        # 计算变化率
        if previous_value != 0:
            change_rate = change / abs(previous_value)
        else:
            change_rate = float('inf') if change > 0 else float('-inf') if change < 0 else 0
            
        # 确定趋势方向
        if change_rate > 0:
            direction = TrendDirection.UP
        elif change_rate < 0:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.STABLE
            
        # 格式化数值
        if is_percentage:
            if change_rate == float('inf'):
                value_str = "∞%"
            else:
                value_str = self.format_percentage(change_rate, decimal_places, show_sign=True)
        else:
            value_str = f"{change:+,.{decimal_places}f}"
            
        # 返回格式化结果
        return f"{value_str} {self.trend_indicators[direction]['symbol']}"
    
    # ============= 数值简化格式化 =============
    
    def format_compact_number(self, value: Union[float, int], 
                             decimal_places: int = 1) -> str:
        """
        格式化为紧凑数值（K, M, B等）
        
        Args:
            value: 数值
            decimal_places: 小数位数
            
        Returns:
            str: 格式化后的紧凑数值
        """
        if value is None:
            return "-"
            
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        
        if abs_value >= 1_000_000_000:
            return f"{sign}{abs_value/1_000_000_000:.{decimal_places}f}B"
        elif abs_value >= 1_000_000:
            return f"{sign}{abs_value/1_000_000:.{decimal_places}f}M"
        elif abs_value >= 1_000:
            return f"{sign}{abs_value/1_000:.{decimal_places}f}K"
        else:
            return f"{sign}{abs_value:.{decimal_places}f}"
    
    # ============= 时间序列格式化 =============
    
    def format_date(self, date_value: Union[str, datetime, date], 
                   format_str: str = '%Y-%m-%d') -> str:
        """
        格式化日期
        
        Args:
            date_value: 日期值
            format_str: 日期格式字符串
            
        Returns:
            str: 格式化后的日期字符串
        """
        if date_value is None:
            return "-"
            
        if isinstance(date_value, str):
            try:
                # 尝试解析字符串日期
                date_obj = datetime.strptime(date_value, '%Y-%m-%d')
                return date_obj.strftime(format_str)
            except:
                return date_value
        elif isinstance(date_value, (datetime, date)):
            return date_value.strftime(format_str)
        else:
            return str(date_value)
    
    def format_date_range(self, start_date: Union[str, datetime, date],
                         end_date: Union[str, datetime, date],
                         format_str: str = '%Y-%m-%d') -> str:
        """
        格式化日期范围
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            format_str: 日期格式字符串
            
        Returns:
            str: 格式化后的日期范围字符串
        """
        start_str = self.format_date(start_date, format_str)
        end_str = self.format_date(end_date, format_str)
        
        return f"{start_str} 至 {end_str}"
    
    # ============= 同比/环比格式化 =============
    
    def format_yoy_change(self, current_value: Union[float, int], 
                         previous_value: Union[float, int],
                         decimal_places: int = 2) -> str:
        """
        格式化同比变化
        
        Args:
            current_value: 当前值
            previous_value: 去年同期值
            decimal_places: 小数位数
            
        Returns:
            str: 格式化后的同比变化
        """
        if current_value is None or previous_value is None or previous_value == 0:
            return "同比: -"
            
        change_rate = (current_value - previous_value) / abs(previous_value)
        percentage = self.format_percentage(change_rate, decimal_places, show_sign=True)
        
        if change_rate > 0:
            return f"同比增长: {percentage}"
        elif change_rate < 0:
            return f"同比下降: {percentage.replace('-', '')}"
        else:
            return f"同比持平: 0%"
    
    def format_mom_change(self, current_value: Union[float, int], 
                         previous_value: Union[float, int],
                         decimal_places: int = 2) -> str:
        """
        格式化环比变化
        
        Args:
            current_value: 当前值
            previous_value: 上期值
            decimal_places: 小数位数
            
        Returns:
            str: 格式化后的环比变化
        """
        if current_value is None or previous_value is None or previous_value == 0:
            return "环比: -"
            
        change_rate = (current_value - previous_value) / abs(previous_value)
        percentage = self.format_percentage(change_rate, decimal_places, show_sign=True)
        
        if change_rate > 0:
            return f"环比增长: {percentage}"
        elif change_rate < 0:
            return f"环比下降: {percentage.replace('-', '')}"
        else:
            return f"环比持平: 0%"
    
    # ============= 表格数据格式化 =============
    
    def format_table_cell(self, value: Any, cell_type: str = 'auto', 
                         options: Dict[str, Any] = None) -> str:
        """
        格式化表格单元格数据
        
        Args:
            value: 单元格值
            cell_type: 单元格类型，可选auto、currency、percentage、date、trend等
            options: 格式化选项
            
        Returns:
            str: 格式化后的单元格内容
        """
        if value is None:
            return "-"
            
        options = options or {}
        
        # 自动检测类型
        if cell_type == 'auto':
            if isinstance(value, (datetime, date)):
                cell_type = 'date'
            elif isinstance(value, (int, float)):
                if abs(value) < 1 and value != 0:
                    cell_type = 'percentage'
                else:
                    cell_type = 'number'
            elif isinstance(value, str):
                if re.match(r'^\d{4}-\d{2}-\d{2}', value):
                    cell_type = 'date'
                else:
                    cell_type = 'text'
            else:
                cell_type = 'text'
        
        # 根据类型格式化
        if cell_type == 'currency':
            currency = options.get('currency', self.default_currency)
            decimal_places = options.get('decimal_places', 2)
            style = options.get('style', FormatStyle.DEFAULT)
            return self.format_currency(value, currency, style, decimal_places)
            
        elif cell_type == 'percentage':
            decimal_places = options.get('decimal_places', 2)
            show_sign = options.get('show_sign', False)
            style = options.get('style', FormatStyle.DEFAULT)
            
            # 如果值已经是百分比形式（如50而不是0.5）
            if abs(value) > 1 and options.get('is_already_percentage', False):
                value = value / 100
                
            return self.format_percentage(value, decimal_places, show_sign, style)
            
        elif cell_type == 'date':
            format_str = options.get('format', '%Y-%m-%d')
            return self.format_date(value, format_str)
            
        elif cell_type == 'trend':
            previous_value = options.get('previous_value')
            if previous_value is not None:
                return self.format_trend(value, previous_value)
            return str(value)
            
        elif cell_type == 'compact':
            decimal_places = options.get('decimal_places', 1)
            return self.format_compact_number(value, decimal_places)
            
        else:  # 文本或其他类型
            return str(value)
    
    def format_table_row(self, row_data: Dict[str, Any], 
                        column_formats: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        格式化表格行数据
        
        Args:
            row_data: 行数据字典
            column_formats: 列格式配置
            
        Returns:
            Dict[str, str]: 格式化后的行数据
        """
        formatted_row = {}
        
        for col_name, value in row_data.items():
            if col_name in column_formats:
                col_format = column_formats[col_name]
                cell_type = col_format.get('type', 'auto')
                options = col_format.get('options', {})
                
                formatted_row[col_name] = self.format_table_cell(value, cell_type, options)
            else:
                formatted_row[col_name] = self.format_table_cell(value)
                
        return formatted_row
    
    # ============= 批量格式化 =============
    
    def batch_format(self, data: Dict[str, Any], format_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        批量格式化数据
        
        Args:
            data: 原始数据字典
            format_config: 格式化配置
            
        Returns:
            Dict[str, Any]: 格式化后的数据
        """
        result = {}
        
        for key, value in data.items():
            if key in format_config:
                config = format_config[key]
                format_type = config.get('type', 'auto')
                options = config.get('options', {})
                
                if format_type == 'currency':
                    result[key] = self.format_currency(
                        value, 
                        options.get('currency'), 
                        options.get('style', FormatStyle.DEFAULT),
                        options.get('decimal_places', 2)
                    )
                elif format_type == 'percentage':
                    result[key] = self.format_percentage(
                        value,
                        options.get('decimal_places', 2),
                        options.get('show_sign', False)
                    )
                elif format_type == 'date':
                    result[key] = self.format_date(
                        value,
                        options.get('format', '%Y-%m-%d')
                    )
                elif format_type == 'compact':
                    result[key] = self.format_compact_number(
                        value,
                        options.get('decimal_places', 1)
                    )
                elif format_type == 'trend':
                    previous_key = options.get('previous_key')
                    if previous_key and previous_key in data:
                        result[key] = self.format_trend(
                            value,
                            data[previous_key],
                            options.get('threshold', 0.01)
                        )
                    else:
                        result[key] = str(value)
                else:
                    result[key] = str(value)
            else:
                result[key] = value
                
        return result


# ============= 工厂函数 =============

def create_financial_formatter(locale_str: str = 'zh_CN', default_currency: str = 'CNY') -> FinancialFormatter:
    """
    创建金融格式化工具实例
    
    Args:
        locale_str: 地区设置
        default_currency: 默认货币
        
    Returns:
        FinancialFormatter: 金融格式化工具实例
    """
    return FinancialFormatter(locale_str, default_currency)


# ============= 使用示例 =============

def main():
    """使用示例"""
    formatter = create_financial_formatter()
    
    # 货币格式化
    print("===== 货币格式化 =====")
    print(f"默认: {formatter.format_currency(1234567.89)}")
    print(f"紧凑: {formatter.format_currency(1234567.89, style=FormatStyle.COMPACT)}")
    print(f"商务: {formatter.format_currency(1234567.89, style=FormatStyle.BUSINESS)}")
    print(f"美元: {formatter.format_currency(1234567.89, currency='USD')}")
    
    # 百分比格式化
    print("\n===== 百分比格式化 =====")
    print(f"默认: {formatter.format_percentage(0.1234)}")
    print(f"技术: {formatter.format_percentage(0.1234, style=FormatStyle.TECHNICAL)}")
    print(f"商务: {formatter.format_percentage(-0.1234, style=FormatStyle.BUSINESS)}")
    
    # 趋势格式化
    print("\n===== 趋势格式化 =====")
    print(f"上升: {formatter.format_trend(110, 100)}")
    print(f"下降: {formatter.format_trend(90, 100)}")
    print(f"稳定: {formatter.format_trend(100.5, 100)}")
    print(f"带值: {formatter.format_trend_with_value(110, 100, is_percentage=True)}")
    
    # 同比环比
    print("\n===== 同比环比 =====")
    print(formatter.format_yoy_change(110, 100))
    print(formatter.format_mom_change(90, 100))
    
    # 表格数据
    print("\n===== 表格数据 =====")
    row = {
        'date': '2023-05-01',
        'amount': 12345.67,
        'growth': 0.0823,
        'users': 1234567
    }
    
    formats = {
        'date': {'type': 'date', 'options': {'format': '%Y年%m月%d日'}},
        'amount': {'type': 'currency'},
        'growth': {'type': 'percentage', 'options': {'show_sign': True}},
        'users': {'type': 'compact'}
    }
    
    formatted = formatter.format_table_row(row, formats)
    print(json.dumps(formatted, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()