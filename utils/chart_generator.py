# utils/chart_generator.py
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import statistics
import math

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """图表类型枚举"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    CANDLESTICK = "candlestick"
    GAUGE = "gauge"
    FUNNEL = "funnel"
    RADAR = "radar"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    WATERFALL = "waterfall"


class ChartTheme(Enum):
    """图表主题枚举"""
    DEFAULT = "default"
    FINANCIAL = "financial"
    DARK = "dark"
    MINIMAL = "minimal"
    COLORFUL = "colorful"


class ChartLibrary(Enum):
    """图表库枚举"""
    ECHARTS = "echarts"
    CHARTJS = "chartjs"
    D3 = "d3"


class ChartGenerator:
    """
    🎨 智能图表生成器 - 根据数据特征自动生成最佳图表

    核心功能：
    1. 智能分析数据特征，推荐最佳图表类型
    2. 自动配置图表样式和交互功能
    3. 支持多种图表库适配
    4. 金融专业图表支持
    """

    def __init__(self):
        """初始化图表生成器"""
        self.config = self._load_chart_config()
        self.themes = self._load_chart_themes()
        self.templates = self._load_chart_templates()

        # 图表生成统计
        self.stats = {
            'total_charts_generated': 0,
            'chart_type_distribution': {},
            'library_usage': {},
            'theme_usage': {}
        }

        logger.info("ChartGenerator initialized successfully")

    def _load_chart_config(self) -> Dict[str, Any]:
        """加载图表配置"""
        return {
            'default_library': ChartLibrary.ECHARTS,
            'default_theme': ChartTheme.FINANCIAL,
            'max_data_points': 1000,
            'color_palettes': {
                'financial': ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1'],
                'business': ['#2f54eb', '#13c2c2', '#52c41a', '#faad14', '#f5222d'],
                'minimal': ['#595959', '#8c8c8c', '#bfbfbf', '#d9d9d9', '#f0f0f0'],
                'dark': ['#177ddc', '#49aa19', '#d89614', '#a61d24', '#531dab']
            },
            'responsive_breakpoints': {
                'mobile': 768,
                'tablet': 1024,
                'desktop': 1440
            }
        }

    def _load_chart_themes(self) -> Dict[str, Dict[str, Any]]:
        """加载图表主题配置"""
        return {
            'financial': {
                'backgroundColor': '#ffffff',
                'textColor': '#262626',
                'gridColor': '#f0f0f0',
                'axisColor': '#bfbfbf',
                'titleFont': {'size': 16, 'weight': 'bold'},
                'labelFont': {'size': 12, 'weight': 'normal'}
            },
            'dark': {
                'backgroundColor': '#1f1f1f',
                'textColor': '#ffffff',
                'gridColor': '#404040',
                'axisColor': '#666666',
                'titleFont': {'size': 16, 'weight': 'bold'},
                'labelFont': {'size': 12, 'weight': 'normal'}
            },
            'minimal': {
                'backgroundColor': '#fafafa',
                'textColor': '#595959',
                'gridColor': '#e8e8e8',
                'axisColor': '#d9d9d9',
                'titleFont': {'size': 14, 'weight': 'normal'},
                'labelFont': {'size': 11, 'weight': 'normal'}
            }
        }

    def _load_chart_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载图表模板"""
        return {
            'trend_analysis': {
                'type': ChartType.LINE,
                'features': ['trend_line', 'data_zoom', 'tooltip'],
                'suitable_for': ['time_series', 'trend_data']
            },
            'financial_overview': {
                'type': ChartType.BAR,
                'features': ['comparison', 'sorting', 'labels'],
                'suitable_for': ['financial_metrics', 'comparison_data']
            },
            'risk_dashboard': {
                'type': ChartType.GAUGE,
                'features': ['threshold_marking', 'color_coding'],
                'suitable_for': ['risk_scores', 'performance_metrics']
            },
            'portfolio_distribution': {
                'type': ChartType.PIE,
                'features': ['percentage_labels', 'hover_details'],
                'suitable_for': ['distribution_data', 'composition_data']
            }
        }

    def generate_chart(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                       chart_context: Dict[str, Any] = None,
                       preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        🎯 智能生成图表 - 主要入口方法

        Args:
            data: 图表数据
            chart_context: 图表上下文（标题、描述等）
            preferences: 用户偏好设置

        Returns:
            完整的图表配置
        """
        try:
            logger.info("🎨 Starting intelligent chart generation...")

            # 1. 数据特征分析
            data_features = self._analyze_data_features(data)

            # 2. 智能推荐图表类型
            recommended_chart = self._recommend_chart_type(data_features, chart_context)

            # 3. 选择图表库
            chart_library = self._select_chart_library(recommended_chart, preferences)

            # 4. 生成图表配置
            chart_config = self._generate_chart_config(
                data, data_features, recommended_chart, chart_library, chart_context, preferences
            )

            # 5. 优化图表配置
            optimized_config = self._optimize_chart_config(chart_config, data_features)

            # 6. 添加交互功能
            interactive_config = self._add_interactive_features(optimized_config, data_features)

            # 7. 响应式配置
            responsive_config = self._make_responsive(interactive_config)

            # 8. 更新统计
            self._update_chart_stats(recommended_chart['type'], chart_library)

            result = {
                'success': True,
                'chart_config': responsive_config,
                'chart_metadata': {
                    'type': recommended_chart['type'].value,
                    'library': chart_library.value,
                    'theme': responsive_config.get('theme', 'default'),
                    'data_points': len(data) if isinstance(data, list) else 1,
                    'features': recommended_chart.get('features', []),
                    'generated_at': datetime.now().isoformat()
                },
                'data_insights': data_features,
                'recommendations': self._generate_chart_recommendations(data_features, recommended_chart)
            }

            logger.info(f"✅ Chart generated successfully: {recommended_chart['type'].value}")
            return result

        except Exception as e:
            logger.error(f"❌ Chart generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_config': self._create_fallback_chart(data)
            }

    def _analyze_data_features(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """分析数据特征"""
        try:
            features = {
                'data_type': 'unknown',
                'data_structure': 'unknown',
                'temporal': False,
                'numerical_fields': [],
                'categorical_fields': [],
                'time_fields': [],
                'data_size': 0,
                'value_ranges': {},
                'suggested_visualizations': []
            }

            # 确定数据结构
            if isinstance(data, list):
                features['data_structure'] = 'array'
                features['data_size'] = len(data)

                if data and isinstance(data[0], dict):
                    features['data_type'] = 'records'
                    # 分析字段类型
                    sample_record = data[0]
                    for key, value in sample_record.items():
                        if isinstance(value, (int, float)):
                            features['numerical_fields'].append(key)
                        elif isinstance(value, str):
                            # 检查是否是日期
                            if self._is_date_field(key, value):
                                features['time_fields'].append(key)
                                features['temporal'] = True
                            else:
                                features['categorical_fields'].append(key)

            elif isinstance(data, dict):
                features['data_structure'] = 'object'
                features['data_size'] = len(data)
                features['data_type'] = 'key_value'

                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        features['numerical_fields'].append(key)
                        features['value_ranges'][key] = {'min': value, 'max': value}
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        features['numerical_fields'].append(key)
                        features['value_ranges'][key] = {
                            'min': min(value),
                            'max': max(value),
                            'avg': sum(value) / len(value)
                        }

            # 计算数值范围
            if isinstance(data, list) and features['numerical_fields']:
                for field in features['numerical_fields']:
                    values = [record.get(field, 0) for record in data if isinstance(record.get(field), (int, float))]
                    if values:
                        features['value_ranges'][field] = {
                            'min': min(values),
                            'max': max(values),
                            'avg': sum(values) / len(values),
                            'range': max(values) - min(values)
                        }

            # 建议可视化类型
            features['suggested_visualizations'] = self._suggest_visualizations(features)

            return features

        except Exception as e:
            logger.error(f"Data feature analysis failed: {str(e)}")
            return {'data_type': 'unknown', 'error': str(e)}

    def _recommend_chart_type(self, data_features: Dict[str, Any],
                              chart_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """智能推荐图表类型"""
        try:
            context = chart_context or {}

            # 基于数据特征的推荐
            if data_features['temporal'] and len(data_features['numerical_fields']) >= 1:
                # 时间序列数据 → 折线图
                return {
                    'type': ChartType.LINE,
                    'confidence': 0.9,
                    'reason': '时间序列数据最适合使用折线图展示趋势',
                    'features': ['data_zoom', 'tooltip', 'trend_analysis']
                }

            elif len(data_features['numerical_fields']) >= 2 and data_features['data_size'] <= 50:
                # 双数值字段且数据量适中 → 散点图
                return {
                    'type': ChartType.SCATTER,
                    'confidence': 0.8,
                    'reason': '双数值字段适合使用散点图分析相关性',
                    'features': ['correlation_analysis', 'tooltip', 'data_zoom']
                }

            elif data_features['data_structure'] == 'key_value' and len(data_features['numerical_fields']) <= 8:
                # 键值对且字段不多 → 饼图或柱状图
                if any(keyword in context.get('title', '').lower() for keyword in
                       ['分布', '比例', '占比', 'distribution']):
                    return {
                        'type': ChartType.PIE,
                        'confidence': 0.85,
                        'reason': '分布数据适合使用饼图展示比例关系',
                        'features': ['percentage_labels', 'hover_details']
                    }
                else:
                    return {
                        'type': ChartType.BAR,
                        'confidence': 0.8,
                        'reason': '键值对数据适合使用柱状图对比',
                        'features': ['value_labels', 'sorting', 'comparison']
                    }

            elif any(keyword in context.get('title', '').lower() for keyword in ['风险', 'risk', '评分', 'score']):
                # 风险或评分数据 → 仪表盘
                return {
                    'type': ChartType.GAUGE,
                    'confidence': 0.9,
                    'reason': '风险评分数据适合使用仪表盘展示',
                    'features': ['threshold_marking', 'color_coding', 'range_indicators']
                }

            elif len(data_features['categorical_fields']) >= 1 and len(data_features['numerical_fields']) >= 1:
                # 分类 + 数值 → 柱状图
                return {
                    'type': ChartType.BAR,
                    'confidence': 0.75,
                    'reason': '分类数据与数值数据适合使用柱状图',
                    'features': ['category_labels', 'value_comparison', 'sorting']
                }

            else:
                # 默认推荐
                return {
                    'type': ChartType.LINE,
                    'confidence': 0.6,
                    'reason': '通用数据使用折线图展示',
                    'features': ['basic_interaction', 'tooltip']
                }

        except Exception as e:
            logger.error(f"Chart type recommendation failed: {str(e)}")
            return {
                'type': ChartType.BAR,
                'confidence': 0.5,
                'reason': f'推荐失败，使用默认图表: {str(e)}'
            }

    def _generate_chart_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                               data_features: Dict[str, Any],
                               recommended_chart: Dict[str, Any],
                               chart_library: ChartLibrary,
                               chart_context: Dict[str, Any] = None,
                               preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成图表配置"""
        try:
            context = chart_context or {}
            prefs = preferences or {}

            # 选择主题
            theme = prefs.get('theme', self.config['default_theme'].value)
            theme_config = self.themes.get(theme, self.themes['financial'])

            # 基础配置
            base_config = {
                'type': recommended_chart['type'].value,
                'library': chart_library.value,
                'theme': theme,
                'title': {
                    'text': context.get('title', '数据分析图表'),
                    'left': 'center',
                    'textStyle': theme_config['titleFont']
                },
                'backgroundColor': theme_config['backgroundColor'],
                'color': self.config['color_palettes'].get(theme, self.config['color_palettes']['financial']),
                'animation': True,
                'animationDuration': 1000
            }

            # 根据图表类型生成具体配置
            if chart_library == ChartLibrary.ECHARTS:
                chart_config = self._generate_echarts_config(
                    data, data_features, recommended_chart, base_config, theme_config
                )
            elif chart_library == ChartLibrary.CHARTJS:
                chart_config = self._generate_chartjs_config(
                    data, data_features, recommended_chart, base_config, theme_config
                )
            else:
                chart_config = base_config

            return chart_config

        except Exception as e:
            logger.error(f"Chart config generation failed: {str(e)}")
            return {'error': str(e)}

    def _generate_echarts_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                 data_features: Dict[str, Any],
                                 recommended_chart: Dict[str, Any],
                                 base_config: Dict[str, Any],
                                 theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成ECharts配置"""
        chart_type = recommended_chart['type']
        config = base_config.copy()

        # 工具箱
        config['toolbox'] = {
            'feature': {
                'saveAsImage': {'title': '保存图片'},
                'dataZoom': {'title': '区域缩放'},
                'restore': {'title': '重置'}
            }
        }

        # 提示框
        config['tooltip'] = {
            'trigger': 'axis' if chart_type in [ChartType.LINE, ChartType.BAR] else 'item',
            'backgroundColor': 'rgba(50,50,50,0.8)',
            'textStyle': {'color': '#fff'}
        }

        if chart_type == ChartType.LINE:
            config.update(self._generate_line_chart_config(data, data_features, theme_config))
        elif chart_type == ChartType.BAR:
            config.update(self._generate_bar_chart_config(data, data_features, theme_config))
        elif chart_type == ChartType.PIE:
            config.update(self._generate_pie_chart_config(data, data_features, theme_config))
        elif chart_type == ChartType.SCATTER:
            config.update(self._generate_scatter_chart_config(data, data_features, theme_config))
        elif chart_type == ChartType.GAUGE:
            config.update(self._generate_gauge_chart_config(data, data_features, theme_config))

        return config

    def _generate_line_chart_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                    data_features: Dict[str, Any],
                                    theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成折线图配置"""
        config = {}

        # 处理数据
        if isinstance(data, list) and data_features['time_fields']:
            # 时间序列数据
            time_field = data_features['time_fields'][0]
            numeric_fields = data_features['numerical_fields']

            # X轴（时间）
            config['xAxis'] = {
                'type': 'category',
                'data': [item.get(time_field, '') for item in data],
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            # Y轴
            config['yAxis'] = {
                'type': 'value',
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']},
                'splitLine': {'lineStyle': {'color': theme_config['gridColor']}}
            }

            # 系列数据
            config['series'] = []
            for i, field in enumerate(numeric_fields[:3]):  # 最多3个系列
                series_data = [item.get(field, 0) for item in data]
                config['series'].append({
                    'name': field,
                    'type': 'line',
                    'data': series_data,
                    'smooth': True,
                    'symbol': 'circle',
                    'symbolSize': 4,
                    'lineStyle': {'width': 2}
                })

            # 图例
            if len(config['series']) > 1:
                config['legend'] = {
                    'data': [series['name'] for series in config['series']],
                    'top': 30,
                    'textStyle': {'color': theme_config['textColor']}
                }

        elif isinstance(data, dict):
            # 键值对数据
            keys = list(data.keys())
            values = list(data.values())

            config['xAxis'] = {
                'type': 'category',
                'data': keys,
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            config['yAxis'] = {
                'type': 'value',
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            config['series'] = [{
                'type': 'line',
                'data': values,
                'smooth': True,
                'areaStyle': {'opacity': 0.3}
            }]

        # 数据缩放
        config['dataZoom'] = [
            {
                'type': 'slider',
                'start': 0,
                'end': 100,
                'height': 20,
                'bottom': 10
            }
        ]

        return config

    def _generate_bar_chart_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                   data_features: Dict[str, Any],
                                   theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成柱状图配置"""
        config = {}

        if isinstance(data, dict):
            # 键值对数据
            items = list(data.items())
            # 按值排序
            items.sort(key=lambda x: x[1], reverse=True)

            config['xAxis'] = {
                'type': 'category',
                'data': [item[0] for item in items],
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {
                    'color': theme_config['textColor'],
                    'rotate': 30 if len(items) > 8 else 0
                }
            }

            config['yAxis'] = {
                'type': 'value',
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']},
                'splitLine': {'lineStyle': {'color': theme_config['gridColor']}}
            }

            config['series'] = [{
                'type': 'bar',
                'data': [item[1] for item in items],
                'barWidth': '60%',
                'itemStyle': {
                    'borderRadius': [4, 4, 0, 0],
                    'shadowBlur': 3,
                    'shadowOffsetY': 2,
                    'shadowColor': 'rgba(0,0,0,0.1)'
                },
                'label': {
                    'show': len(items) <= 10,
                    'position': 'top',
                    'color': theme_config['textColor']
                }
            }]

        elif isinstance(data, list) and data_features['categorical_fields'] and data_features['numerical_fields']:
            # 分类数据
            cat_field = data_features['categorical_fields'][0]
            num_field = data_features['numerical_fields'][0]

            categories = [item.get(cat_field, '') for item in data]
            values = [item.get(num_field, 0) for item in data]

            config['xAxis'] = {
                'type': 'category',
                'data': categories,
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            config['yAxis'] = {
                'type': 'value',
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            config['series'] = [{
                'type': 'bar',
                'data': values,
                'barWidth': '60%'
            }]

        return config

    def _generate_pie_chart_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                   data_features: Dict[str, Any],
                                   theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成饼图配置"""
        config = {}

        if isinstance(data, dict):
            # 键值对数据
            pie_data = [{'name': k, 'value': v} for k, v in data.items()]
        elif isinstance(data, list) and data_features['categorical_fields'] and data_features['numerical_fields']:
            # 分类数据
            cat_field = data_features['categorical_fields'][0]
            num_field = data_features['numerical_fields'][0]
            pie_data = [{'name': item.get(cat_field, ''), 'value': item.get(num_field, 0)} for item in data]
        else:
            pie_data = []

        config['series'] = [{
            'type': 'pie',
            'radius': ['40%', '70%'],
            'center': ['50%', '50%'],
            'data': pie_data,
            'emphasis': {
                'itemStyle': {
                    'shadowBlur': 10,
                    'shadowOffsetX': 0,
                    'shadowColor': 'rgba(0, 0, 0, 0.5)'
                }
            },
            'label': {
                'show': True,
                'formatter': '{b}: {c} ({d}%)',
                'color': theme_config['textColor']
            },
            'labelLine': {
                'show': True
            }
        }]

        config['legend'] = {
            'orient': 'horizontal',
            'bottom': 10,
            'data': [item['name'] for item in pie_data],
            'textStyle': {'color': theme_config['textColor']}
        }

        return config

    def _generate_scatter_chart_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                       data_features: Dict[str, Any],
                                       theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成散点图配置"""
        config = {}

        if isinstance(data, list) and len(data_features['numerical_fields']) >= 2:
            x_field = data_features['numerical_fields'][0]
            y_field = data_features['numerical_fields'][1]

            scatter_data = [[item.get(x_field, 0), item.get(y_field, 0)] for item in data]

            config['xAxis'] = {
                'type': 'value',
                'name': x_field,
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            config['yAxis'] = {
                'type': 'value',
                'name': y_field,
                'axisLine': {'lineStyle': {'color': theme_config['axisColor']}},
                'axisLabel': {'color': theme_config['textColor']}
            }

            config['series'] = [{
                'type': 'scatter',
                'data': scatter_data,
                'symbolSize': 8,
                'itemStyle': {
                    'shadowBlur': 2,
                    'shadowOffsetY': 1,
                    'shadowColor': 'rgba(0,0,0,0.3)'
                }
            }]

        return config

    def _generate_gauge_chart_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                     data_features: Dict[str, Any],
                                     theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成仪表盘配置"""
        config = {}

        # 提取数值
        if isinstance(data, dict) and data_features['numerical_fields']:
            value = data.get(data_features['numerical_fields'][0], 0)
        elif isinstance(data, list) and data and data_features['numerical_fields']:
            value = data[0].get(data_features['numerical_fields'][0], 0)
        else:
            value = 0

        # 确定最大值
        max_value = 100
        if data_features['value_ranges']:
            field_ranges = list(data_features['value_ranges'].values())
            if field_ranges:
                max_value = max(range_info.get('max', 100) for range_info in field_ranges)
                max_value = max_value * 1.2  # 留一些余量

        config['series'] = [{
            'type': 'gauge',
            'center': ['50%', '60%'],
            'startAngle': 200,
            'endAngle': -40,
            'min': 0,
            'max': max_value,
            'splitNumber': 10,
            'radius': '75%',
            'data': [{'value': value, 'name': '当前值'}],
            'detail': {
                'valueAnimation': True,
                'formatter': '{value}',
                'color': theme_config['textColor']
            },
            'axisLine': {
                'lineStyle': {
                    'width': 6,
                    'color': [
                        [0.3, '#67e0e3'],
                        [0.7, '#37a2da'],
                        [1, '#fd666d']
                    ]
                }
            },
            'pointer': {
                'itemStyle': {
                    'color': 'auto'
                }
            },
            'axisTick': {
                'distance': -30,
                'length': 8,
                'lineStyle': {
                    'color': '#fff',
                    'width': 2
                }
            },
            'splitLine': {
                'distance': -30,
                'length': 30,
                'lineStyle': {
                    'color': '#fff',
                    'width': 4
                }
            },
            'axisLabel': {
                'color': theme_config['textColor'],
                'distance': 40,
                'fontSize': 12
            },
            'title': {
                'offsetCenter': [0, '-10%'],
                'fontSize': 16,
                'color': theme_config['textColor']
            }
        }]

        return config

    def _generate_chartjs_config(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                 data_features: Dict[str, Any],
                                 recommended_chart: Dict[str, Any],
                                 base_config: Dict[str, Any],
                                 theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成Chart.js配置"""
        chart_type = recommended_chart['type']

        # Chart.js基础配置结构
        config = {
            'type': self._echarts_to_chartjs_type(chart_type),
            'data': {},
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': base_config['title']['text'],
                        'color': theme_config['textColor']
                    },
                    'legend': {
                        'labels': {
                            'color': theme_config['textColor']
                        }
                    }
                },
                'scales': {
                    'x': {
                        'ticks': {
                            'color': theme_config['textColor']
                        },
                        'grid': {
                            'color': theme_config['gridColor']
                        }
                    },
                    'y': {
                        'ticks': {
                            'color': theme_config['textColor']
                        },
                        'grid': {
                            'color': theme_config['gridColor']
                        }
                    }
                }
            }
        }

        # 根据图表类型生成数据
        if chart_type == ChartType.LINE:
            config['data'] = self._generate_chartjs_line_data(data, data_features)
        elif chart_type == ChartType.BAR:
            config['data'] = self._generate_chartjs_bar_data(data, data_features)
        elif chart_type == ChartType.PIE:
            config['data'] = self._generate_chartjs_pie_data(data, data_features)
            # 饼图不需要scales
            del config['options']['scales']

        return config

    def _echarts_to_chartjs_type(self, echarts_type: ChartType) -> str:
        """ECharts图表类型转换为Chart.js类型"""
        mapping = {
            ChartType.LINE: 'line',
            ChartType.BAR: 'bar',
            ChartType.PIE: 'pie',
            ChartType.SCATTER: 'scatter',
            ChartType.AREA: 'line'
        }
        return mapping.get(echarts_type, 'bar')

    def _generate_chartjs_line_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                    data_features: Dict[str, Any]) -> Dict[str, Any]:
        """生成Chart.js折线图数据"""
        if isinstance(data, dict):
            return {
                'labels': list(data.keys()),
                'datasets': [{
                    'label': '数据',
                    'data': list(data.values()),
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.1
                }]
            }
        elif isinstance(data, list) and data_features['time_fields']:
            time_field = data_features['time_fields'][0]
            numeric_fields = data_features['numerical_fields']

            datasets = []
            colors = ['rgb(75, 192, 192)', 'rgb(255, 99, 132)', 'rgb(54, 162, 235)']

            for i, field in enumerate(numeric_fields[:3]):
                datasets.append({
                    'label': field,
                    'data': [item.get(field, 0) for item in data],
                    'borderColor': colors[i % len(colors)],
                    'backgroundColor': colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                    'tension': 0.1
                })

            return {
                'labels': [item.get(time_field, '') for item in data],
                'datasets': datasets
            }

        return {'labels': [], 'datasets': []}

    def _generate_chartjs_bar_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                   data_features: Dict[str, Any]) -> Dict[str, Any]:
        """生成Chart.js柱状图数据"""
        if isinstance(data, dict):
            return {
                'labels': list(data.keys()),
                'datasets': [{
                    'label': '数据',
                    'data': list(data.values()),
                    'backgroundColor': [
                        'rgba(255, 99, 132, 0.6)',
                        'rgba(54, 162, 235, 0.6)',
                        'rgba(255, 205, 86, 0.6)',
                        'rgba(75, 192, 192, 0.6)',
                        'rgba(153, 102, 255, 0.6)'
                    ]
                }]
            }

        return {'labels': [], 'datasets': []}

    def _generate_chartjs_pie_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                   data_features: Dict[str, Any]) -> Dict[str, Any]:
        """生成Chart.js饼图数据"""
        if isinstance(data, dict):
            return {
                'labels': list(data.keys()),
                'datasets': [{
                    'data': list(data.values()),
                    'backgroundColor': [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF',
                        '#FF9F40'
                    ]
                }]
            }

        return {'labels': [], 'datasets': []}

    def _select_chart_library(self, recommended_chart: Dict[str, Any],
                              preferences: Dict[str, Any] = None) -> ChartLibrary:
        """选择图表库"""
        prefs = preferences or {}

        # 用户偏好优先
        if 'library' in prefs:
            try:
                return ChartLibrary(prefs['library'])
            except ValueError:
                pass

        # 基于图表类型选择
        chart_type = recommended_chart['type']

        # ECharts适合复杂图表
        if chart_type in [ChartType.GAUGE, ChartType.RADAR, ChartType.HEATMAP, ChartType.CANDLESTICK]:
            return ChartLibrary.ECHARTS

        # Chart.js适合简单图表
        elif chart_type in [ChartType.LINE, ChartType.BAR, ChartType.PIE]:
            return ChartLibrary.CHARTJS

        # 默认使用ECharts
        return self.config['default_library']

    def _optimize_chart_config(self, chart_config: Dict[str, Any],
                               data_features: Dict[str, Any]) -> Dict[str, Any]:
        """优化图表配置"""
        optimized = chart_config.copy()

        # 数据量优化
        if data_features['data_size'] > self.config['max_data_points']:
            # 大数据量时启用数据采样
            optimized['sampling'] = {'enabled': True, 'threshold': self.config['max_data_points']}

        # 性能优化
        if data_features['data_size'] > 500:
            # 大数据量时禁用动画
            optimized['animation'] = False

        # 移动端优化
        optimized['mobile_optimized'] = {
            'legend': {'orient': 'horizontal', 'bottom': 0},
            'grid': {'left': '10%', 'right': '10%', 'top': '15%', 'bottom': '20%'}
        }

        return optimized

    def _add_interactive_features(self, chart_config: Dict[str, Any],
                                  data_features: Dict[str, Any]) -> Dict[str, Any]:
        """添加交互功能"""
        interactive = chart_config.copy()

        # 基础交互功能
        if chart_config.get('library') == 'echarts':
            # 添加缩放功能（时间序列图表）
            if data_features['temporal']:
                if 'dataZoom' not in interactive:
                    interactive['dataZoom'] = [
                        {'type': 'slider', 'start': 70, 'end': 100},
                        {'type': 'inside'}
                    ]

            # 添加图例交互
            if 'legend' in interactive:
                interactive['legend']['selectedMode'] = 'multiple'

            # 添加刷选功能（散点图）
            if chart_config.get('type') == 'scatter':
                interactive['brush'] = {
                    'toolbox': ['rect', 'polygon', 'clear'],
                    'xAxisIndex': 'all',
                    'yAxisIndex': 'all'
                }

        return interactive

    def _make_responsive(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """使图表响应式"""
        responsive = chart_config.copy()

        # 响应式配置
        responsive['responsive'] = True
        responsive['breakpoints'] = self.config['responsive_breakpoints']

        # 移动端适配
        responsive['mobile'] = {
            'title': {'textStyle': {'fontSize': 14}},
            'legend': {'itemWidth': 20, 'itemHeight': 10, 'textStyle': {'fontSize': 10}},
            'tooltip': {'textStyle': {'fontSize': 10}}
        }

        # 平板适配
        responsive['tablet'] = {
            'title': {'textStyle': {'fontSize': 15}},
            'legend': {'itemWidth': 22, 'itemHeight': 12, 'textStyle': {'fontSize': 11}}
        }

        return responsive

    def _suggest_visualizations(self, features: Dict[str, Any]) -> List[str]:
        """建议可视化类型"""
        suggestions = []

        if features['temporal']:
            suggestions.extend(['line', 'area', 'candlestick'])

        if len(features['numerical_fields']) >= 2:
            suggestions.append('scatter')

        if features['data_structure'] == 'key_value':
            suggestions.extend(['bar', 'pie', 'treemap'])

        if any('risk' in field.lower() or 'score' in field.lower() for field in features['numerical_fields']):
            suggestions.append('gauge')

        return list(set(suggestions))

    def _is_date_field(self, field_name: str, field_value: str) -> bool:
        """判断是否是日期字段"""
        date_indicators = ['date', 'time', '日期', '时间', 'created', 'updated']

        # 字段名包含日期关键词
        if any(indicator in field_name.lower() for indicator in date_indicators):
            return True

        # 值格式像日期
        if isinstance(field_value, str):
            # 简单的日期格式检查
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                r'\d{8}',  # YYYYMMDD
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'  # YYYY-MM-DD HH:MM:SS
            ]
            return any(re.match(pattern, field_value) for pattern in date_patterns)

        return False

    def _create_fallback_chart(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """创建后备图表配置"""
        return {
            'type': 'bar',
            'library': 'echarts',
            'title': {'text': '数据图表'},
            'series': [{
                'type': 'bar',
                'data': [1, 2, 3, 4, 5] if not data else []
            }],
            'xAxis': {'type': 'category', 'data': ['A', 'B', 'C', 'D', 'E']},
            'yAxis': {'type': 'value'}
        }

    def _generate_chart_recommendations(self, data_features: Dict[str, Any],
                                        recommended_chart: Dict[str, Any]) -> List[str]:
        """生成图表使用建议"""
        recommendations = []

        # 基于图表类型的建议
        chart_type = recommended_chart['type']

        if chart_type == ChartType.LINE:
            recommendations.append("折线图适合展示趋势变化，建议关注数据的连续性")
            if data_features['data_size'] > 100:
                recommendations.append("数据量较大，建议使用数据缩放功能")

        elif chart_type == ChartType.PIE:
            recommendations.append("饼图适合展示比例关系，建议控制分类数量在8个以内")
            if len(data_features.get('categorical_fields', [])) > 8:
                recommendations.append("分类过多，考虑合并小比例项或使用柱状图")

        elif chart_type == ChartType.BAR:
            recommendations.append("柱状图适合对比分析，建议按数值大小排序")

        elif chart_type == ChartType.GAUGE:
            recommendations.append("仪表盘适合展示单一指标的状态，建议设置合理的阈值")

        # 基于数据特征的建议
        if data_features['data_size'] > 1000:
            recommendations.append("数据量较大，建议启用数据采样或分页显示")

        if data_features['temporal']:
            recommendations.append("包含时间数据，建议添加时间范围选择器")

        return recommendations

    def _update_chart_stats(self, chart_type: ChartType, library: ChartLibrary):
        """更新图表生成统计"""
        self.stats['total_charts_generated'] += 1

        type_name = chart_type.value
        if type_name not in self.stats['chart_type_distribution']:
            self.stats['chart_type_distribution'][type_name] = 0
        self.stats['chart_type_distribution'][type_name] += 1

        lib_name = library.value
        if lib_name not in self.stats['library_usage']:
            self.stats['library_usage'][lib_name] = 0
        self.stats['library_usage'][lib_name] += 1

    # ============= 便捷方法 =============

    def generate_financial_overview_chart(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成财务概览图表"""
        return self.generate_chart(
            financial_data,
            chart_context={'title': '财务概览', 'category': 'financial'},
            preferences={'theme': 'financial', 'library': 'echarts'}
        )

    def generate_trend_chart(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成趋势图表"""
        return self.generate_chart(
            time_series_data,
            chart_context={'title': '趋势分析', 'category': 'trend'},
            preferences={'chart_type': 'line'}
        )

    def generate_distribution_chart(self, distribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成分布图表"""
        return self.generate_chart(
            distribution_data,
            chart_context={'title': '数据分布', 'category': 'distribution'},
            preferences={'chart_type': 'pie'}
        )

    def generate_risk_gauge(self, risk_score: float, max_score: float = 100) -> Dict[str, Any]:
        """生成风险评分仪表盘"""
        risk_data = {'风险评分': risk_score, '最大值': max_score}
        return self.generate_chart(
            risk_data,
            chart_context={'title': '风险评估仪表盘', 'category': 'risk'},
            preferences={'chart_type': 'gauge', 'theme': 'financial'}
        )

    def get_chart_stats(self) -> Dict[str, Any]:
        """获取图表生成统计"""
        return {
            'generation_stats': self.stats,
            'supported_types': [chart_type.value for chart_type in ChartType],
            'supported_libraries': [lib.value for lib in ChartLibrary],
            'supported_themes': list(self.themes.keys())
        }


# ============= 工厂函数 =============

def create_chart_generator() -> ChartGenerator:
    """
    创建图表生成器实例

    Returns:
        ChartGenerator实例
    """
    return ChartGenerator()


# ============= 使用示例 =============

def main():
    """使用示例"""
    generator = create_chart_generator()

    # 示例1：财务数据柱状图
    financial_data = {
        '总余额': 8223695,
        '总投资': 30772686,
        '总入金': 17227143,
        '总出金': 9003448
    }

    result1 = generator.generate_financial_overview_chart(financial_data)
    print("财务概览图表:")
    print(f"图表类型: {result1['chart_metadata']['type']}")
    print(f"图表库: {result1['chart_metadata']['library']}")

    # 示例2：时间序列折线图
    trend_data = [
        {'date': '2025-01-01', 'revenue': 100000, 'cost': 80000},
        {'date': '2025-01-02', 'revenue': 120000, 'cost': 85000},
        {'date': '2025-01-03', 'revenue': 110000, 'cost': 82000},
        {'date': '2025-01-04', 'revenue': 130000, 'cost': 88000}
    ]

    result2 = generator.generate_trend_chart(trend_data)
    print("\n趋势分析图表:")
    print(f"图表类型: {result2['chart_metadata']['type']}")
    print(f"特征: {result2['chart_metadata']['features']}")

    # 示例3：风险评分仪表盘
    result3 = generator.generate_risk_gauge(75.5, 100)
    print("\n风险评分仪表盘:")
    print(f"图表类型: {result3['chart_metadata']['type']}")

    # 获取统计信息
    stats = generator.get_chart_stats()
    print(f"\n图表生成统计: {stats['generation_stats']}")


if __name__ == "__main__":
    main()