"""
语义化数据收集器 - 第一层
负责将原始API结果转换为有语义含义的数据结构
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class SemanticDataCollector:
    """语义化数据收集器"""

    def __init__(self):
        self.current_date = datetime.now()

    def organize_semantic_data(self,
                               raw_api_results: Dict[str, Any],
                               query_analysis) -> Dict[str, Any]:
        """
        将原始API结果组织成语义化的数据结构

        Args:
            raw_api_results: 来自API的原始结果
            query_analysis: 查询分析结果

        Returns:
            语义化组织的数据
        """
        try:
            if not raw_api_results.get('success') or not raw_api_results.get('results'):
                return {'error': '无有效的API数据'}

            semantic_data = {}
            api_calls = getattr(query_analysis, 'api_calls', [])

            # 遍历所有API结果，进行语义化分类
            for result_key, result_data in raw_api_results['results'].items():
                if not result_data.get('success') or not result_data.get('data'):
                    continue

                # 提取API调用索引
                api_index = self._extract_api_index(result_key)

                # 获取对应的API调用信息
                api_call_info = None
                if api_index < len(api_calls):
                    api_call_info = api_calls[api_index]

                # 生成语义化标识
                semantic_key = self._generate_semantic_key(
                    result_key, result_data, api_call_info
                )

                # 提取并清理数据
                clean_data = self._extract_clean_data(result_data['data'])

                # 添加时间上下文
                time_context = self._extract_time_context(api_call_info)

                semantic_data[semantic_key] = {
                    'data': clean_data,
                    'api_info': {
                        'method': result_key.split('_')[0] if '_' in result_key else 'unknown',
                        'success': result_data.get('success', False),
                        'original_key': result_key
                    },
                    'time_context': time_context,
                    'data_quality': self._assess_data_quality(clean_data),
                    'api_call_info': api_call_info
                }

            return {
                'semantic_data': semantic_data,
                'collection_summary': self._generate_collection_summary(semantic_data),
                'collection_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"语义化数据收集失败: {e}")
            return {'error': f'语义化收集失败: {str(e)}'}

    def _extract_api_index(self, result_key: str) -> int:
        """从结果键中提取API调用索引"""
        try:
            # result_key 格式如: "get_daily_data_0", "get_system_data_1"
            parts = result_key.split('_')
            if parts and parts[-1].isdigit():
                return int(parts[-1])
            return 0
        except:
            return 0

    def _generate_semantic_key(self,
                               result_key: str,
                               result_data: Dict[str, Any],
                               api_call_info: Optional[Dict[str, Any]]) -> str:
        """生成语义化的数据键"""

        if not api_call_info:
            return f'general_{result_key}'

        reason = api_call_info.get('reason', '')
        method = api_call_info.get('method', '')
        params = api_call_info.get('params', {})

        # 🎯 时间语义识别（最重要）
        if '本周' in reason:
            return f'current_week_{method}'
        elif '上周' in reason:
            return f'last_week_{method}'
        elif '今天' in reason or '今日' in reason:
            return f'today_{method}'
        elif '昨天' in reason or '昨日' in reason:
            return f'yesterday_{method}'
        elif '前天' in reason:
            return f'day_before_yesterday_{method}'
        elif '明天' in reason:
            return f'tomorrow_{method}'

        # 🎯 日期参数语义识别
        if 'date' in params:
            date_str = params['date']
            date_semantic = self._classify_date_semantic(date_str)
            return f'{date_semantic}_{method}'

        # 🎯 区间语义识别
        if 'start_date' in params and 'end_date' in params:
            start_date = params['start_date']
            end_date = params['end_date']
            interval_semantic = self._classify_interval_semantic(start_date, end_date)
            return f'{interval_semantic}_{method}'

        # 🎯 业务功能语义识别
        if '系统' in reason or '概览' in reason:
            return f'system_overview_{method}'
        elif '产品' in reason and '到期' in reason:
            return f'product_expiry_{method}'
        elif '产品' in reason:
            return f'product_info_{method}'
        elif '用户' in reason:
            return f'user_info_{method}'

        # 默认语义
        return f'general_{method}_{hash(reason) % 1000}'

    def _classify_date_semantic(self, date_str: str) -> str:
        """分类日期的语义含义"""
        try:
            target_date = datetime.strptime(date_str, '%Y%m%d')
            today = self.current_date.date()
            target_date_only = target_date.date()

            diff_days = (target_date_only - today).days

            if diff_days == 0:
                return 'today'
            elif diff_days == -1:
                return 'yesterday'
            elif diff_days == -2:
                return 'day_before_yesterday'
            elif diff_days == 1:
                return 'tomorrow'
            elif -7 <= diff_days <= -1:
                return 'recent_past'
            elif 1 <= diff_days <= 7:
                return 'near_future'
            else:
                return f'date_{date_str}'
        except:
            return f'date_{date_str}'

    def _classify_interval_semantic(self, start_date: str, end_date: str) -> str:
        """分类区间的语义含义"""
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 计算区间天数
            interval_days = (end_dt - start_dt).days + 1

            # 判断是否是完整的周或月
            if interval_days == 7:
                return 'week_interval'
            elif 28 <= interval_days <= 31:
                return 'month_interval'
            elif interval_days <= 7:
                return 'short_interval'
            elif interval_days <= 14:
                return 'biweek_interval'
            else:
                return 'long_interval'
        except:
            return f'interval_{start_date}_{end_date}'

    def _extract_clean_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取并清理数据"""
        clean_data = {}

        for key, value in raw_data.items():
            if isinstance(value, (str, int, float)):
                # 尝试转换数值
                try:
                    if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        clean_data[key] = float(value) if '.' in value else int(value)
                    else:
                        clean_data[key] = value
                except:
                    clean_data[key] = value
            elif isinstance(value, dict) and len(value) < 20:  # 避免嵌套太深
                clean_data[key] = value
            elif isinstance(value, list) and len(value) < 100:  # 避免列表太长
                clean_data[key] = value

        return clean_data

    def _extract_time_context(self, api_call_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """提取时间上下文信息"""
        if not api_call_info:
            return {}

        params = api_call_info.get('params', {})
        time_context = {}

        # 提取日期参数
        if 'date' in params:
            time_context['target_date'] = params['date']
            time_context['time_type'] = 'single_date'
        elif 'start_date' in params and 'end_date' in params:
            time_context['start_date'] = params['start_date']
            time_context['end_date'] = params['end_date']
            time_context['time_type'] = 'date_range'
        else:
            time_context['time_type'] = 'current'

        return time_context

    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估数据质量"""
        if not data:
            return {'score': 0.0, 'issues': ['数据为空']}

        score = 1.0
        issues = []

        # 检查关键字段
        key_fields = ['入金', '出金', '注册人数', '日期']
        missing_fields = [field for field in key_fields if field not in data]

        if missing_fields:
            score -= 0.2 * len(missing_fields)
            issues.append(f'缺少关键字段: {missing_fields}')

        # 检查数值合理性
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if value < 0 and key in ['入金', '注册人数']:
                    score -= 0.1
                    issues.append(f'{key}值异常: {value}')

        return {
            'score': max(0.0, score),
            'issues': issues,
            'field_count': len(data)
        }

    def _generate_collection_summary(self, semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成收集摘要"""
        return {
            'total_semantic_keys': len(semantic_data),
            'semantic_categories': list(set(key.split('_')[0] for key in semantic_data.keys())),
            'time_coverage': self._analyze_time_coverage(semantic_data),
            'data_types': list(set(
                data['api_info']['method'] for data in semantic_data.values()
            ))
        }

    def _analyze_time_coverage(self, semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析时间覆盖范围"""
        time_keys = []

        for key in semantic_data.keys():
            if any(time_word in key for time_word in
                   ['today', 'yesterday', 'current_week', 'last_week', 'date_']):
                time_keys.append(key)

        return {
            'has_time_data': len(time_keys) > 0,
            'time_keys': time_keys,
            'coverage_type': 'comparison' if len(time_keys) >= 2 else 'single_point'
        }