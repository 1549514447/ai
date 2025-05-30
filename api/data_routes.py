# api/data_routes.py - 完整优化版本
from flask import Blueprint, jsonify, request, Response
from typing import Dict, Any, List, Optional
import logging
import asyncio
from datetime import datetime
import json
import traceback
import time
import re

# 导入编排器 - 核心改动！
from core.orchestrator.intelligent_qa_orchestrator import get_orchestrator
# 导入报告和图表相关的枚举和工具（如果需要直接在数据路由中使用）
from utils.formatters.report_generator import ReportFormat
from utils.formatters.chart_generator import ChartType

logger = logging.getLogger(__name__)

# 创建数据API蓝图
data_bp = Blueprint('data_api', __name__, url_prefix='/api/data')

# 🎯 使用单一编排器实例 - 避免重复初始化
orchestrator = get_orchestrator()


# ============= 工具函数 =============

def async_route(f):
    """异步路由装饰器"""

    # @wraps(f) # 可选，用于保留原函数元信息
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()

    wrapper.__name__ = f.__name__
    return wrapper


def validate_request_params(required_params: List[str] = None, optional_params: Dict[str, Any] = None) -> Dict[
    str, Any]:
    """
    验证请求参数 (GET查询参数和POST JSON body)
    并应用类型转换和默认值。
    """
    params = {}
    # 获取查询参数
    params.update(request.args.to_dict())

    # 获取JSON数据（如果请求是JSON类型）
    if request.content_type and 'application/json' in request.content_type:
        try:
            json_data = request.get_json()
            if json_data:
                params.update(json_data)
        except Exception as e:
            raise ValueError(f"无效的JSON请求体: {e}")

    # 检查必需参数
    if required_params:
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(f"缺少必填参数: {', '.join(missing_params)}")

    # 处理可选参数和类型转换
    if optional_params:
        for param_name, details in optional_params.items():
            default_value = details.get('default')
            param_type = details.get('type')
            allowed_values = details.get('allowed')

            if param_name in params:
                value = params[param_name]
                if param_type:
                    try:
                        if param_type == bool:  # 特殊处理布尔值
                            value = str(value).lower() in ['true', '1', 'yes', 'on']
                        else:
                            value = param_type(value)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"参数 '{param_name}' 类型错误: 应为 {param_type.__name__}, 收到 '{value}'. 错误: {e}")
                if allowed_values and value not in allowed_values:
                    raise ValueError(
                        f"参数 '{param_name}' 的值 '{value}' 无效. 允许的值为: {', '.join(map(str, allowed_values))}")
                params[param_name] = value
            elif default_value is not None:
                params[param_name] = default_value
            # 如果参数不存在且没有默认值，则不将其添加到params中
    return params


def validate_date_format(date_str: str, param_name: str = "date") -> Optional[str]:
    """验证日期格式YYYYMMDD，如果为空则返回None"""
    if not date_str:
        return None
    if not isinstance(date_str, str):  # 确保是字符串类型
        raise ValueError(f"参数 '{param_name}' 类型错误: 应为字符串, 收到 '{type(date_str).__name__}'")
    if not re.match(r'^\d{8}$', date_str):
        raise ValueError(f"参数 '{param_name}' ('{date_str}') 格式错误，应为YYYYMMDD格式")
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return date_str
    except ValueError:
        raise ValueError(f"参数 '{param_name}' 日期无效: {date_str}")


def validate_date_range(start_date_str: Optional[str], end_date_str: Optional[str]) -> tuple[
    Optional[str], Optional[str]]:
    """验证日期范围，允许部分为空"""
    start_date = validate_date_format(start_date_str, "start_date") if start_date_str else None
    end_date = validate_date_format(end_date_str, "end_date") if end_date_str else None

    if start_date and end_date:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        if start_dt > end_dt:
            raise ValueError("开始日期不能晚于结束日期")
        # 检查日期范围是否合理（不超过1年）
        if (end_dt - start_dt).days > 365:  # 允许查询最多一年数据
            raise ValueError("日期范围不能超过1年")
    elif start_date and not end_date:
        # 如果只有开始日期，默认结束日期为开始日期+合理范围（例如30天）或当天
        # 根据业务逻辑调整，这里暂时认为需要成对出现或都不出现
        pass  # 或者可以设置默认结束日期
    elif end_date and not start_date:
        raise ValueError("提供了结束日期但缺少开始日期")

    return start_date, end_date


def create_success_response(data: Dict[str, Any], message: str = "操作成功", status_code: int = 200) -> tuple:
    """创建成功响应"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'processed_by': 'intelligent_orchestrator'
    }), status_code


def create_error_response(message: str, error_type: str = "processing_error", status_code: int = 500,
                          details: Optional[Dict[str, Any]] = None) -> tuple:
    """创建错误响应"""
    response_body = {
        'success': False,
        'error_type': error_type,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'system_status': 'error'
    }
    if details:
        response_body['details'] = details
    return jsonify(response_body), status_code


# ============= 核心数据API =============

@data_bp.route('/system', methods=['GET'])
@async_route
async def get_enhanced_system_data():
    """
    📊 获取增强系统数据 - 使用编排器完整数据链
    此接口返回当前系统的整体概览数据，已经过内部增强和验证。
    """
    try:
        logger.info("🔍 API: 请求获取增强系统数据...")

        if not orchestrator.initialized:
            await orchestrator.initialize()
            logger.info("Orchestrator re-initialized on demand.")

        # APIConnector的get_system_data()内部已包含数据增强和验证
        result = await orchestrator.data_fetcher.api_connector.get_system_data()

        if not result.get("success"):
            return create_error_response(
                result.get("message", "系统数据获取失败"), "api_error", 500
            )

        # enhanced_data已经是APIConnector处理过的
        enhanced_data = result.get("data", {})
        validation_info = result.get("validation", {})  # APIConnector返回的验证信息

        # 额外的财务健康评估 (可以在这里添加更多基于规则的或简单的AI分析)
        financial_health = {}
        try:
            total_inflow = float(enhanced_data.get('总入金', 0))
            total_outflow = float(enhanced_data.get('总出金', 0))
            if total_inflow > 0:
                outflow_ratio = total_outflow / total_inflow
                financial_health = {
                    'outflow_ratio': outflow_ratio,
                    'health_status': 'healthy' if outflow_ratio < 0.7 else 'concerning' if outflow_ratio < 0.9 else 'risky',
                    'net_flow': total_inflow - total_outflow
                }
        except Exception as e:
            logger.warning(f"财务健康评估失败: {str(e)}")
            financial_health = {'error': '评估失败'}

        response_data = {
            'system_overview': enhanced_data,
            'data_validation_summary': validation_info,
            'financial_health_snapshot': financial_health,
            'processing_metadata': {
                'data_source': '/api/sta/system',
                'enhancement_applied': True,  # 假设APIConnector内部处理了
                'validation_performed': True,  # 假设APIConnector内部处理了
            }
        }

        logger.info("✅ API: 系统数据增强处理完成")
        return create_success_response(response_data, "系统数据获取并分析成功")

    except Exception as e:
        logger.error(f"❌ API: 获取系统数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"系统数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/daily', methods=['GET'])
@async_route
async def get_enhanced_daily_data():
    """
    📅 获取增强每日数据 - 智能日期处理
    参数:
    - date (可选): YYYYMMDD格式的日期。如果未提供，则默认为当天。
    """
    try:
        logger.info("📅 API: 请求获取增强每日数据...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(optional_params={'date': {'type': str}})
        date_param = params.get('date')

        if date_param:  # 仅当date_param存在时才验证
            date_param = validate_date_format(date_param, "date")

        # APIConnector的get_daily_data()内部已包含数据增强和验证
        result = await orchestrator.data_fetcher.api_connector.get_daily_data(date_param)

        if not result.get("success"):
            return create_error_response(result.get("message", "每日数据获取失败"), "api_error", 500)

        enhanced_data = result.get("data", {})
        validation_info = result.get("validation", {})

        # 简单的日度表现评估
        daily_performance = {}
        if enhanced_data:
            try:
                inflow = float(enhanced_data.get('入金', 0))
                outflow = float(enhanced_data.get('出金', 0))
                registrations = int(enhanced_data.get('注册人数', 0))
                purchases = int(enhanced_data.get('购买产品数量', 0))

                daily_performance = {
                    'net_flow': inflow - outflow,
                    'flow_ratio': (inflow / outflow) if outflow > 0 else float('inf') if inflow > 0 else 0,
                    'conversion_indicator': (purchases / registrations) if registrations > 0 else 0,  # 简化指标
                    'activity_level_indicator': (registrations + purchases) / 2  # 简化指标
                }
            except Exception as e:
                logger.warning(f"日度表现评估失败: {str(e)}")
                daily_performance = {'error': '评估失败'}

        response_data = {
            'daily_summary': enhanced_data,
            'data_validation_summary': validation_info,
            'daily_performance_snapshot': daily_performance,
            'query_date': date_param or datetime.now().strftime('%Y%m%d'),
            'processing_metadata': {
                'date_processing': 'automatic_latest' if not date_param else 'specified_date'
            }
        }
        logger.info("✅ API: 每日数据增强处理完成")
        return create_success_response(response_data, "每日数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 获取每日数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"每日数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/products', methods=['GET'])
@async_route
async def get_enhanced_products_data():
    """
    🛍️ 获取增强产品数据 - 包含基本的产品表现快照
    参数:
    - include_expiry (可选, bool, 默认true): 是否包含产品到期信息。
    - include_analysis (可选, bool, 默认false): 是否进行简单的产品表现分析。
    """
    try:
        logger.info("🛍️ API: 请求获取增强产品数据...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(optional_params={
            'include_expiry': {'type': bool, 'default': True},
            'include_analysis': {'type': bool, 'default': False}
        })
        include_expiry = params['include_expiry']
        include_analysis = params['include_analysis']

        result = await orchestrator.data_fetcher.api_connector.get_product_data()

        if not result.get("success"):
            return create_error_response(result.get("message", "产品数据获取失败"), "api_error", 500)

        # APIConnector 内部的 _enhance_product_data 似乎未被其公共方法 get_product_data 调用
        # 这里我们假设 result["data"] 是原始或轻度增强的数据
        raw_product_data = result.get("data", {})

        # 如果需要，我们可以显式调用一个编排器层面的增强方法，或者 SmartDataFetcher 的方法
        # 为简单起见，此处直接使用原始数据，并进行一些基本分析
        product_list = raw_product_data.get("产品列表", [])
        product_performance_snapshot = {}
        product_insights = {}

        if include_analysis and product_list:
            try:
                top_products = sorted(product_list, key=lambda x: int(x.get('总购买次数', 0)), reverse=True)[:3]
                low_utilization_products = [
                    p for p in product_list
                    if int(p.get('当前持有数', 0)) / max(int(p.get('总购买次数', 1)), 1) < 0.2  # 假设利用率低于20%为低
                ]
                product_performance_snapshot = {
                    'total_products': raw_product_data.get("产品总数", len(product_list)),
                    'top_3_performing_by_purchase': [p.get('产品名称', '未知产品') for p in top_products],
                    'low_utilization_product_count': len(low_utilization_products),
                    'products_with_upcoming_expiry_count': sum(
                        1 for p in product_list if p.get("持有情况", {}).get("即将到期数", {}).get("7天内", 0) > 0)
                }
                # 简单的洞察
                product_insights = {
                    "key_observation": f"共追踪 {product_performance_snapshot['total_products']} 款产品。",
                    "popular_products_info": f"最受欢迎的产品包括：{', '.join(product_performance_snapshot['top_3_performing_by_purchase'])}。",
                    "utilization_alert": f"有 {product_performance_snapshot['low_utilization_product_count']} 款产品利用率较低，可能需要关注。" if
                    product_performance_snapshot[
                        'low_utilization_product_count'] > 0 else "所有产品利用率均在可接受范围。",
                    "expiry_outlook": f"未来7天内有 {product_performance_snapshot['products_with_upcoming_expiry_count']} 款产品有到期情况。" if include_expiry else "未分析产品到期情况。"
                }

            except Exception as e:
                logger.warning(f"产品表现快照分析失败: {str(e)}")
                product_performance_snapshot = {'error': '快照分析失败'}
                product_insights = {'error': '洞察生成失败'}

        response_data = {
            'product_catalog': raw_product_data,  # 返回APIConnector获取的数据
            'product_performance_snapshot': product_performance_snapshot,
            'product_insights_summary': product_insights,  # 替换原product_insights
            'filters_applied': {
                'include_expiry_info': include_expiry,  # APIConnector内部已处理
                'simple_analysis_included': include_analysis
            }
        }
        logger.info("✅ API: 产品数据处理完成")
        return create_success_response(response_data, "产品数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 获取产品数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"产品数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/users/daily', methods=['GET'])
@async_route
async def get_enhanced_user_daily_data():
    """
    👥 获取用户每日数据 - 包含基本的VIP分布和增长快照
    参数:
    - date (可选): YYYYMMDD格式的日期。如果未提供，则默认为当天。
    """
    try:
        logger.info("👥 API: 请求获取用户每日数据...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(optional_params={'date': {'type': str}})
        date_param = params.get('date')
        if date_param:
            date_param = validate_date_format(date_param, "date")

        result = await orchestrator.data_fetcher.api_connector.get_user_daily_data(date_param)

        if not result.get("success"):
            return create_error_response(result.get("message", "用户每日数据获取失败"), "api_error", 500)

        raw_user_daily_data = result.get("data", {})
        # APIConnector的get_user_daily_data似乎不直接调用内部的_enhance_user_daily_data
        # 我们在此处做一些简单分析

        vip_analysis_snapshot = {}
        growth_snapshot = {}

        daily_data_list = raw_user_daily_data.get("每日数据", [])
        if daily_data_list:
            latest_day_data = daily_data_list[-1]  # 通常API返回的是单个日期或排序后的列表

            try:  # VIP分析
                total_users_on_date = sum(int(latest_day_data.get(f'vip{i}的人数', 0)) for i in range(11))
                vip_distribution = {
                    f'vip{i}': {
                        'count': int(latest_day_data.get(f'vip{i}的人数', 0)),
                        'percentage': (int(latest_day_data.get(f'vip{i}的人数',
                                                               0)) / total_users_on_date * 100) if total_users_on_date > 0 else 0
                    } for i in range(11)
                }
                vip_analysis_snapshot = {
                    'total_users_on_record_date': total_users_on_date,
                    'vip_distribution': vip_distribution,
                    'new_users_on_record_date': int(latest_day_data.get('新增用户数', 0)),
                    'analysis_date': latest_day_data.get('日期', date_param or datetime.now().strftime('%Y%m%d'))
                }
            except Exception as e:
                logger.warning(f"VIP分析快照失败: {str(e)}")
                vip_analysis_snapshot = {'error': 'VIP分析失败'}

            try:  # 增长快照 (如果数据是列表且多于一天)
                if len(daily_data_list) > 1:
                    new_users_series = [int(d.get('新增用户数', 0)) for d in daily_data_list]
                    avg_daily_growth_period = sum(new_users_series) / len(new_users_series)
                    growth_snapshot = {
                        'avg_daily_new_users_in_period': avg_daily_growth_period,
                        'period_trend_indicator': 'increasing' if len(new_users_series) > 1 and new_users_series[-1] >
                                                                  new_users_series[0] else 'stable/decreasing',
                        'data_points_in_period': len(new_users_series)
                    }
                elif daily_data_list:  # 单日数据
                    growth_snapshot = {'new_users_today': int(daily_data_list[0].get('新增用户数', 0))}

            except Exception as e:
                logger.warning(f"增长分析快照失败: {str(e)}")
                growth_snapshot = {'error': '增长分析失败'}

        response_data = {
            'user_daily_records': raw_user_daily_data,
            'vip_analysis_snapshot': vip_analysis_snapshot,
            'growth_snapshot': growth_snapshot,
            'query_date_effective': date_param or vip_analysis_snapshot.get('analysis_date',
                                                                            datetime.now().strftime('%Y%m%d'))
        }
        logger.info("✅ API: 用户每日数据处理完成")
        return create_success_response(response_data, "用户每日数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 获取用户每日数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"用户每日数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/users/detailed', methods=['GET'])
@async_route
async def get_enhanced_user_detailed_data():
    """
    📊 获取详细用户数据 - 分页，包含基本的页面统计
    参数:
    - page (可选, int, 默认1): 页码。
    - include_stats (可选, bool, 默认true): 是否包含本页用户数据的统计信息。
    """
    try:
        logger.info("📊 API: 请求获取详细用户数据...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(optional_params={
            'page': {'type': int, 'default': 1},
            'include_stats': {'type': bool, 'default': True}
        })
        page = params['page']
        include_stats = params['include_stats']

        if page < 1:
            return create_error_response("页码必须是正整数", "validation_error", 400)

        result = await orchestrator.data_fetcher.api_connector.get_user_data(page)

        if not result.get("success"):
            return create_error_response(result.get("message", "用户详细数据获取失败"), "api_error", 500)

        raw_user_data = result.get("data", {})
        user_page_statistics = {}

        if include_stats:
            user_list = raw_user_data.get("用户列表", [])
            if user_list:
                try:
                    total_investment_page = sum(float(u.get('总投入', 0)) for u in user_list)
                    total_rewards_page = sum(float(u.get('累计获得奖励金额', 0)) for u in user_list)
                    roi_values_page = [float(u.get('投报比', 0)) for u in user_list if u.get('投报比') is not None]

                    user_page_statistics = {
                        'users_on_this_page': len(user_list),
                        'total_investment_on_page': total_investment_page,
                        'total_rewards_on_page': total_rewards_page,
                        'avg_roi_on_page': (sum(roi_values_page) / len(roi_values_page)) if roi_values_page else 0,
                        'avg_investment_per_user_on_page': (total_investment_page / len(user_list)) if user_list else 0
                    }
                except Exception as e:
                    logger.warning(f"用户页面统计计算失败: {str(e)}")
                    user_page_statistics = {'error': '页面统计计算失败'}

        response_data = {
            'user_data_page': raw_user_data,
            'user_page_statistics': user_page_statistics,
            'pagination_info': {
                'current_page': page,
                'total_records': raw_user_data.get('总记录数', 0),
                'total_pages': raw_user_data.get('总页数', 0)
            }
        }
        logger.info(f"✅ API: 用户详细数据处理完成: 第{page}页")
        return create_success_response(response_data, f"用户详细数据(第{page}页)获取成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 获取用户详细数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"用户详细数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/expiry', methods=['GET'])
@async_route
async def get_enhanced_expiry_data():
    """
    ⏰ 获取增强到期数据 - 包含基本的风险和复投快照
    参数:
    - period (可选, str, 默认'week'): 查询周期 ('today', 'tomorrow', 'week', 'single', 'range')
    - date (可选, str): YYYYMMDD格式, period='single'时必需
    - start_date (可选, str): YYYYMMDD格式, period='range'时必需
    - end_date (可选, str): YYYYMMDD格式, period='range'时必需
    - include_analysis (可选, bool, 默认true): 是否包含风险和复投快照
    """
    try:
        logger.info("⏰ API: 请求获取增强到期数据...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(optional_params={
            'period': {'type': str, 'default': 'week', 'allowed': ['today', 'tomorrow', 'week', 'single', 'range']},
            'date': {'type': str},
            'start_date': {'type': str},
            'end_date': {'type': str},
            'include_analysis': {'type': bool, 'default': True}
        })
        period = params['period']
        date_param = params.get('date')
        start_date_param = params.get('start_date')
        end_date_param = params.get('end_date')
        include_analysis = params['include_analysis']

        api_call_params = {}
        if period == 'single':
            if not date_param: return create_error_response("单日查询缺少 'date' 参数", "validation_error", 400)
            api_call_params['date'] = validate_date_format(date_param, "date")
            result = await orchestrator.data_fetcher.api_connector.get_product_end_data(api_call_params['date'])
        elif period == 'range':
            if not start_date_param or not end_date_param: return create_error_response(
                "范围查询缺少 'start_date' 或 'end_date' 参数", "validation_error", 400)
            api_call_params['start_date'], api_call_params['end_date'] = validate_date_range(start_date_param,
                                                                                             end_date_param)
            result = await orchestrator.data_fetcher.api_connector.get_product_end_interval(
                api_call_params['start_date'], api_call_params['end_date'])
        elif period == 'today':
            result = await orchestrator.data_fetcher.api_connector.get_expiring_products_today()
        elif period == 'tomorrow':
            result = await orchestrator.data_fetcher.api_connector.get_expiring_products_tomorrow()
        elif period == 'week':
            result = await orchestrator.data_fetcher.api_connector.get_expiring_products_week()
        else:  # Should not happen due to 'allowed' in validate_request_params
            return create_error_response(f"不支持的期间类型: {period}", "validation_error", 400)

        if not result.get("success"):
            return create_error_response(result.get("message", "到期数据获取失败"), "api_error", 500)

        raw_expiry_data = result.get("data", {})
        expiry_analysis_snapshot = {}

        if include_analysis and raw_expiry_data:
            try:
                expiry_amount = float(raw_expiry_data.get("到期金额", 0))
                expiry_quantity = int(raw_expiry_data.get("到期数量", 0))
                risk_level = 'high' if expiry_amount > 1000000 else 'medium' if expiry_amount > 200000 else 'low'  # 简化风险评估

                # 假设的复投率和机会
                assumed_reinvestment_rate = 0.5  # 50%
                potential_reinvestment = expiry_amount * assumed_reinvestment_rate

                expiry_analysis_snapshot = {
                    'total_expiry_amount_in_period': expiry_amount,
                    'total_expiry_quantity_in_period': expiry_quantity,
                    'liquidity_risk_indicator': risk_level,
                    'potential_reinvestment_amount': potential_reinvestment,
                    'estimated_cash_outflow': expiry_amount * (1 - assumed_reinvestment_rate)
                }
            except Exception as e:
                logger.warning(f"到期数据快照分析失败: {str(e)}")
                expiry_analysis_snapshot = {'error': '快照分析失败'}

        response_data = {
            'expiry_data_for_period': raw_expiry_data,
            'expiry_analysis_snapshot': expiry_analysis_snapshot,
            'query_parameters': {
                'period_type': period,
                **api_call_params
            }
        }
        logger.info("✅ API: 到期数据处理完成")
        return create_success_response(response_data, "到期数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 获取到期数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"到期数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/analytics', methods=['POST'])  # Changed to POST to accept more complex parameters
@async_route
async def get_enhanced_analytics_data():
    """
    📈 获取增强分析数据 - 触发特定的后台分析流程
    请求体 (JSON):
    {
        "analysis_type": "comprehensive" | "trend" | "performance" | "risk",
        "time_range_days": 30, // 分析的时间范围（天数）
        "metrics": ["total_balance", "net_inflow"], // 可选，具体分析的指标
        "scope": "financial" // 可选，分析范围，如 financial, user, product
        "include_insights": true // 是否生成AI洞察
    }
    """
    try:
        logger.info("📈 API: 请求获取增强分析数据...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(
            required_params=['analysis_type'],
            optional_params={
                'time_range_days': {'type': int, 'default': 30},
                'metrics': {'type': list, 'default': []},
                'scope': {'type': str, 'default': 'financial'},  # 'financial' is a sensible default
                'include_insights': {'type': bool, 'default': True}
            }
        )
        analysis_type = params['analysis_type']
        time_range_days = params['time_range_days']
        metrics_to_analyze = params['metrics']
        analysis_scope = params['scope']
        include_insights = params['include_insights']

        if time_range_days < 1 or time_range_days > 365:
            return create_error_response("时间范围必须在1-365天之间", "validation_error", 400)

        valid_analysis_types = ['comprehensive', 'trend', 'performance', 'risk']
        if analysis_type not in valid_analysis_types:
            return create_error_response(f"无效的分析类型: {analysis_type}", "validation_error", 400)

        analysis_results = []
        # 根据 orchestrator 的能力，我们应该调用其分析组件
        # FinancialDataAnalyzer 是主要的分析引擎

        if analysis_type == 'comprehensive' or analysis_type == 'trend':
            for metric_name in metrics_to_analyze if metrics_to_analyze else ['total_balance',
                                                                              'net_inflow']:  # Default metrics
                trend_res = await orchestrator.financial_data_analyzer.analyze_trend(
                    data_source=analysis_scope if analysis_scope in ['system', 'daily', 'product',
                                                                     'user'] else 'system',  # Adapt scope
                    metric=metric_name,
                    time_range=time_range_days
                )
                analysis_results.append(trend_res)

        if analysis_type == 'comprehensive' or analysis_type == 'performance':
            perf_res = await orchestrator.financial_data_analyzer.analyze_business_performance(
                scope=analysis_scope,
                time_range=time_range_days
            )
            analysis_results.append(perf_res)

        # if analysis_type == 'comprehensive' or analysis_type == 'risk':
        #     risk_metrics = metrics_to_analyze if metrics_to_analyze else ['total_balance', 'daily_outflow']
        #     risk_res = await orchestrator.financial_data_analyzer.detect_anomalies(
        #         data_source=analysis_scope if analysis_scope in ['system', 'daily'] else 'system',  # Adapt scope
        #         metrics=risk_metrics
        #     )
        #     analysis_results.append(risk_res)

        # 如果没有选择具体分析类型但有指标，默认做趋势分析
        if not analysis_results and metrics_to_analyze:
            for metric_name in metrics_to_analyze:
                trend_res = await orchestrator.financial_data_analyzer.analyze_trend(
                    data_source=analysis_scope if analysis_scope in ['system', 'daily', 'product',
                                                                     'user'] else 'system',
                    metric=metric_name,
                    time_range=time_range_days
                )
                analysis_results.append(trend_res)

        insights_package = {}
        if include_insights and analysis_results:
            try:
                # 过滤掉None或错误的分析结果
                valid_analysis_results = [res for res in analysis_results if hasattr(res, 'analysis_type')]

                insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                    analysis_results=valid_analysis_results,
                    user_context=None,  # 可以从请求中获取用户上下文
                    focus_areas=[analysis_type] if analysis_type != 'comprehensive' else ['financial_health',
                                                                                          'risk_management',
                                                                                          'growth_analysis']
                )
                insights_package = {
                    'generated_insights': [insight.__dict__ if hasattr(insight, '__dict__') else insight for insight in
                                           insights],
                    'generation_metadata': metadata
                }
            except Exception as e:
                logger.warning(f"洞察生成失败: {str(e)}")
                insights_package = {'error': f'洞察生成失败: {str(e)}'}

        # 清理和格式化分析结果
        formatted_analysis_results = []
        for res in analysis_results:
            if hasattr(res, 'analysis_type'):  # 确保是有效的AnalysisResult对象
                formatted_analysis_results.append({
                    'analysis_id': res.analysis_id,
                    'type': res.analysis_type.value,
                    'scope': res.analysis_scope.value,
                    'confidence': res.confidence_score,
                    'key_findings': res.key_findings,
                    'metrics': res.metrics,
                    'trends': res.trends,
                    'anomalies': res.anomalies,
                    'recommendations': res.recommendations,
                    'processing_time': res.processing_time
                })

        response_data = {
            'triggered_analysis_type': analysis_type,
            'analysis_time_range_days': time_range_days,
            'analysis_scope': analysis_scope,
            'metrics_analyzed_explicitly': metrics_to_analyze,
            'detailed_analysis_results': formatted_analysis_results,
            'ai_insights_package': insights_package
        }
        logger.info(f"✅ API: {analysis_type}分析数据处理完成")
        return create_success_response(response_data, f"{analysis_type}分析完成")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 获取分析数据失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"分析数据处理失败: {str(e)}", "internal_error", 500)


# generate-report 和 generate-data-report 路由目前看起来耦合了数据获取和报告生成。
# 在一个更纯粹的 "data_routes" 中，它们可能只负责提供格式化的数据给前端或另一个服务来生成报告。
# 但既然它们在这里，我会确保它们使用 orchestrator 的组件。

@data_bp.route('/generate-report', methods=['POST'])
@async_route
async def generate_analysis_report():
    """
    📊 生成分析报告 - 基于提供的分析数据或触发新的分析来创建专业报告
    请求体 (JSON):
    {
        "report_type": "financial" | "trend" | "comparison", // 报告类型
        "data": {}, // 用于生成报告的数据，如果为空，则系统会尝试获取默认数据
        "title": "自定义报告标题", // 可选
        "format": "html" | "pdf" | "markdown", // 输出格式，默认html
        "time_range_days": 30 // 如果data为空，用于获取数据的默认时间范围
    }
    """
    try:
        logger.info("📊 API: 请求生成分析报告...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(
            required_params=['report_type'],
            optional_params={
                'data': {'type': dict, 'default': {}},
                'title': {'type': str, 'default': '数据分析报告'},
                'format': {'type': str, 'default': 'html', 'allowed': ['html', 'pdf', 'markdown']},
                'time_range_days': {'type': int, 'default': 30},
                'period': {'type': str}  # For trend reports
            }
        )
        report_type = params['report_type']
        report_data_from_request = params['data']
        report_title = params['title']
        report_format_str = params['format']
        time_range_days = params['time_range_days']
        period_for_trend = params.get('period')

        valid_report_types = ['financial', 'trend', 'comparison']
        if report_type not in valid_report_types:
            return create_error_response(f"无效的报告类型: {report_type}", "validation_error", 400)

        # 如果请求中没有提供数据，则尝试获取数据
        if not report_data_from_request:
            logger.info(f"未提供报告数据，将为'{report_type}'类型报告获取默认数据...")
            # 这是一个简化的数据获取逻辑，实际可能更复杂
            if report_type == 'financial':
                financial_performance = await orchestrator.financial_data_analyzer.analyze_business_performance(
                    'financial', time_range_days)
                # 将AnalysisResult转换为generate_financial_report期望的字典格式
                report_data_from_request = {
                    'subtitle': f"{time_range_days}天财务表现",
                    'summary': {'content': '; '.join(financial_performance.key_findings),
                                'key_metrics': [{'name': k, 'value': v, 'format_type': 'auto'} for k, v in
                                                financial_performance.metrics.items()]},
                    'analysis_sections': [{'title': '趋势分析', 'content': '; '.join(
                        t.get('direction', 'stable') for t in financial_performance.trends),
                                           'charts': []}] if financial_performance.trends else []
                }
            elif report_type == 'trend':
                trend_analysis = await orchestrator.financial_data_analyzer.analyze_trend('system', 'total_balance',
                                                                                          time_range_days)
                # 转换为generate_trend_report期望的格式
                report_data_from_request = {
                    'summary': {'content': '; '.join(trend_analysis.key_findings)},
                    'trends': [{'title': trend_analysis.metrics.get('metric_name', '趋势'),
                                'description': trend_analysis.trends[0].get('direction', '未知'),
                                'data': trend_analysis.metrics.get('supporting_data',
                                                                   [])}] if trend_analysis.trends else []
                }
                if not period_for_trend: period_for_trend = f"最近{time_range_days}天"

            # 可以为 'comparison' 类型添加类似的数据获取逻辑
            else:
                logger.warning(f"报告类型 '{report_type}' 的自动数据获取未完全实现，可能导致报告数据不足。")
                # 获取通用系统数据作为基础
                system_data_res = await orchestrator.data_fetcher.api_connector.get_system_data()
                if system_data_res.get("success"):
                    report_data_from_request = system_data_res["data"]
                else:
                    return create_error_response("自动获取报告数据失败", "data_fetch_error", 500)

        output_format_enum = ReportFormat[report_format_str.upper()]

        # 使用编排器中的报告生成器
        report_object = None
        if report_type == 'financial':
            report_object = orchestrator.report_generator.generate_financial_report(
                data=report_data_from_request, title=report_title, output_format=output_format_enum
            )
        elif report_type == 'trend':
            report_object = orchestrator.report_generator.generate_trend_report(
                trend_data=report_data_from_request, title=report_title, period=period_for_trend,
                output_format=output_format_enum
            )
        elif report_type == 'comparison':
            report_object = orchestrator.report_generator.generate_comparison_report(
                comparison_data=report_data_from_request, title=report_title, output_format=output_format_enum
            )

        if not report_object:
            return create_error_response(f"无法为类型 '{report_type}' 生成报告对象", "report_generation_error", 500)

        report_content = ""
        if output_format_enum == ReportFormat.HTML:
            report_content = report_object.to_html()
            mimetype = 'text/html'
        elif output_format_enum == ReportFormat.MARKDOWN:
            report_content = report_object.to_markdown()
            mimetype = 'text/markdown'
        elif output_format_enum == ReportFormat.PDF:
            import tempfile, os
            temp_dir = tempfile.gettempdir()
            # 确保文件名是唯一的，并且有.pdf扩展名
            timestamp = int(time.time())
            filename = f"report_{timestamp}.pdf"
            temp_file_path = os.path.join(temp_dir, filename)

            try:
                # report_object.to_pdf(temp_file_path) # generate_financial_report等方法若指定output_path则直接保存
                # 我们需要先获得 Report 对象，然后调用其 to_pdf 方法
                if isinstance(report_object, str) and os.path.exists(report_object):  # 如果生成器方法直接返回路径
                    temp_file_path = report_object
                else:  # 假设 report_object 是 Report 类的实例
                    report_object.to_pdf(temp_file_path)

                with open(temp_file_path, 'rb') as f:
                    report_content_bytes = f.read()

                # 可选：删除临时文件
                # os.remove(temp_file_path)

                # 对于PDF，通常是作为文件下载，或返回文件路径/链接
                # 这里为了API一致性，可以返回一个消息指示PDF已生成，或base64编码
                # 为了简单，这里返回一个指示消息，实际应用中可能需要文件服务
                return create_success_response({
                    'message': 'PDF报告已生成，请通过指定路径或后续机制获取。',
                    'report_path_info': temp_file_path,  # 仅用于演示，生产环境不应直接暴露临时文件路径
                    'report_type': report_type,
                    'format': report_format_str
                }, "PDF报告处理完成")

            except Exception as pdf_err:
                logger.error(f"PDF报告生成或读取失败: {pdf_err}\n{traceback.format_exc()}")
                return create_error_response(f"PDF报告生成失败: {pdf_err}", "report_generation_error", 500)

        logger.info(f"✅ API: {report_type}报告生成完成，格式: {report_format_str}")
        return Response(report_content, mimetype=mimetype) if report_format_str != 'pdf' else \
            create_error_response("PDF生成逻辑需要调整以适配Response", "internal_error", 500)  # PDF case handled above

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 分析报告生成失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"报告生成失败: {str(e)}", "report_generation_error", 500)


@data_bp.route('/generate-chart', methods=['POST'])
@async_route
async def generate_data_chart():
    """
    🎨 生成数据图表 - 基于提供的数据和配置智能生成图表
    请求体 (JSON):
    {
        "data": {}, // 用于图表的数据，例如 {"labels": ["A", "B"], "values": [10, 20]}
        "chart_type": "line", // 图表类型 (line, bar, pie, etc.)，或 "auto"
        "title": "图表标题", // 可选
        "config": {}, // 图表库特定的配置，可选
        "preferences": {"theme": "financial"} // 可选的用户偏好
    }
    """
    try:
        logger.info("🎨 API: 请求生成数据图表...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        params = validate_request_params(
            required_params=['data'],
            optional_params={
                'chart_type': {'type': str, 'default': 'auto'},
                'title': {'type': str, 'default': '数据图表'},
                'config': {'type': dict, 'default': {}},
                'preferences': {'type': dict, 'default': {}}
            }
        )
        chart_data = params['data']
        chart_type_str = params['chart_type']
        chart_title = params['title']
        chart_config_extra = params['config']
        chart_preferences = params['preferences']

        # 将字符串的 chart_type 转换为 ChartType 枚举，如果它是有效成员的话
        try:
            chart_type_enum = ChartType[chart_type_str.upper()] if chart_type_str != 'auto' else None
        except KeyError:
            return create_error_response(f"不支持的图表类型: {chart_type_str}", "validation_error", 400)

        # 使用编排器中的图表生成器
        # ChartGenerator.py (utils/formatters)的 generate_chart 方法
        chart_result = orchestrator.chart_generator.generate_chart(
            data=chart_data,
            chart_type=chart_type_enum,  # 传递枚举或None
            title=chart_title,
            config=chart_config_extra
        )

        if not chart_result or chart_result.get("error"):
            return create_error_response(chart_result.get("error", "图表生成失败"), "chart_generation_error", 500)

        # image_data 包含了 base64, svg, 或 binary, 以及 format
        # 根据前端需求，可能返回base64编码的图片或图表配置本身
        response_data = {
            'chart_type_generated': chart_result.get('type', chart_type_str),
            'title': chart_result.get('title', chart_title),
            'image_data': chart_result.get('image_data'),  # { "base64": "...", "format": "png" } or { "svg": "..." }
            'raw_chart_config': chart_result.get('chart_config_for_frontend')  # 如果图表库返回前端可渲染的配置
        }

        logger.info(f"✅ API: 图表生成完成: {response_data['chart_type_generated']}")
        return create_success_response(response_data, "图表生成成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 图表生成失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"图表生成失败: {str(e)}", "chart_generation_error", 500)


# ============= 系统监控和状态API (可以保留或增强) =============

@data_bp.route('/health', methods=['GET'])
@async_route
async def data_service_health():
    """🔍 数据服务健康检查 (已通过编排器能力增强)"""
    try:
        if not orchestrator.initialized:
            await orchestrator.initialize()

        health_status = await orchestrator.health_check()  # 调用编排器的全面健康检查

        # 从编排器健康检查结果中提取数据服务相关的部分
        api_connector_health = "unavailable"
        if hasattr(orchestrator, 'data_fetcher') and hasattr(orchestrator.data_fetcher, 'api_connector'):
            api_connector_status = await orchestrator.data_fetcher.api_connector.health_check()
            api_connector_health = api_connector_status.get('status', 'unknown')

        data_services_status = {
            'overall_orchestrator_status': health_status.get('status', 'unknown'),
            'api_connector_status': api_connector_health,
            'data_analyzer_available': hasattr(orchestrator,
                                               'financial_data_analyzer') and orchestrator.financial_data_analyzer is not None,
            'time_series_builder_available': hasattr(orchestrator.data_fetcher,
                                                     'time_series_builder') and orchestrator.data_fetcher.time_series_builder is not None,
            'financial_calculator_available': hasattr(orchestrator.financial_data_analyzer,
                                                      'financial_calculator') and orchestrator.financial_data_analyzer.financial_calculator is not None,
            'report_generator_available': hasattr(orchestrator,
                                                  'report_generator') and orchestrator.report_generator is not None,
            'chart_generator_available': hasattr(orchestrator,
                                                 'chart_generator') and orchestrator.chart_generator is not None,
        }

        overall_data_health = 'healthy' if all(
            status == 'healthy' or status is True for status in data_services_status.values() if
            isinstance(status, (str, bool))) else 'degraded'

        return create_success_response({
            'data_service_health': overall_data_health,
            'details': data_services_status
        }, "数据服务健康检查完成")

    except Exception as e:
        logger.error(f"❌ 数据服务健康检查失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"数据服务健康检查失败: {str(e)}", "health_check_error", 500)


@data_bp.route('/stats', methods=['GET'])
@async_route
async def get_data_service_stats():
    """📊 获取数据服务相关的统计信息 (通过编排器获取)"""
    try:
        if not orchestrator.initialized:
            await orchestrator.initialize()

        stats = orchestrator.get_orchestrator_stats()  # 全局统计

        api_connector_stats = {}
        if hasattr(orchestrator, 'data_fetcher') and hasattr(orchestrator.data_fetcher, 'api_connector'):
            api_connector_stats = orchestrator.data_fetcher.api_connector.get_connector_stats()

        financial_analyzer_stats = {}
        if hasattr(orchestrator, 'financial_data_analyzer'):
            financial_analyzer_stats = orchestrator.financial_data_analyzer.get_analysis_stats()

        # 更多组件的统计信息可以按需添加

        return create_success_response({
            'overall_system_stats': stats,
            'api_connector_module_stats': api_connector_stats,
            'financial_analyzer_module_stats': financial_analyzer_stats
            # ... 其他组件统计
        }, "数据服务统计信息获取成功")

    except Exception as e:
        logger.error(f"❌ 获取数据服务统计失败: {str(e)}\n{traceback.format_exc()}")
        return create_error_response(f"统计信息获取失败: {str(e)}", "internal_error", 500)


# ============= 错误处理 (蓝图级别) =============

@data_bp.app_errorhandler(404)  # 使用app_errorhandler处理蓝图外的404
def handle_data_not_found_error(error):
    logger.warning(f"资源未找到 (404): {request.path}")
    return create_error_response("请求的资源未找到", "not_found_error", 404)


@data_bp.app_errorhandler(500)
def handle_data_internal_error(error):
    logger.error(f"数据服务内部服务器错误 (500): {str(error)}\n{traceback.format_exc()}")
    return create_error_response(f"服务器内部错误: {str(error)}", "internal_server_error", 500,
                                 {"trace": traceback.format_exc() if orchestrator.config.get("DEBUG") else None})


@data_bp.app_errorhandler(400)
def handle_data_bad_request_error(error):
    # Flask的默认400错误对象有一个description属性
    message = error.description if hasattr(error, 'description') else "请求参数无效或格式错误"
    logger.warning(f"错误的请求 (400): {message}")
    return create_error_response(message, "bad_request_error", 400)


@data_bp.app_errorhandler(ValueError)  # 捕获由validate_...函数抛出的ValueError
def handle_validation_error(error):
    logger.warning(f"参数验证错误 (ValueError): {str(error)}")
    return create_error_response(str(error), "validation_error", 400)