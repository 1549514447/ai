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

logger = logging.getLogger(__name__)

# 创建数据API蓝图
data_bp = Blueprint('data_api', __name__, url_prefix='/api/data')

# 🎯 使用单一编排器实例 - 避免重复初始化
orchestrator = get_orchestrator()


# ============= 工具函数 =============

def async_route(f):
    """异步路由装饰器"""

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()

    wrapper.__name__ = f.__name__
    return wrapper


def validate_request_params(required_params: List[str] = None) -> Dict[str, Any]:
    """验证请求参数"""
    try:
        # 获取查询参数
        params = request.args.to_dict()

        # 获取JSON数据（如果有）
        if request.content_type and 'application/json' in request.content_type:
            json_data = request.get_json() or {}
            params.update(json_data)

        if required_params:
            missing_params = [param for param in required_params if param not in params]
            if missing_params:
                raise ValueError(f"缺少必填参数: {missing_params}")

        return params
    except Exception as e:
        raise ValueError(f"参数验证失败: {str(e)}")


def validate_date_format(date_str: str, param_name: str = "date") -> str:
    """验证日期格式 YYYYMMDD"""
    if not date_str:
        return None
    
    if not re.match(r'^\d{8}$', date_str):
        raise ValueError(f"{param_name}格式错误，应为YYYYMMDD格式")
    
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return date_str
    except ValueError:
        raise ValueError(f"{param_name}日期无效: {date_str}")


def validate_date_range(start_date: str, end_date: str) -> tuple:
    """验证日期范围"""
    start_date = validate_date_format(start_date, "start_date")
    end_date = validate_date_format(end_date, "end_date")
    
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    if start_dt > end_dt:
        raise ValueError("开始日期不能晚于结束日期")
    
    # 检查日期范围是否合理（不超过1年）
    if (end_dt - start_dt).days > 365:
        raise ValueError("日期范围不能超过1年")
    
    return start_date, end_date


def create_success_response(data: Dict[str, Any], message: str = "操作成功") -> Dict[str, Any]:
    """创建成功响应"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'processed_by': 'intelligent_orchestrator'
    })


def create_error_response(message: str, error_type: str = "processing_error", status_code: int = 500) -> tuple:
    """创建错误响应"""
    return jsonify({
        'success': False,
        'error_type': error_type,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'system_status': 'error'
    }), status_code


# ============= 核心数据API =============

@data_bp.route('/system', methods=['GET'])
@async_route
async def get_enhanced_system_data():
    """
    📊 获取增强系统数据 - 使用编排器完整数据链
    """
    try:
        logger.info("🔍 获取增强系统数据...")

        # 检查编排器状态
        if not orchestrator.initialized:
            return create_error_response("智能编排器未初始化", "system_unavailable", 503)

        # 使用编排器的API连接器获取原始数据
        result = await orchestrator.api_connector.get_system_data()

        if not result.get("success"):
            return create_error_response(
                result.get("message", "系统数据获取失败"),
                "api_error",
                500
            )

        # 使用编排器的数据增强器增强数据
        enhanced_data = orchestrator.data_enhancer.enhance_system_data(result["data"])

        # 使用编排器的数据验证器验证数据
        validation_result = orchestrator.data_analyzer._calculate_statistical_metrics([
            ('today', float(result["data"].get('总余额', 0)))
        ])

        # 财务健康评估
        financial_health = {}
        try:
            total_inflow = float(result["data"].get('总入金', 0))
            total_outflow = float(result["data"].get('总出金', 0))
            if total_inflow > 0:
                outflow_ratio = total_outflow / total_inflow
                financial_health = {
                    'outflow_ratio': outflow_ratio,
                    'health_status': 'healthy' if outflow_ratio < 0.7 else 'concerning' if outflow_ratio < 0.9 else 'risky',
                    'net_flow': total_inflow - total_outflow
                }
        except Exception as e:
            logger.warning(f"财务健康评估失败: {str(e)}")

        response_data = {
            'enhanced_data': enhanced_data,
            'validation': validation_result,
            'financial_health': financial_health,
            'processing_metadata': {
                'enhancement_applied': True,
                'validation_performed': True,
                'data_source': 'system_api',
                'processing_components': [
                    'api_connector',
                    'data_enhancer',
                    'data_validator'
                ]
            }
        }

        logger.info("✅ 系统数据增强处理完成")
        return create_success_response(response_data, "系统数据获取并增强成功")

    except Exception as e:
        logger.error(f"❌ 获取系统数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(f"系统数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/daily', methods=['GET'])
@async_route
async def get_enhanced_daily_data():
    """
    📅 获取增强每日数据 - 智能日期处理
    """
    try:
        logger.info("📅 获取增强每日数据...")

        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)

        # 验证参数
        params = validate_request_params()
        date = params.get('date')  # 可选日期参数
        
        # 验证日期格式
        if date:
            date = validate_date_format(date, "date")

        # 使用编排器获取每日数据
        result = await orchestrator.api_connector.get_daily_data(date)

        if not result.get("success"):
            return create_error_response(
                result.get("message", "每日数据获取失败"),
                "api_error",
                500
            )

        # 使用编排器增强每日数据
        enhanced_data = orchestrator.data_enhancer.enhance_daily_data(result["data"], date)

        # 智能趋势分析
        trend_info = {}
        try:
            trend_analysis = await orchestrator.data_analyzer.analyze_trend(
                'daily', 'net_inflow', 7
            )
            trend_info = {
                'direction': trend_analysis.trends[0]['direction'] if trend_analysis.trends else 'stable',
                'confidence': trend_analysis.confidence_score,
                'growth_rate': trend_analysis.metrics.get('growth_rate', 0)
            }
        except Exception as e:
            logger.warning(f"趋势分析失败: {str(e)}")
            trend_info = {'direction': 'unknown', 'confidence': 0.0}

        # 日度表现评估
        daily_performance = {}
        if result["data"]:
            try:
                inflow = float(result["data"].get('入金', 0))
                outflow = float(result["data"].get('出金', 0))
                registrations = int(result["data"].get('注册人数', 0))
                purchases = int(result["data"].get('购买产品数量', 0))
                
                daily_performance = {
                    'net_flow': inflow - outflow,
                    'flow_ratio': inflow / outflow if outflow > 0 else float('inf'),
                    'conversion_rate': purchases / registrations if registrations > 0 else 0,
                    'activity_score': (registrations + purchases) / 2
                }
            except Exception as e:
                logger.warning(f"日度表现评估失败: {str(e)}")

        response_data = {
            'enhanced_data': enhanced_data,
            'trend_analysis': trend_info,
            'daily_performance': daily_performance,
            'query_date': date or datetime.now().strftime('%Y%m%d'),
            'processing_metadata': {
                'intelligent_enhancement': True,
                'trend_analysis_included': True,
                'date_processing': 'automatic' if not date else 'specified'
            }
        }

        logger.info("✅ 每日数据增强处理完成")
        return create_success_response(response_data, "每日数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 获取每日数据失败: {str(e)}")
        return create_error_response(f"每日数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/products', methods=['GET'])
@async_route
async def get_enhanced_products_data():
    """
    🛍️ 获取增强产品数据 - 智能产品分析
    """
    try:
        logger.info("🛍️ 获取增强产品数据...")

        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)

        # 验证参数
        params = validate_request_params()
        status = params.get('status', 'active')
        include_expiry = params.get('include_expiry', 'true').lower() == 'true'
        include_analysis = params.get('include_analysis', 'false').lower() == 'true'

        # 使用编排器获取产品数据
        result = await orchestrator.api_connector.get_product_data()

        if not result.get("success"):
            return create_error_response(
                result.get("message", "产品数据获取失败"),
                "api_error",
                500
            )

        # 使用编排器增强产品数据
        enhanced_data = orchestrator.data_enhancer.enhance_product_data(
            result["data"],
            include_expiry=include_expiry
        )

        # 产品表现分析
        product_performance = {}
        try:
            product_list = result["data"].get("产品列表", [])
            if product_list:
                # 计算产品表现指标
                top_products = sorted(product_list, key=lambda x: x.get('总购买次数', 0), reverse=True)[:5]
                low_utilization = [p for p in product_list 
                                 if p.get('当前持有数', 0) / max(p.get('总购买次数', 1), 1) < 0.3]
                
                product_performance = {
                    'top_performing_products': [p.get('产品名称', '') for p in top_products],
                    'low_utilization_count': len(low_utilization),
                    'total_products': len(product_list),
                    'avg_utilization_rate': sum(
                        p.get('当前持有数', 0) / max(p.get('总购买次数', 1), 1) 
                        for p in product_list
                    ) / len(product_list) if product_list else 0
                }
        except Exception as e:
            logger.warning(f"产品表现分析失败: {str(e)}")

        # 可选的智能产品分析
        product_insights = {}
        if include_analysis:
            try:
                # 使用编排器生成产品洞察
                insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                    analysis_results=[],
                    user_context=None,
                    focus_areas=['product_analysis']
                )
                product_insights = {
                    'insights': insights,
                    'analysis_metadata': metadata
                }
            except Exception as e:
                logger.warning(f"产品洞察生成失败: {str(e)}")
                product_insights = {'error': '产品洞察分析暂不可用'}

        response_data = {
            'enhanced_data': enhanced_data,
            'product_performance': product_performance,
            'product_insights': product_insights,
            'filters': {
                'status': status,
                'include_expiry': include_expiry,
                'include_analysis': include_analysis
            },
            'processing_metadata': {
                'enhancement_level': 'comprehensive',
                'insights_included': include_analysis,
                'expiry_analysis': include_expiry
            }
        }

        logger.info("✅ 产品数据增强处理完成")
        return create_success_response(response_data, "产品数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 获取产品数据失败: {str(e)}")
        return create_error_response(f"产品数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/users/daily', methods=['GET'])
@async_route
async def get_enhanced_user_daily_data():
    """
    👥 获取用户每日数据 - 用户行为分析
    """
    try:
        logger.info("👥 获取用户每日数据...")
        
        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)
        
        # 验证参数
        params = validate_request_params()
        date = params.get('date')  # 可选日期参数
        
        # 验证日期格式
        if date:
            date = validate_date_format(date, "date")
        
        # 使用编排器获取用户每日数据
        result = await orchestrator.api_connector.get_user_daily_data(date)
        
        if not result.get("success"):
            return create_error_response(
                result.get("message", "用户每日数据获取失败"),
                "api_error", 500
            )
        
        # 增强用户数据分析
        enhanced_data = orchestrator.data_enhancer.enhance_user_daily_data(
            result["data"], date
        )
        
        # VIP分布分析
        vip_analysis = {}
        try:
            daily_data = result["data"].get("每日数据", [])
            if daily_data:
                latest_data = daily_data[-1]  # 最新日期的数据
                
                # 计算VIP分布
                total_users = sum(latest_data.get(f'vip{i}的人数', 0) for i in range(11))
                vip_distribution = {}
                for i in range(11):
                    vip_count = latest_data.get(f'vip{i}的人数', 0)
                    vip_distribution[f'vip{i}'] = {
                        'count': vip_count,
                        'percentage': (vip_count / total_users * 100) if total_users > 0 else 0
                    }
                
                vip_analysis = {
                    'total_users': total_users,
                    'vip_distribution': vip_distribution,
                    'high_value_users': sum(latest_data.get(f'vip{i}的人数', 0) for i in range(3, 11)),
                    'analysis_date': latest_data.get('日期', '')
                }
        except Exception as e:
            logger.warning(f"VIP分析失败: {str(e)}")
        
        # 用户增长趋势
        growth_analysis = {}
        try:
            daily_data = result["data"].get("每日数据", [])
            if len(daily_data) > 1:
                # 计算新增用户趋势
                new_users = [d.get('新增用户数', 0) for d in daily_data[-7:]]  # 最近7天
                avg_daily_growth = sum(new_users) / len(new_users) if new_users else 0
                
                growth_analysis = {
                    'avg_daily_new_users': avg_daily_growth,
                    'recent_trend': 'increasing' if len(new_users) > 1 and new_users[-1] > new_users[0] else 'stable',
                    'data_points': len(new_users)
                }
        except Exception as e:
            logger.warning(f"增长分析失败: {str(e)}")
        
        response_data = {
            'enhanced_data': enhanced_data,
            'vip_analysis': vip_analysis,
            'growth_analysis': growth_analysis,
            'query_date': date,
            'processing_metadata': {
                'user_behavior_analysis': True,
                'vip_distribution_included': True,
                'growth_trend_analyzed': True
            }
        }
        
        logger.info("✅ 用户每日数据处理完成")
        return create_success_response(response_data, "用户每日数据获取成功")
        
    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 获取用户每日数据失败: {str(e)}")
        return create_error_response(f"用户数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/users/detailed', methods=['GET'])
@async_route
async def get_enhanced_user_detailed_data():
    """
    📊 获取详细用户数据 - 分页用户详情
    """
    try:
        logger.info("📊 获取详细用户数据...")
        
        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)
        
        # 验证参数
        params = validate_request_params()
        page = int(params.get('page', 1))
        include_analysis = params.get('include_analysis', 'false').lower() == 'true'
        
        # 参数验证
        if page < 1:
            return create_error_response("页码必须大于0", "validation_error", 400)
        
        # 使用编排器获取用户详细数据
        result = await orchestrator.api_connector.get_user_data(page)
        
        if not result.get("success"):
            return create_error_response(
                result.get("message", "用户详细数据获取失败"),
                "api_error", 500
            )
        
        # 增强用户数据
        enhanced_data = orchestrator.data_enhancer.enhance_user_detailed_data(
            result["data"], include_analysis
        )
        
        # 用户统计分析
        user_statistics = {}
        try:
            user_list = result["data"].get("用户列表", [])
            if user_list:
                # 计算用户统计
                total_investment = sum(float(u.get('总投入', 0)) for u in user_list)
                total_rewards = sum(float(u.get('累计获得奖励金额', 0)) for u in user_list)
                roi_values = [float(u.get('投报比', 0)) for u in user_list if float(u.get('投报比', 0)) > 0]
                
                user_statistics = {
                    'page_user_count': len(user_list),
                    'total_investment_this_page': total_investment,
                    'total_rewards_this_page': total_rewards,
                    'avg_roi': sum(roi_values) / len(roi_values) if roi_values else 0,
                    'high_roi_users': len([r for r in roi_values if r > 0.1]),  # ROI > 10%
                    'avg_investment_per_user': total_investment / len(user_list) if user_list else 0
                }
        except Exception as e:
            logger.warning(f"用户统计分析失败: {str(e)}")
        
        # 可选的用户群体分析
        user_insights = {}
        if include_analysis:
            try:
                insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                    analysis_results=[],
                    user_context=None,
                    focus_areas=['user_behavior']
                )
                user_insights = {
                    'insights': insights,
                    'metadata': metadata
                }
            except Exception as e:
                logger.warning(f"用户洞察生成失败: {str(e)}")
        
        response_data = {
            'enhanced_data': enhanced_data,
            'user_statistics': user_statistics,
            'user_insights': user_insights,
            'pagination_info': {
                'current_page': page,
                'total_records': enhanced_data.get('总记录数', 0),
                'total_pages': enhanced_data.get('总页数', 0),
                'records_per_page': 1000
            },
            'processing_metadata': {
                'detailed_analysis': include_analysis,
                'enhancement_applied': True,
                'statistics_calculated': True
            }
        }
        
        logger.info(f"✅ 用户详细数据处理完成: 第{page}页")
        return create_success_response(response_data, f"用户详细数据获取成功(第{page}页)")
        
    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 获取用户详细数据失败: {str(e)}")
        return create_error_response(f"用户详细数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/expiry', methods=['GET'])
@async_route
async def get_enhanced_expiry_data():
    """
    ⏰ 获取增强到期数据 - 智能到期分析
    """
    try:
        logger.info("⏰ 获取增强到期数据...")

        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)

        # 验证参数
        params = validate_request_params()
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        date = params.get('date')  # 单日查询
        period = params.get('period', 'week')  # today, tomorrow, week, month, single, range

        # 参数验证
        if period == 'range':
            if not start_date or not end_date:
                return create_error_response(
                    "范围查询需要start_date和end_date参数",
                    "validation_error", 400
                )
            start_date, end_date = validate_date_range(start_date, end_date)
        elif period == 'single':
            if not date:
                return create_error_response(
                    "单日查询需要date参数",
                    "validation_error", 400
                )
            date = validate_date_format(date, "date")

        # 使用编排器根据period获取数据
        if period == 'today':
            result = await orchestrator.api_connector.get_expiring_products_today()
        elif period == 'tomorrow':
            result = await orchestrator.api_connector.get_expiring_products_tomorrow()
        elif period == 'week':
            result = await orchestrator.api_connector.get_expiring_products_week()
        elif period == 'single':
            result = await orchestrator.api_connector.get_product_end_data(date)
        elif period == 'range':
            result = await orchestrator.api_connector.get_expiring_products_range(start_date, end_date)
        else:
            return create_error_response(f"不支持的期间类型: {period}", "validation_error", 400)

        if not result.get("success"):
            return create_error_response(
                result.get("message", "到期数据获取失败"),
                "api_error", 500
            )

        # 使用编排器增强到期数据
        enhanced_data = orchestrator.data_enhancer.enhance_expiry_data(result["data"], period)

        # 智能风险评估
        risk_analysis = {}
        try:
            expiry_amount = float(result["data"].get("到期金额", 0))
            
            # 风险级别评估
            if expiry_amount > 5000000:  # 500万以上
                risk_level = 'high'
            elif expiry_amount > 1000000:  # 100万以上
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            risk_analysis = {
                'risk_level': risk_level,
                'expiry_amount': expiry_amount,
                'liquidity_requirement': expiry_amount * 1.1,  # 110%准备金
                'preparation_days_needed': 3 if risk_level == 'high' else 1
            }
            
            # 异常检测
            anomaly_result = await orchestrator.data_analyzer.detect_anomalies(
                'system', ['expiry_amount'], sensitivity=1.5
            )
            risk_analysis.update({
                'anomaly_detected': len(anomaly_result.anomalies) > 0,
                'anomaly_count': len(anomaly_result.anomalies),
                'confidence': anomaly_result.confidence_score
            })
        except Exception as e:
            logger.warning(f"风险分析失败: {str(e)}")
            risk_analysis = {'risk_level': 'unknown', 'confidence': 0.0}

        # 复投机会分析
        reinvestment_analysis = {}
        try:
            expiry_amount = float(result["data"].get("到期金额", 0))
            if expiry_amount > 0:
                # 估算复投潜力
                estimated_reinvestment_rate = 0.6  # 假设60%复投率
                potential_reinvestment = expiry_amount * estimated_reinvestment_rate
                
                reinvestment_analysis = {
                    'estimated_reinvestment_rate': estimated_reinvestment_rate,
                    'potential_reinvestment_amount': potential_reinvestment,
                    'cash_outflow_risk': expiry_amount * (1 - estimated_reinvestment_rate),
                    'optimization_opportunity': expiry_amount * 0.1  # 10%提升空间
                }
        except Exception as e:
            logger.warning(f"复投分析失败: {str(e)}")

        response_data = {
            'enhanced_data': enhanced_data,
            'risk_analysis': risk_analysis,
            'reinvestment_analysis': reinvestment_analysis,
            'query_params': {
                'period': period,
                'start_date': start_date,
                'end_date': end_date,
                'date': date
            },
            'processing_metadata': {
                'risk_assessment_included': True,
                'enhancement_applied': True,
                'intelligent_analysis': True,
                'reinvestment_analysis': True
            }
        }

        logger.info("✅ 到期数据增强处理完成")
        return create_success_response(response_data, "到期数据获取并分析成功")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 获取到期数据失败: {str(e)}")
        return create_error_response(f"到期数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/analytics', methods=['GET'])
@async_route
async def get_enhanced_analytics_data():
    """
    📈 获取增强分析数据 - 深度智能分析
    """
    try:
        logger.info("📈 获取增强分析数据...")

        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)

        # 验证参数
        params = validate_request_params()
        analysis_type = params.get('type', 'comprehensive')  # comprehensive, trend, performance, risk
        time_range = int(params.get('time_range', 30))
        include_insights = params.get('include_insights', 'true').lower() == 'true'
        
        # 参数验证
        if time_range < 1 or time_range > 365:
            return create_error_response("时间范围必须在1-365天之间", "validation_error", 400)

        # 使用编排器获取不同类型的分析数据
        analysis_results = []

        if analysis_type == 'comprehensive':
            # 综合分析 - 使用编排器的多种分析能力
            try:
                # 趋势分析
                trend_result = await orchestrator.data_analyzer.analyze_trend('system', 'total_balance', time_range)
                analysis_results.append(trend_result)

                # 业务绩效分析
                performance_result = await orchestrator.data_analyzer.analyze_business_performance('financial', time_range)
                analysis_results.append(performance_result)

                # 异常检测
                anomaly_result = await orchestrator.data_analyzer.detect_anomalies('system',
                                                                                   ['total_balance', 'daily_inflow'])
                analysis_results.append(anomaly_result)

            except Exception as e:
                logger.warning(f"部分分析失败: {str(e)}")

        elif analysis_type == 'trend':
            # 趋势分析
            trend_result = await orchestrator.data_analyzer.analyze_trend('system', 'total_balance', time_range)
            analysis_results.append(trend_result)
            
        elif analysis_type == 'performance':
            # 业务表现分析
            performance_result = await orchestrator.data_analyzer.analyze_business_performance('financial', time_range)
            analysis_results.append(performance_result)
            
        elif analysis_type == 'risk':
            # 风险分析
            anomaly_result = await orchestrator.data_analyzer.detect_anomalies('system', ['total_balance', 'daily_inflow'])
            analysis_results.append(anomaly_result)

        # 生成智能洞察（如果请求）
        insights_data = {}
        if include_insights and analysis_results:
            try:
                insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                    analysis_results=analysis_results,
                    user_context=None,
                    focus_areas=[analysis_type]
                )
                insights_data = {
                    'insights': insights,
                    'metadata': metadata
                }
            except Exception as e:
                logger.warning(f"洞察生成失败: {str(e)}")
                insights_data = {'error': '洞察生成暂不可用'}

        # 综合响应数据
        response_data = {
            'analysis_results': [
                {
                    'analysis_id': getattr(result, 'analysis_id', f'analysis_{i}'),
                    'type': result.analysis_type.value if hasattr(result, 'analysis_type') else 'unknown',
                    'confidence': result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                    'key_findings': result.key_findings if hasattr(result, 'key_findings') else [],
                    'trends': result.trends if hasattr(result, 'trends') else [],
                    'anomalies': result.anomalies if hasattr(result, 'anomalies') else [],
                    'metrics': result.metrics if hasattr(result, 'metrics') else {},
                    'business_insights': result.business_insights if hasattr(result, 'business_insights') else [],
                    'recommendations': result.recommendations if hasattr(result, 'recommendations') else []
                }
                for i, result in enumerate(analysis_results)
            ],
            'insights': insights_data,
            'analysis_summary': {
                'analysis_type': analysis_type,
                'time_range_days': time_range,
                'total_analyses': len(analysis_results),
                'insights_generated': include_insights and bool(insights_data),
                'overall_confidence': sum(getattr(r, 'confidence_score', 0) for r in analysis_results) / len(analysis_results) if analysis_results else 0
            },
            'processing_metadata': {
                'analysis_count': len(analysis_results),
                'insights_generated': include_insights and bool(insights_data),
                'intelligence_level': 'advanced',
                'processing_components': [
                    'data_analyzer',
                    'insight_generator' if include_insights else None,
                    'data_enhancer'
                ]
            }
        }

        logger.info(f"✅ {analysis_type}分析数据处理完成")
        return create_success_response(response_data, f"{analysis_type}分析完成")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 获取分析数据失败: {str(e)}")
        return create_error_response(f"分析数据处理失败: {str(e)}", "internal_error", 500)


@data_bp.route('/batch', methods=['POST'])
@async_route
async def get_enhanced_batch_data():
    """
    📦 批量获取增强数据 - 智能批处理
    """
    try:
        logger.info("📦 开始批量数据处理...")

        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)

        # 验证请求数据
        try:
            request_data = validate_request_params(['requests'])
        except ValueError as e:
            return create_error_response(str(e), "validation_error", 400)

        data_requests = request_data.get('requests', [])
        include_analysis = request_data.get('include_analysis', False)
        max_concurrent = min(int(request_data.get('max_concurrent', 5)), 10)  # 限制并发数

        if not data_requests:
            return create_error_response("请求列表为空", "validation_error", 400)
        
        if len(data_requests) > 20:
            return create_error_response("批量请求不能超过20个", "validation_error", 400)

        # 并行处理多个数据请求
        batch_results = []
        start_time = time.time()

        # 分批处理以控制并发
        for i in range(0, len(data_requests), max_concurrent):
            batch = data_requests[i:i + max_concurrent]
            batch_tasks = []
            
            for req in batch:
                task = asyncio.create_task(
                    self._process_single_batch_request(req, i + data_requests.index(req))
                )
                batch_tasks.append(task)
            
            batch_results.extend(await asyncio.gather(*batch_tasks, return_exceptions=True))

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                processed_results.append({
                    'request_id': f'req_{i}',
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)

        # 可选的综合分析
        comprehensive_analysis = {}
        if include_analysis and processed_results:
            try:
                successful_results = [r for r in processed_results if r['success']]
                if successful_results:
                    insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                        analysis_results=[],
                        user_context=None,
                        focus_areas=['batch_analysis']
                    )
                    comprehensive_analysis = {
                        'insights': insights,
                        'metadata': metadata
                    }
            except Exception as e:
                logger.warning(f"批量综合分析失败: {str(e)}")
                comprehensive_analysis = {'error': '综合分析暂不可用'}

        processing_time = time.time() - start_time
        successful_count = sum(1 for r in processed_results if r["success"])

        response_data = {
            'results': processed_results,
            'comprehensive_analysis': comprehensive_analysis,
            'batch_summary': {
                'total_requests': len(data_requests),
                'successful_requests': successful_count,
                'failed_requests': len(data_requests) - successful_count,
                'success_rate': f"{(successful_count / len(data_requests)) * 100:.1f}%",
                'processing_time': f"{processing_time:.2f}秒",
                'avg_time_per_request': f"{processing_time / len(data_requests):.2f}秒"
            },
            'processing_metadata': {
                'batch_processing': True,
                'enhancement_applied': True,
                'comprehensive_analysis_included': include_analysis,
                'max_concurrent_used': max_concurrent
            }
        }

        logger.info(f"✅ 批量数据处理完成: {successful_count}/{len(data_requests)} 成功")
        return create_success_response(response_data, "批量数据处理完成")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 批量数据处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(f"批量数据处理失败: {str(e)}", "internal_error", 500)


async def _process_single_batch_request(self, req: Dict[str, Any], index: int) -> Dict[str, Any]:
    """处理单个批量请求"""
    req_type = req.get('type')
    req_params = req.get('params', {})
    req_id = req.get('id', f'req_{index}')

    try:
        if req_type == 'system':
            result = await orchestrator.api_connector.get_system_data()
            if result.get('success'):
                enhanced_data = orchestrator.data_enhancer.enhance_system_data(result['data'])
                result['enhanced_data'] = enhanced_data

        elif req_type == 'daily':
            date = req_params.get('date')
            if date:
                date = validate_date_format(date, "date")
            result = await orchestrator.api_connector.get_daily_data(date)
            if result.get('success'):
                enhanced_data = orchestrator.data_enhancer.enhance_daily_data(
                    result['data'], date
                )
                result['enhanced_data'] = enhanced_data

        elif req_type == 'products':
            result = await orchestrator.api_connector.get_product_data()
            if result.get('success'):
                enhanced_data = orchestrator.data_enhancer.enhance_product_data(
                    result['data'], include_expiry=req_params.get('include_expiry', True)
                )
                result['enhanced_data'] = enhanced_data

        elif req_type == 'users_daily':
            date = req_params.get('date')
            if date:
                date = validate_date_format(date, "date")
            result = await orchestrator.api_connector.get_user_daily_data(date)
            if result.get('success'):
                enhanced_data = orchestrator.data_enhancer.enhance_user_daily_data(
                    result['data'], date
                )
                result['enhanced_data'] = enhanced_data

        elif req_type == 'users_detailed':
            page = int(req_params.get('page', 1))
            if page < 1:
                raise ValueError("页码必须大于0")
            result = await orchestrator.api_connector.get_user_data(page)
            if result.get('success'):
                enhanced_data = orchestrator.data_enhancer.enhance_user_detailed_data(
                    result['data'], req_params.get('include_analysis', False)
                )
                result['enhanced_data'] = enhanced_data

        elif req_type == 'expiry':
            period = req_params.get('period', 'week')
            if period == 'today':
                result = await orchestrator.api_connector.get_expiring_products_today()
            elif period == 'week':
                result = await orchestrator.api_connector.get_expiring_products_week()
            elif period == 'single':
                date = validate_date_format(req_params.get('date'), "date")
                result = await orchestrator.api_connector.get_product_end_data(date)
            elif period == 'range':
                start_date, end_date = validate_date_range(
                    req_params.get('start_date'),
                    req_params.get('end_date')
                )
                result = await orchestrator.api_connector.get_expiring_products_range(start_date, end_date)
            else:
                result = await orchestrator.api_connector.get_expiring_products_week()

            if result.get('success'):
                enhanced_data = orchestrator.data_enhancer.enhance_expiry_data(result['data'], period)
                result['enhanced_data'] = enhanced_data

        else:
            result = {'success': False, 'error': f'不支持的请求类型: {req_type}'}

        return {
            "request_id": req_id,
            "request_type": req_type,
            "success": result.get("success", False),
            "data": result.get("enhanced_data") or result.get("data") if result.get("success") else None,
            "error": result.get("error") or result.get("message") if not result.get("success") else None,
            "processing_time": time.time()
        }

    except Exception as e:
        logger.error(f"批处理请求 {req_id} 失败: {str(e)}")
        return {
            "request_id": req_id,
            "request_type": req_type,
            "success": False,
            "error": str(e),
            "processing_time": time.time()
        }


@data_bp.route('/insights/generate', methods=['POST'])
@async_route
async def generate_data_insights():
    """
    💡 数据洞察生成 - 使用编排器的洞察能力
    """
    try:
        logger.info("💡 生成数据洞察...")

        if not orchestrator.initialized:
            return create_error_response("系统未就绪", "system_unavailable", 503)

        # 验证请求数据
        try:
            request_data = validate_request_params(['data_type'])
        except ValueError as e:
            return create_error_response(str(e), "validation_error", 400)

        data_type = request_data.get('data_type')  # system, daily, product, user, expiry
        focus_areas = request_data.get('focus_areas', [])
        analysis_depth = request_data.get('analysis_depth', 'standard')  # basic, standard, comprehensive
        time_range = int(request_data.get('time_range', 30))

        # 参数验证
        valid_data_types = ['system', 'daily', 'product', 'user', 'expiry', 'financial']
        if data_type not in valid_data_types:
            return create_error_response(
                f"不支持的数据类型: {data_type}，支持的类型: {valid_data_types}",
                "validation_error", 400
            )

        # 根据数据类型获取相应的分析结果
        analysis_results = []

        if data_type == 'system':
            trend_result = await orchestrator.data_analyzer.analyze_trend('system', 'total_balance', time_range)
            performance_result = await orchestrator.data_analyzer.analyze_business_performance('financial', time_range)
            analysis_results.extend([trend_result, performance_result])

        elif data_type == 'daily':
            trend_result = await orchestrator.data_analyzer.analyze_trend('daily', 'net_inflow', min(time_range, 14))
            anomaly_result = await orchestrator.data_analyzer.detect_anomalies('daily', ['inflow', 'outflow'])
            analysis_results.extend([trend_result, anomaly_result])

        elif data_type == 'product':
            performance_result = await orchestrator.data_analyzer.analyze_business_performance('product', time_range)
            analysis_results.append(performance_result)

        elif data_type == 'user':
            performance_result = await orchestrator.data_analyzer.analyze_business_performance('user', time_range)
            analysis_results.append(performance_result)

        elif data_type == 'expiry':
            anomaly_result = await orchestrator.data_analyzer.detect_anomalies('system', ['expiry_amount'])
            analysis_results.append(anomaly_result)

        elif data_type == 'financial':
            trend_result = await orchestrator.data_analyzer.analyze_trend('system', 'total_balance', time_range)
            performance_result = await orchestrator.data_analyzer.analyze_business_performance('financial', time_range)
            risk_result = await orchestrator.data_analyzer.detect_anomalies('system', ['total_balance', 'daily_inflow'])
            analysis_results.extend([trend_result, performance_result, risk_result])

        # 使用编排器生成洞察
        insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
            analysis_results=analysis_results,
            user_context=None,
            focus_areas=focus_areas
        )

        response_data = {
            'insights': insights,
            'metadata': metadata,
            'analysis_summary': {
                'data_type': data_type,
                'analysis_depth': analysis_depth,
                'focus_areas': focus_areas,
                'time_range_days': time_range,
                'analysis_count': len(analysis_results),
                'insights_count': len(insights)
            },
            'processing_metadata': {
                'dual_ai_collaboration': True,
                'intelligent_analysis': True,
                'insight_quality': 'high',
                'business_actionable': True
            }
        }

        logger.info(f"✅ {data_type}数据洞察生成完成: {len(insights)}条洞察")
        return create_success_response(response_data, "数据洞察生成完成")

    except ValueError as e:
        return create_error_response(str(e), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ 数据洞察生成失败: {str(e)}")
        return create_error_response(f"洞察生成失败: {str(e)}", "internal_error", 500)


@data_bp.route('/generate-report', methods=['POST'])
@async_route
async def generate_analysis_report():
    """
    📊 生成分析报告 - 基于数据分析结果创建专业报告
    
    支持多种报告类型:
    - 金融分析报告
    - 趋势分析报告
    - 对比分析报告
    
    支持多种输出格式:
    - HTML
    - PDF
    - Markdown
    """
    try:
        # 验证请求参数
        params = validate_request_params(['report_type', 'data'])
        
        report_type = params.get('report_type', 'financial')
        report_data = params.get('data', {})
        report_title = params.get('title', '数据分析报告')
        report_format = params.get('format', 'html').lower()
        output_path = params.get('output_path', None)
        
        # 验证报告类型
        valid_report_types = ['financial', 'trend', 'comparison']
        if report_type not in valid_report_types:
            return create_error_response(
                f"无效的报告类型: {report_type}，支持的类型: {', '.join(valid_report_types)}", 
                "invalid_parameter", 
                400
            )
        
        # 验证报告格式
        valid_formats = ['html', 'pdf', 'markdown']
        if report_format not in valid_formats:
            return create_error_response(
                f"无效的报告格式: {report_format}，支持的格式: {', '.join(valid_formats)}", 
                "invalid_parameter", 
                400
            )
        
        # 转换报告格式为枚举类型
        from utils.formatters.report_generator import ReportFormat
        format_mapping = {
            'html': ReportFormat.HTML,
            'pdf': ReportFormat.PDF,
            'markdown': ReportFormat.MARKDOWN
        }
        output_format = format_mapping[report_format]
        
        # 创建报告生成器
        from utils.formatters.report_generator import create_report_generator
        report_generator = create_report_generator()
        
        # 根据报告类型生成相应的报告
        if report_type == 'financial':
            report_result = report_generator.generate_financial_report(
                data=report_data,
                title=report_title,
                output_format=output_format,
                output_path=output_path
            )
        elif report_type == 'trend':
            period = params.get('period', None)
            report_result = report_generator.generate_trend_report(
                trend_data=report_data,
                title=report_title,
                period=period,
                output_format=output_format,
                output_path=output_path
            )
        elif report_type == 'comparison':
            report_result = report_generator.generate_comparison_report(
                comparison_data=report_data,
                title=report_title,
                output_format=output_format,
                output_path=output_path
            )
        
        # 处理报告结果
        if output_path and isinstance(report_result, str):
            # 如果指定了输出路径，返回文件路径
            return create_success_response({
                'report_path': report_result,
                'report_type': report_type,
                'format': report_format
            }, "报告生成成功")
        else:
            # 否则返回报告内容
            if report_format == 'html':
                report_content = report_result.to_html()
            elif report_format == 'markdown':
                report_content = report_result.to_markdown()
            else:
                # PDF格式需要保存到临时文件
                import tempfile
                import os
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"report_{int(time.time())}.pdf")
                report_result.save(temp_file, ReportFormat.PDF)
                report_content = temp_file
            
            return create_success_response({
                'report_content': report_content,
                'report_type': report_type,
                'format': report_format
            }, "报告生成成功")
            
    except Exception as e:
        logger.error(f"报告生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(f"报告生成失败: {str(e)}", "report_generation_error", 500)


@data_bp.route('/generate-data-report', methods=['POST'])
@async_route
async def generate_data_report():
    """
    📈 生成数据报告 - 基于特定数据类型生成专业报告
    
    支持的数据类型:
    - user_data: 用户数据分析报告
    - financial_data: 财务数据分析报告
    - product_data: 产品数据分析报告
    - trend_data: 趋势数据分析报告
    """
    try:
        # 验证请求参数
        params = validate_request_params(['data_type'])
        
        data_type = params.get('data_type')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        user_id = params.get('user_id')
        product_id = params.get('product_id')
        report_format = params.get('format', 'html').lower()
        
        # 验证日期格式（如果提供）
        if start_date and end_date:
            validate_date_range(start_date, end_date)
        
        # 根据数据类型获取相应的数据
        report_data = {}
        report_title = "数据分析报告"
        
        if data_type == 'user_data':
            # 获取用户数据
            if not user_id:
                return create_error_response("缺少必要参数: user_id", "missing_parameter", 400)
            
            # 获取详细用户数据
            user_data_params = {'user_id': user_id}
            if start_date and end_date:
                user_data_params.update({'start_date': start_date, 'end_date': end_date})
            
            user_data_result = await orchestrator.get_enhanced_user_data(user_data_params)
            
            # 准备报告数据
            report_data = {
                'subtitle': f"用户 {user_id} 数据分析",
                'summary': {
                    'content': f"本报告分析了用户 {user_id} 在系统中的行为和表现。",
                    'key_metrics': []
                },
                'analysis_sections': []
            }
            
            # 添加用户基本信息
            if 'user_info' in user_data_result:
                user_info = user_data_result['user_info']
                basic_info_section = {
                    'title': "用户基本信息",
                    'content': "用户的基本属性和注册信息。",
                    'metrics': []
                }
                
                # 添加用户指标
                for key, value in user_info.items():
                    if key not in ['id', 'user_id']:
                        basic_info_section['metrics'].append({
                            'name': key,
                            'value': value,
                            'format_type': 'plain'
                        })
                
                report_data['analysis_sections'].append(basic_info_section)
            
            # 添加用户活动数据
            if 'activity_data' in user_data_result:
                activity_data = user_data_result['activity_data']
                activity_section = {
                    'title': "用户活动分析",
                    'content': "用户的活动模式和参与度分析。",
                    'charts': []
                }
                
                # 创建活动趋势图表
                if 'daily_activity' in activity_data:
                    activity_section['charts'].append({
                        'chart_type': 'line',
                        'title': '用户活动趋势',
                        'data': activity_data['daily_activity'],
                        'caption': '用户每日活动量变化趋势'
                    })
                
                report_data['analysis_sections'].append(activity_section)
            
            report_title = f"用户 {user_id} 数据分析报告"
            
        elif data_type == 'financial_data':
            # 获取财务数据
            financial_params = {}
            if start_date and end_date:
                financial_params.update({'start_date': start_date, 'end_date': end_date})
            
            financial_data_result = await orchestrator.get_enhanced_financial_data(financial_params)
            
            # 准备报告数据
            period_text = ""
            if start_date and end_date:
                period_text = f"（{start_date} 至 {end_date}）"
            
            report_data = {
                'subtitle': f"财务数据分析{period_text}",
                'summary': {
                    'content': "本报告提供了关键财务指标和趋势的综合分析。",
                    'key_metrics': []
                },
                'analysis_sections': []
            }
            
            # 添加财务摘要指标
            if 'summary' in financial_data_result:
                for key, value in financial_data_result['summary'].items():
                    report_data['summary']['key_metrics'].append({
                        'name': key,
                        'value': value,
                        'format_type': 'currency' if 'revenue' in key.lower() or 'profit' in key.lower() else 'percentage' if 'rate' in key.lower() or 'growth' in key.lower() else 'plain'
                    })
            
            # 添加收入分析部分
            if 'revenue_data' in financial_data_result:
                revenue_data = financial_data_result['revenue_data']
                revenue_section = {
                    'title': "收入分析",
                    'content': "收入来源和趋势分析。",
                    'charts': []
                }
                
                # 创建收入趋势图表
                if 'trend' in revenue_data:
                    revenue_section['charts'].append({
                        'chart_type': 'line',
                        'title': '收入趋势',
                        'data': revenue_data['trend'],
                        'caption': '收入变化趋势分析'
                    })
                
                # 创建收入分布图表
                if 'distribution' in revenue_data:
                    revenue_section['charts'].append({
                        'chart_type': 'pie',
                        'title': '收入分布',
                        'data': revenue_data['distribution'],
                        'caption': '收入来源分布'
                    })
                
                report_data['analysis_sections'].append(revenue_section)
            
            report_title = "财务数据分析报告"
            
        elif data_type == 'product_data':
            # 获取产品数据
            product_params = {}
            if product_id:
                product_params['product_id'] = product_id
            if start_date and end_date:
                product_params.update({'start_date': start_date, 'end_date': end_date})
            
            product_data_result = await orchestrator.get_enhanced_product_data(product_params)
            
            # 准备报告数据
            product_text = f"产品 {product_id}" if product_id else "产品"
            period_text = ""
            if start_date and end_date:
                period_text = f"（{start_date} 至 {end_date}）"
            
            report_data = {
                'subtitle': f"{product_text}数据分析{period_text}",
                'summary': {
                    'content': f"本报告提供了{product_text}性能和趋势的综合分析。",
                    'key_metrics': []
                },
                'analysis_sections': []
            }
            
            # 添加产品摘要指标
            if 'summary' in product_data_result:
                for key, value in product_data_result['summary'].items():
                    report_data['summary']['key_metrics'].append({
                        'name': key,
                        'value': value,
                        'format_type': 'currency' if 'revenue' in key.lower() or 'price' in key.lower() else 'plain'
                    })
            
            # 添加产品性能分析部分
            if 'performance_data' in product_data_result:
                performance_data = product_data_result['performance_data']
                performance_section = {
                    'title': "产品性能分析",
                    'content': "产品性能指标和趋势分析。",
                    'charts': []
                }
                
                # 创建性能趋势图表
                if 'trend' in performance_data:
                    performance_section['charts'].append({
                        'chart_type': 'line',
                        'title': '性能趋势',
                        'data': performance_data['trend'],
                        'caption': '产品性能变化趋势'
                    })
                
                report_data['analysis_sections'].append(performance_section)
            
            report_title = f"{product_text}数据分析报告"
            
        elif data_type == 'trend_data':
            # 获取趋势数据
            trend_params = {}
            if start_date and end_date:
                trend_params.update({'start_date': start_date, 'end_date': end_date})
            
            trend_data_result = await orchestrator.get_enhanced_trend_data(trend_params)
            
            # 准备报告数据
            period_text = ""
            if start_date and end_date:
                period_text = f"{start_date} 至 {end_date}"
            
            report_data = {
                'subtitle': "趋势分析",
                'period': period_text,
                'summary': {
                    'content': "本报告提供了关键指标的趋势分析和预测。",
                    'key_trends': []
                },
                'trends': []
            }
            
            # 添加关键趋势指标
            if 'key_trends' in trend_data_result:
                for trend in trend_data_result['key_trends']:
                    report_data['summary']['key_trends'].append({
                        'name': trend['name'],
                        'value': trend['value'],
                        'format_type': trend.get('format_type', 'percentage'),
                        'description': trend.get('description', '')
                    })
            
            # 添加各指标趋势
            if 'trends' in trend_data_result:
                for trend in trend_data_result['trends']:
                    trend_item = {
                        'title': trend['title'],
                        'description': trend.get('description', ''),
                        'chart': trend.get('chart', {}),
                        'data': trend.get('data', []),
                        'analysis': trend.get('analysis', '')
                    }
                    report_data['trends'].append(trend_item)
            
            report_title = "趋势分析报告"
            
        else:
            return create_error_response(
                f"不支持的数据类型: {data_type}", 
                "unsupported_data_type", 
                400
            )
        
        # 转换报告格式为枚举类型
        from utils.formatters.report_generator import ReportFormat
        format_mapping = {
            'html': ReportFormat.HTML,
            'pdf': ReportFormat.PDF,
            'markdown': ReportFormat.MARKDOWN
        }
        output_format = format_mapping[report_format]
        
        # 创建报告生成器
        from utils.formatters.report_generator import create_report_generator
        report_generator = create_report_generator()
        
        # 根据数据类型选择适当的报告生成方法
        if data_type == 'trend_data':
            report_result = report_generator.generate_trend_report(
                trend_data=report_data,
                title=report_title,
                period=period_text if period_text else None,
                output_format=output_format
            )
        else:
            report_result = report_generator.generate_financial_report(
                data=report_data,
                title=report_title,
                output_format=output_format
            )
        
        # 处理报告结果
        if report_format == 'html':
            report_content = report_result.to_html()
        elif report_format == 'markdown':
            report_content = report_result.to_markdown()
        else:
            # PDF格式需要保存到临时文件
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"report_{int(time.time())}.pdf")
            report_result.save(temp_file, ReportFormat.PDF)
            report_content = temp_file
        
        return create_success_response({
            'report_content': report_content,
            'report_type': data_type,
            'format': report_format
        }, "报告生成成功")
            
    except Exception as e:
        logger.error(f"数据报告生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(f"数据报告生成失败: {str(e)}", "report_generation_error", 500)


# ============= 系统监控和状态API =============

@data_bp.route('/health', methods=['GET'])
@async_route
async def data_service_health():
    """🔍 数据服务健康检查"""
    try:
        # 检查编排器健康状态
        health_status = await orchestrator.health_check()

        # 检查各数据组件状态
        component_status = {
            'api_connector': 'healthy' if hasattr(orchestrator, 'api_connector') else 'unavailable',
            'data_analyzer': 'healthy' if hasattr(orchestrator, 'data_analyzer') else 'unavailable',
            'data_enhancer': 'healthy' if hasattr(orchestrator, 'data_enhancer') else 'unavailable',
            'insight_generator': 'healthy' if hasattr(orchestrator, 'insight_generator') else 'unavailable'
        }

        # 系统性能统计
        orchestrator_stats = orchestrator.get_orchestrator_stats()

        # API连接器统计
        api_stats = {}
        if hasattr(orchestrator, 'api_connector'):
            api_stats = orchestrator.api_connector.get_connector_stats()

        overall_status = 'healthy' if all(status == 'healthy' for status in component_status.values()) else 'degraded'

        response_data = {
            'overall_status': overall_status,
            'orchestrator_health': health_status,
            'component_status': component_status,
            'system_stats': orchestrator_stats,
            'api_connector_stats': api_stats,
            'service_info': {
                'service_name': 'intelligent_data_service',
                'version': '2.0.0',
                'supported_apis': [
                    '/api/sta/system',
                    '/api/sta/day',
                    '/api/sta/product',
                    '/api/sta/user_daily',
                    '/api/sta/user',
                    '/api/sta/product_end',
                    '/api/sta/product_end_interval'
                ],
                'features': [
                    'intelligent_data_enhancement',
                    'multi_source_analysis',
                    'automated_insights',
                    'batch_processing',
                    'real_time_analytics',
                    'risk_assessment',
                    'trend_analysis',
                    'user_behavior_analysis'
                ]
            },
            'api_endpoints': {
                'data_endpoints': 8,
                'analysis_endpoints': 2,
                'utility_endpoints': 2
            }
        }

        status_code = 200 if overall_status == 'healthy' else 503
        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return create_error_response("健康检查失败", "health_check_error", 500)


@data_bp.route('/stats', methods=['GET'])
@async_route
async def get_data_service_stats():
    """📊 获取数据服务统计信息"""
    try:
        orchestrator_stats = orchestrator.get_orchestrator_stats()
        
        # API连接器统计
        api_stats = {}
        if hasattr(orchestrator, 'api_connector'):
            api_stats = orchestrator.api_connector.get_connector_stats()

        # 数据分析器统计
        analyzer_stats = {}
        if hasattr(orchestrator, 'data_analyzer'):
            analyzer_stats = orchestrator.data_analyzer.get_analysis_stats()

        # 洞察生成器统计
        insight_stats = {}
        if hasattr(orchestrator, 'insight_generator'):
            insight_stats = orchestrator.insight_generator.get_insight_stats()

        response_data = {
            'orchestrator_stats': orchestrator_stats,
            'api_connector_stats': api_stats,
            'data_analyzer_stats': analyzer_stats,
            'insight_generator_stats': insight_stats,
            'service_summary': {
                'total_components': 4,
                'active_components': sum(1 for stats in [api_stats, analyzer_stats, insight_stats] if stats),
                'health_score': 1.0 if all([api_stats, analyzer_stats, insight_stats]) else 0.75
            }
        }

        return create_success_response(response_data, "数据服务统计信息获取成功")

    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return create_error_response(f"统计信息获取失败: {str(e)}", "internal_error", 500)


# ============= 错误处理 =============

@data_bp.errorhandler(404)
def data_not_found(error):
    """404错误处理"""
    return create_error_response("数据服务端点不存在", "not_found", 404)


@data_bp.errorhandler(500)
def data_internal_error(error):
    """500错误处理"""
    logger.error(f"数据服务内部错误: {str(error)}")
    return create_error_response("数据服务内部错误", "internal_error", 500)


@data_bp.errorhandler(400)
def data_bad_request(error):
    """400错误处理"""
    return create_error_response("请求参数错误", "bad_request", 400)