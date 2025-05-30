# api/qa_routes.py - 完整优化版本
from flask import Blueprint, jsonify, request
from core.orchestrator.intelligent_qa_orchestrator import get_orchestrator
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import time

logger = logging.getLogger(__name__)
qa_routes_bp = Blueprint('qa_routes', __name__, url_prefix='/api/qa')

# 🎯 使用单一编排器实例
orchestrator = get_orchestrator()


# ============= 工具函数 =============

def async_wrapper(f):
    """异步包装器装饰器"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = f.__name__
    return wrapper


def validate_query_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证查询请求数据"""
    if not data:
        raise ValueError("请求数据为空")
    
    query = data.get('query', '').strip()
    if not query:
        raise ValueError("查询内容不能为空")
    
    if len(query) > 1000:
        raise ValueError("查询内容过长，请控制在1000字符以内")
    
    user_id = data.get('user_id', 1)
    if not isinstance(user_id, int) or user_id < 1:
        raise ValueError("user_id必须是正整数")
    
    return {
        'query': query,
        'user_id': user_id,
        'conversation_id': data.get('conversation_id'),
        'preferences': data.get('preferences', {}),
        'context': data.get('context', {}),
        'analysis_depth': data.get('analysis_depth', 'standard'),
        'include_visualizations': data.get('include_visualizations', True),
        'response_format': data.get('response_format', 'detailed')
    }


def create_success_response(data: Dict[str, Any], message: str = "操作成功") -> Dict[str, Any]:
    """创建成功响应"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'service': 'intelligent_qa_service'
    })


def create_error_response(message: str, error_type: str = "processing_error", 
                         status_code: int = 500, details: Dict[str, Any] = None) -> tuple:
    """创建错误响应"""
    error_response = {
        'success': False,
        'error_type': error_type,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'service': 'intelligent_qa_service'
    }
    
    if details:
        error_response['details'] = details
    
    return jsonify(error_response), status_code


# ============= 核心问答API =============

@qa_routes_bp.route('/ask', methods=['POST'])
def intelligent_question_processing():
    """🧠 智能问答 - 完整编排器处理"""

    @async_wrapper
    async def _process_query():
        try:
            # 参数验证
            request_data = request.get_json()
            if not request_data:
                return create_error_response("请求数据格式错误", "validation_error", 400)
            
            validated_data = validate_query_request(request_data)
            
            logger.info(f"🧠 处理智能查询: {validated_data['query'][:50]}...")
            
            # 检查编排器状态
            if not orchestrator.initialized:
                return create_error_response("智能问答系统未就绪", "system_unavailable", 503)
            
            start_time = time.time()
            
            # 使用编排器处理查询
            result = await orchestrator.process_intelligent_query(
                user_query=validated_data['query'],
                user_id=validated_data['user_id'],
                conversation_id=validated_data['conversation_id'],
                preferences=validated_data['preferences']
            )
            
            processing_time = time.time() - start_time
            
            # 增强响应数据
            response_data = {
                'query_result': {
                    'success': result.success,
                    'response_text': result.response_text,
                    'insights': result.insights,
                    'visualizations': result.visualizations if validated_data['include_visualizations'] else [],
                    'confidence_score': result.confidence_score,
                    'conversation_id': result.conversation_id
                },
                'analysis_metadata': {
                    'query_complexity': result.processing_metadata.get('query_complexity', 'unknown'),
                    'processing_time': processing_time,
                    'ai_models_used': result.processing_metadata.get('ai_models_used', []),
                    'data_sources_accessed': result.processing_metadata.get('data_sources', []),
                    'analysis_depth': validated_data['analysis_depth']
                },
                'quality_metrics': {
                    'response_completeness': len(result.response_text) > 100,
                    'insights_provided': len(result.insights) > 0,
                    'visualizations_included': len(result.visualizations) > 0,
                    'confidence_level': 'high' if result.confidence_score > 0.8 else 'medium' if result.confidence_score > 0.6 else 'low'
                }
            }
            
            logger.info(f"✅ 查询处理完成: 置信度={result.confidence_score:.2f}, 耗时={processing_time:.2f}秒")
            
            return create_success_response(response_data, "智能查询处理完成")
        
        except ValueError as e:
            return create_error_response(str(e), "validation_error", 400)
        except Exception as e:
            logger.error(f"❌ 智能查询处理失败: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_response(f"查询处理失败: {str(e)}", "internal_error", 500)

    return _process_query()


@qa_routes_bp.route('/analyze', methods=['POST'])
def intelligent_data_analysis():
    """📊 智能数据分析 - 使用编排器的分析能力"""

    @async_wrapper
    async def _analyze_data():
        try:
            request_data = request.get_json()
            if not request_data:
                return create_error_response("请求数据格式错误", "validation_error", 400)
            
            # 参数验证
            data_source = request_data.get('data_source', 'system')
            metric = request_data.get('metric', 'total_balance')
            time_range = int(request_data.get('time_range', 30))
            analysis_type = request_data.get('analysis_type', 'trend')  # trend, performance, anomaly, comprehensive
            
            # 参数合法性检查
            valid_data_sources = ['system', 'daily', 'product', 'user', 'expiry']
            valid_analysis_types = ['trend', 'performance', 'anomaly', 'comprehensive']
            
            if data_source not in valid_data_sources:
                return create_error_response(
                    f"不支持的数据源: {data_source}，支持: {valid_data_sources}",
                    "validation_error", 400
                )
            
            if analysis_type not in valid_analysis_types:
                return create_error_response(
                    f"不支持的分析类型: {analysis_type}，支持: {valid_analysis_types}",
                    "validation_error", 400
                )
            
            if not (1 <= time_range <= 365):
                return create_error_response(
                    "时间范围必须在1-365天之间",
                    "validation_error", 400
                )
            
            logger.info(f"📊 执行数据分析: {analysis_type} - {data_source}.{metric}")
            
            # 检查编排器状态
            if not orchestrator.initialized:
                return create_error_response("数据分析系统未就绪", "system_unavailable", 503)
            
            start_time = time.time()
            analysis_results = []
            
            # 根据分析类型执行不同的分析
            if analysis_type == 'trend':
                # 趋势分析
                analysis_result = await orchestrator.data_analyzer.analyze_trend(
                    data_source=data_source,
                    metric=metric,
                    time_range=time_range
                )
                analysis_results.append(analysis_result)
                
            elif analysis_type == 'performance':
                # 业务表现分析
                analysis_result = await orchestrator.data_analyzer.analyze_business_performance(
                    scope=data_source,
                    time_range=time_range
                )
                analysis_results.append(analysis_result)
                
            elif analysis_type == 'anomaly':
                # 异常检测
                metrics_to_check = [metric] if metric else ['total_balance', 'daily_inflow', 'daily_outflow']
                analysis_result = await orchestrator.data_analyzer.detect_anomalies(
                    data_source=data_source,
                    metrics=metrics_to_check,
                    sensitivity=2.0
                )
                analysis_results.append(analysis_result)
                
            elif analysis_type == 'comprehensive':
                # 综合分析
                trend_result = await orchestrator.data_analyzer.analyze_trend(
                    data_source=data_source, metric=metric, time_range=time_range
                )
                performance_result = await orchestrator.data_analyzer.analyze_business_performance(
                    scope=data_source, time_range=time_range
                )
                anomaly_result = await orchestrator.data_analyzer.detect_anomalies(
                    data_source=data_source, metrics=[metric]
                )
                analysis_results.extend([trend_result, performance_result, anomaly_result])
            
            processing_time = time.time() - start_time
            
            # 整理分析结果
            formatted_results = []
            for result in analysis_results:
                formatted_result = {
                    'analysis_id': result.analysis_id if hasattr(result, 'analysis_id') else f'analysis_{len(formatted_results)}',
                    'analysis_type': result.analysis_type.value if hasattr(result, 'analysis_type') else analysis_type,
                    'confidence_score': result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                    'key_findings': result.key_findings if hasattr(result, 'key_findings') else [],
                    'trends': result.trends if hasattr(result, 'trends') else [],
                    'anomalies': result.anomalies if hasattr(result, 'anomalies') else [],
                    'metrics': result.metrics if hasattr(result, 'metrics') else {},
                    'business_insights': result.business_insights if hasattr(result, 'business_insights') else [],
                    'recommendations': result.recommendations if hasattr(result, 'recommendations') else [],
                    'processing_time': result.processing_time if hasattr(result, 'processing_time') else 0
                }
                formatted_results.append(formatted_result)
            
            # 计算综合指标
            avg_confidence = sum(r['confidence_score'] for r in formatted_results) / len(formatted_results) if formatted_results else 0
            total_findings = sum(len(r['key_findings']) for r in formatted_results)
            total_anomalies = sum(len(r['anomalies']) for r in formatted_results)
            
            response_data = {
                'analysis_results': formatted_results,
                'analysis_summary': {
                    'total_analyses': len(formatted_results),
                    'avg_confidence': avg_confidence,
                    'total_key_findings': total_findings,
                    'total_anomalies_detected': total_anomalies,
                    'overall_status': 'healthy' if total_anomalies == 0 else 'needs_attention'
                },
                'request_parameters': {
                    'data_source': data_source,
                    'metric': metric,
                    'time_range': time_range,
                    'analysis_type': analysis_type
                },
                'processing_metadata': {
                    'processing_time': processing_time,
                    'data_quality': 'high',
                    'ai_enhanced': True
                }
            }
            
            logger.info(f"✅ 数据分析完成: {len(formatted_results)}个分析结果")
            
            return create_success_response(response_data, "数据分析完成")
        
        except ValueError as e:
            return create_error_response(str(e), "validation_error", 400)
        except Exception as e:
            logger.error(f"❌ 数据分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_response(f"数据分析失败: {str(e)}", "internal_error", 500)

    return _analyze_data()


@qa_routes_bp.route('/insights', methods=['POST'])
def generate_business_insights():
    """💡 业务洞察生成 - 使用编排器的洞察生成器"""

    @async_wrapper
    async def _generate_insights():
        try:
            request_data = request.get_json()
            if not request_data:
                return create_error_response("请求数据格式错误", "validation_error", 400)
            
            # 参数验证
            data_type = request_data.get('data_type', 'comprehensive')
            focus_areas = request_data.get('focus_areas', [])
            time_range = int(request_data.get('time_range', 30))
            insight_depth = request_data.get('insight_depth', 'standard')  # basic, standard, comprehensive
            include_recommendations = request_data.get('include_recommendations', True)
            
            # 参数合法性检查
            valid_data_types = ['system', 'daily', 'product', 'user', 'expiry', 'financial', 'comprehensive']
            valid_depths = ['basic', 'standard', 'comprehensive']
            
            if data_type not in valid_data_types:
                return create_error_response(
                    f"不支持的数据类型: {data_type}，支持: {valid_data_types}",
                    "validation_error", 400
                )
            
            if insight_depth not in valid_depths:
                return create_error_response(
                    f"不支持的洞察深度: {insight_depth}，支持: {valid_depths}",
                    "validation_error", 400
                )
            
            if not (1 <= time_range <= 365):
                return create_error_response(
                    "时间范围必须在1-365天之间",
                    "validation_error", 400
                )
            
            logger.info(f"💡 生成业务洞察: {data_type} - 深度: {insight_depth}")
            
            # 检查编排器状态
            if not orchestrator.initialized:
                return create_error_response("洞察生成系统未就绪", "system_unavailable", 503)
            
            start_time = time.time()
            
            # 获取相关分析结果作为洞察生成的基础
            analysis_results = []
            
            if data_type in ['system', 'financial', 'comprehensive']:
                # 获取系统和财务分析
                trend_result = await orchestrator.data_analyzer.analyze_trend('system', 'total_balance', time_range)
                performance_result = await orchestrator.data_analyzer.analyze_business_performance('financial', time_range)
                analysis_results.extend([trend_result, performance_result])
            
            if data_type in ['daily', 'comprehensive']:
                # 获取每日数据分析
                daily_trend = await orchestrator.data_analyzer.analyze_trend('daily', 'net_inflow', min(time_range, 14))
                analysis_results.append(daily_trend)
            
            if data_type in ['product', 'comprehensive']:
                # 获取产品分析
                product_performance = await orchestrator.data_analyzer.analyze_business_performance('product', time_range)
                analysis_results.append(product_performance)
            
            if data_type in ['user', 'comprehensive']:
                # 获取用户分析
                user_performance = await orchestrator.data_analyzer.analyze_business_performance('user', time_range)
                analysis_results.append(user_performance)
            
            if data_type in ['expiry', 'comprehensive']:
                # 获取到期风险分析
                expiry_anomalies = await orchestrator.data_analyzer.detect_anomalies('system', ['expiry_amount'])
                analysis_results.append(expiry_anomalies)
            
            # 使用编排器的洞察生成器
            insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                analysis_results=analysis_results,
                user_context=None,
                focus_areas=focus_areas if focus_areas else [data_type]
            )
            
            processing_time = time.time() - start_time
            
            # 按优先级分类洞察
            critical_insights = [i for i in insights if hasattr(i, 'priority') and i.priority.value == 'critical']
            high_insights = [i for i in insights if hasattr(i, 'priority') and i.priority.value == 'high']
            medium_insights = [i for i in insights if hasattr(i, 'priority') and i.priority.value == 'medium']
            low_insights = [i for i in insights if hasattr(i, 'priority') and i.priority.value == 'low']
            
            # 提取行动建议
            all_recommendations = []
            for insight in insights:
                if hasattr(insight, 'recommended_actions'):
                    all_recommendations.extend(insight.recommended_actions)
            
            # 格式化洞察数据
            formatted_insights = []
            for insight in insights:
                formatted_insight = {
                    'insight_id': insight.insight_id if hasattr(insight, 'insight_id') else f'insight_{len(formatted_insights)}',
                    'type': insight.insight_type.value if hasattr(insight, 'insight_type') else 'general',
                    'priority': insight.priority.value if hasattr(insight, 'priority') else 'medium',
                    'title': insight.title if hasattr(insight, 'title') else 'Business Insight',
                    'summary': insight.summary if hasattr(insight, 'summary') else '',
                    'detailed_analysis': insight.detailed_analysis if hasattr(insight, 'detailed_analysis') else '',
                    'confidence_score': insight.confidence_score if hasattr(insight, 'confidence_score') else 0.0,
                    'key_metrics': insight.key_metrics if hasattr(insight, 'key_metrics') else {},
                    'recommended_actions': insight.recommended_actions if hasattr(insight, 'recommended_actions') else [],
                    'expected_impact': insight.expected_impact if hasattr(insight, 'expected_impact') else '',
                    'data_sources': insight.data_sources if hasattr(insight, 'data_sources') else []
                }
                formatted_insights.append(formatted_insight)
            
            response_data = {
                'insights': formatted_insights,
                'insights_summary': {
                    'total_insights': len(formatted_insights),
                    'critical_count': len(critical_insights),
                    'high_priority_count': len(high_insights),
                    'medium_priority_count': len(medium_insights),
                    'low_priority_count': len(low_insights),
                    'avg_confidence': sum(i['confidence_score'] for i in formatted_insights) / len(formatted_insights) if formatted_insights else 0,
                    'actionable_insights': len([i for i in formatted_insights if i['recommended_actions']])
                },
                'recommendations': all_recommendations if include_recommendations else [],
                'metadata': metadata,
                'request_parameters': {
                    'data_type': data_type,
                    'focus_areas': focus_areas,
                    'time_range': time_range,
                    'insight_depth': insight_depth
                },
                'processing_metadata': {
                    'processing_time': processing_time,
                    'analysis_inputs': len(analysis_results),
                    'ai_collaboration': True,
                    'business_context_applied': True
                }
            }
            
            logger.info(f"✅ 业务洞察生成完成: {len(formatted_insights)}条洞察")
            
            return create_success_response(response_data, "业务洞察生成完成")
        
        except ValueError as e:
            return create_error_response(str(e), "validation_error", 400)
        except Exception as e:
            logger.error(f"❌ 业务洞察生成失败: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_response(f"洞察生成失败: {str(e)}", "internal_error", 500)

    return _generate_insights()


@qa_routes_bp.route('/query/parse', methods=['POST'])
def parse_complex_query():
    """🔍 复杂查询解析 - 查询理解和分析"""

    @async_wrapper
    async def _parse_query():
        try:
            request_data = request.get_json()
            if not request_data:
                return create_error_response("请求数据格式错误", "validation_error", 400)
            
            query = request_data.get('query', '').strip()
            if not query:
                return create_error_response("查询内容不能为空", "validation_error", 400)
            
            context = request_data.get('context', {})
            parse_depth = request_data.get('parse_depth', 'standard')  # basic, standard, comprehensive
            
            logger.info(f"🔍 解析复杂查询: {query[:50]}...")
            
            # 检查编排器状态
            if not orchestrator.initialized:
                return create_error_response("查询解析系统未就绪", "system_unavailable", 503)
            
            start_time = time.time()
            
            # 使用编排器的查询解析器
            parse_result = await orchestrator.query_parser.parse_complex_query(query, context)
            
            processing_time = time.time() - start_time
            
            # 格式化解析结果
            response_data = {
                'query_analysis': {
                    'original_query': parse_result.original_query,
                    'complexity': parse_result.complexity.value,
                    'query_type': parse_result.query_type.value,
                    'business_scenario': parse_result.business_scenario.value,
                    'confidence_score': parse_result.confidence_score
                },
                'time_requirements': parse_result.time_requirements,
                'data_requirements': {
                    'required_apis': parse_result.required_apis,
                    'data_sources': parse_result.data_requirements
                },
                'business_parameters': parse_result.business_parameters,
                'execution_plan': [
                    {
                        'step_id': step.step_id,
                        'step_type': step.step_type,
                        'description': step.description,
                        'estimated_time': step.estimated_time,
                        'ai_model_preference': step.ai_model_preference
                    }
                    for step in parse_result.execution_plan
                ],
                'ai_collaboration_plan': parse_result.ai_collaboration_plan,
                'processing_strategy': parse_result.processing_strategy,
                'estimated_total_time': parse_result.estimated_total_time,
                'processing_metadata': {
                    'parsing_time': processing_time,
                    'parse_depth': parse_depth,
                    'date_parse_success': parse_result.date_parse_result is not None,
                    'execution_steps': len(parse_result.execution_plan)
                }
            }
            
            logger.info(f"✅ 查询解析完成: 复杂度={parse_result.complexity.value}")
            
            return create_success_response(response_data, "查询解析完成")
        
        except Exception as e:
            logger.error(f"❌ 查询解析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_response(f"查询解析失败: {str(e)}", "internal_error", 500)

    return _parse_query()


@qa_routes_bp.route('/users/analyze', methods=['POST'])
def analyze_user_behavior():
    """👥 用户行为分析 - 专门的用户数据分析"""
    
    @async_wrapper
    async def _analyze_users():
        try:
            request_data = request.get_json()
            if not request_data:
                return create_error_response("请求数据格式错误", "validation_error", 400)
            
            analysis_type = request_data.get('analysis_type', 'behavior')  # behavior, growth, retention, vip
            time_range = int(request_data.get('time_range', 30))
            include_vip_analysis = request_data.get('include_vip_analysis', True)
            segment_users = request_data.get('segment_users', False)
            
            # 参数验证
            valid_analysis_types = ['behavior', 'growth', 'retention', 'vip', 'comprehensive']
            if analysis_type not in valid_analysis_types:
                return create_error_response(
                    f"不支持的分析类型: {analysis_type}，支持: {valid_analysis_types}",
                    "validation_error", 400
                )
            
            if not (1 <= time_range <= 365):
                return create_error_response(
                    "时间范围必须在1-365天之间",
                    "validation_error", 400
                )
            
            logger.info(f"👥 用户行为分析: {analysis_type} - {time_range}天")
            
            # 检查编排器状态
            if not orchestrator.initialized:
                return create_error_response("用户分析系统未就绪", "system_unavailable", 503)
            
            start_time = time.time()
            analysis_results = []
            
            # 根据分析类型执行相应分析
            if analysis_type in ['behavior', 'comprehensive']:
                # 用户行为分析
                behavior_result = await orchestrator.data_analyzer.analyze_business_performance('user', time_range)
                analysis_results.append(behavior_result)
            
            if analysis_type in ['growth', 'comprehensive']:
                # 用户增长分析
                growth_result = await orchestrator.data_analyzer.analyze_trend('user', 'new_users', time_range)
                analysis_results.append(growth_result)
            
            if analysis_type in ['vip', 'comprehensive'] and include_vip_analysis:
                # VIP用户分析
                vip_result = await orchestrator.data_analyzer.analyze_business_performance('user', time_range)
                analysis_results.append(vip_result)
            
            # 获取用户详细数据进行补充分析
            user_data_result = await orchestrator.api_connector.get_user_data(1)  # 第一页用户数据
            user_daily_result = await orchestrator.api_connector.get_user_daily_data()  # 用户每日数据
            
            # 用户洞察生成
            user_insights = []
            if analysis_results:
                insights, metadata = await orchestrator.insight_generator.generate_comprehensive_insights(
                    analysis_results=analysis_results,
                    user_context=None,
                    focus_areas=['user_behavior', 'user_growth']
                )
                user_insights = insights
            
            processing_time = time.time() - start_time
            
            # 用户统计摘要
            user_summary = {}
            if user_data_result.get('success') and user_daily_result.get('success'):
                user_list = user_data_result['data'].get('用户列表', [])
                daily_data = user_daily_result['data'].get('每日数据', [])
                
                if user_list:
                    total_investment = sum(float(u.get('总投入', 0)) for u in user_list)
                    total_rewards = sum(float(u.get('累计获得奖励金额', 0)) for u in user_list)
                    avg_roi = sum(float(u.get('投报比', 0)) for u in user_list) / len(user_list)
                    
                    user_summary = {
                        'total_users_analyzed': len(user_list),
                        'total_investment': total_investment,
                        'total_rewards': total_rewards,
                        'avg_roi': avg_roi,
                        'high_value_users': len([u for u in user_list if float(u.get('投报比', 0)) > 0.1])
                    }
                
                if daily_data:
                    latest_day = daily_data[-1] if daily_data else {}
                    total_users = sum(latest_day.get(f'vip{i}的人数', 0) for i in range(11))
                    user_summary.update({
                        'total_registered_users': total_users,
                        'vip_distribution': {
                            f'vip{i}': latest_day.get(f'vip{i}的人数', 0) 
                            for i in range(11)
                        }
                    })
            
            # 格式化分析结果
            formatted_results = []
            for result in analysis_results:
                formatted_result = {
                    'analysis_type': result.analysis_type.value if hasattr(result, 'analysis_type') else analysis_type,
                    'confidence_score': result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                    'key_findings': result.key_findings if hasattr(result, 'key_findings') else [],
                    'business_insights': result.business_insights if hasattr(result, 'business_insights') else [],
                    'recommendations': result.recommendations if hasattr(result, 'recommendations') else []
                }
                formatted_results.append(formatted_result)
            
            response_data = {
                'analysis_results': formatted_results,
                'user_insights': [
                    {
                        'insight_id': insight.insight_id if hasattr(insight, 'insight_id') else f'insight_{i}',
                        'title': insight.title if hasattr(insight, 'title') else '',
                        'summary': insight.summary if hasattr(insight, 'summary') else '',
                        'priority': insight.priority.value if hasattr(insight, 'priority') else 'medium',
                        'recommended_actions': insight.recommended_actions if hasattr(insight, 'recommended_actions') else []
                    }
                    for i, insight in enumerate(user_insights)
                ],
                'user_summary': user_summary,
                'analysis_parameters': {
                    'analysis_type': analysis_type,
                    'time_range': time_range,
                    'include_vip_analysis': include_vip_analysis,
                    'segment_users': segment_users
                },
                'processing_metadata': {
                    'processing_time': processing_time,
                    'analyses_performed': len(formatted_results),
                    'insights_generated': len(user_insights),
                    'data_quality': 'high'
                }
            }
            
            logger.info(f"✅ 用户行为分析完成: {len(formatted_results)}个分析结果")
            
            return create_success_response(response_data, "用户行为分析完成")
        
        except ValueError as e:
            return create_error_response(str(e), "validation_error", 400)
        except Exception as e:
            logger.error(f"❌ 用户行为分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_response(f"用户分析失败: {str(e)}", "internal_error", 500)
    
    return _analyze_users()


@qa_routes_bp.route('/conversations/<int:user_id>', methods=['GET'])
def get_user_conversations(user_id):
    """💬 获取用户对话 - 使用编排器的对话管理器"""
    try:
        # 参数验证
        if user_id < 1:
            return create_error_response("用户ID必须是正整数", "validation_error", 400)
        
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        if limit < 1 or limit > 100:
            return create_error_response("limit必须在1-100之间", "validation_error", 400)
        
        if offset < 0:
            return create_error_response("offset不能为负数", "validation_error", 400)
        
        logger.info(f"💬 获取用户对话: user_id={user_id}")
        
        # 检查编排器状态
        if not orchestrator.initialized:
            return create_error_response("对话系统未就绪", "system_unavailable", 503)
        
        conversations = orchestrator.conversation_manager.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        # 统计信息
        conversation_stats = {
            'total_conversations': len(conversations),
            'active_conversations': len([c for c in conversations if c.get('status') == 'active']),
            'avg_messages_per_conversation': sum(c.get('message_count', 0) for c in conversations) / len(conversations) if conversations else 0
        }
        
        response_data = {
            'conversations': conversations,
            'conversation_stats': conversation_stats,
            'pagination': {
                'user_id': user_id,
                'limit': limit,
                'offset': offset,
                'returned_count': len(conversations)
            }
        }
        
        return create_success_response(response_data, "用户对话获取成功")

    except Exception as e:
        logger.error(f"❌ 获取用户对话失败: {str(e)}")
        return create_error_response(f"对话获取失败: {str(e)}", "internal_error", 500)


@qa_routes_bp.route('/system/health', methods=['GET'])
def system_health_check():
    """🔍 系统健康检查 - 使用编排器的健康检查"""

    @async_wrapper
    async def _health_check():
        try:
            logger.info("🔍 执行QA系统健康检查")
            
            # 编排器健康检查
            health_status = await orchestrator.health_check()
            
            # 获取系统统计
            orchestrator_stats = orchestrator.get_orchestrator_stats()
            
            # 各组件状态检查
            component_health = {
                'query_parser': 'healthy' if hasattr(orchestrator, 'query_parser') else 'unavailable',
                'data_analyzer': 'healthy' if hasattr(orchestrator, 'data_analyzer') else 'unavailable',
                'insight_generator': 'healthy' if hasattr(orchestrator, 'insight_generator') else 'unavailable',
                'conversation_manager': 'healthy' if hasattr(orchestrator, 'conversation_manager') else 'unavailable',
                'api_connector': 'healthy' if hasattr(orchestrator, 'api_connector') else 'unavailable'
            }
            
            # AI模型状态
            ai_models_status = {
                'claude_available': orchestrator.claude_client is not None,
                'gpt_available': orchestrator.gpt_client is not None,
                'dual_ai_collaboration': orchestrator.claude_client is not None and orchestrator.gpt_client is not None
            }
            
            # 计算整体健康状态
            healthy_components = sum(1 for status in component_health.values() if status == 'healthy')
            total_components = len(component_health)
            health_score = healthy_components / total_components
            
            overall_status = 'healthy' if health_score >= 0.8 else 'degraded' if health_score >= 0.5 else 'unhealthy'
            
            response_data = {
                'overall_status': overall_status,
                'health_score': health_score,
                'orchestrator_health': health_status,
                'component_health': component_health,
                'ai_models_status': ai_models_status,
                'system_stats': orchestrator_stats,
                'service_info': {
                    'service_name': 'intelligent_qa_service',
                    'version': '2.0.0',
                    'components_count': total_components,
                    'healthy_components': healthy_components,
                    'ai_models': ['claude-sonnet-4', 'gpt-4o'],
                    'capabilities': [
                        'intelligent_query_processing',
                        'complex_data_analysis',
                        'business_insight_generation',
                        'conversation_management',
                        'multi_source_data_integration',
                        'real_time_analysis',
                        'predictive_analytics',
                        'risk_assessment'
                    ]
                },
                'performance_indicators': {
                    'avg_query_processing_time': health_status.get('avg_processing_time', 0),
                    'success_rate': health_status.get('success_rate', 0),
                    'ai_collaboration_usage': health_status.get('ai_collaboration_rate', 0)
                }
            }
            
            status_code = 200 if overall_status == 'healthy' else 503
            
            return jsonify(response_data), status_code
        
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {str(e)}")
            return create_error_response("健康检查失败", "health_check_error", 500)

    return _health_check()


@qa_routes_bp.route('/charts', methods=['POST'])
def generate_intelligent_charts():
    """🎨 智能图表生成 - 使用编排器的图表生成器"""
    try:
        request_data = request.get_json()
        if not request_data:
            return create_error_response("请求数据格式错误", "validation_error", 400)
        
        # 参数验证
        data = request_data.get('data', {})
        chart_type = request_data.get('chart_type', 'auto')  # auto, line, bar, pie, scatter
        config = request_data.get('config', {})
        preferences = request_data.get('preferences', {})
        
        if not data:
            return create_error_response("图表数据不能为空", "validation_error", 400)
        
        logger.info(f"🎨 生成智能图表: {chart_type}")
        
        # 检查编排器状态
        if not orchestrator.initialized:
            return create_error_response("图表生成系统未就绪", "system_unavailable", 503)
        
        # 使用编排器的图表生成器
        chart_result = orchestrator.chart_generator.generate_chart(
            data=data,
            config=config,
            preferences=preferences
        )
        
        # 增强图表结果
        if chart_result.get('success'):
            chart_result['generation_metadata'] = {
                'chart_type_used': chart_result.get('chart_type', chart_type),
                'data_points': len(data.get('values', [])) if isinstance(data, dict) else len(data),
                'ai_optimized': True,
                'generation_time': datetime.now().isoformat()
            }
        
        return create_success_response(chart_result, "图表生成完成")

    except Exception as e:
        logger.error(f"❌ 图表生成失败: {str(e)}")
        return create_error_response(f"图表生成失败: {str(e)}", "internal_error", 500)


@qa_routes_bp.route('/stats', methods=['GET'])
def get_qa_service_stats():
    """📊 获取QA服务统计信息"""
    
    @async_wrapper
    async def _get_stats():
        try:
            # 编排器统计
            orchestrator_stats = orchestrator.get_orchestrator_stats()
            
            # 查询解析器统计
            parser_stats = {}
            if hasattr(orchestrator, 'query_parser'):
                parser_stats = orchestrator.query_parser.get_processing_stats()
            
            # 数据分析器统计
            analyzer_stats = {}
            if hasattr(orchestrator, 'data_analyzer'):
                analyzer_stats = orchestrator.data_analyzer.get_analysis_stats()
            
            # 洞察生成器统计
            insight_stats = {}
            if hasattr(orchestrator, 'insight_generator'):
                insight_stats = orchestrator.insight_generator.get_insight_stats()
            
            # 计算综合指标
            total_queries = parser_stats.get('total_queries', 0)
            total_analyses = analyzer_stats.get('total_analyses', 0)
            total_insights = insight_stats.get('total_insights_generated', 0)
            
            response_data = {
                'service_overview': {
                    'total_queries_processed': total_queries,
                    'total_analyses_performed': total_analyses,
                    'total_insights_generated': total_insights,
                    'avg_confidence_score': (
                        parser_stats.get('average_confidence', 0) +
                        analyzer_stats.get('avg_confidence_score', 0) +
                        insight_stats.get('avg_confidence_score', 0)
                    ) / 3,
                    'service_uptime': orchestrator_stats.get('uptime', 0)
                },
                'component_stats': {
                    'orchestrator': orchestrator_stats,
                    'query_parser': parser_stats,
                    'data_analyzer': analyzer_stats,
                    'insight_generator': insight_stats
                },
                'performance_metrics': {
                    'ai_collaboration_rate': parser_stats.get('ai_collaboration_usage', 0) / max(total_queries, 1) * 100,
                    'critical_insights_rate': insight_stats.get('critical_insights', 0) / max(total_insights, 1) * 100,
                    'actionable_insights_rate': insight_stats.get('actionable_insights', 0) / max(total_insights, 1) * 100
                }
            }
            
            return create_success_response(response_data, "QA服务统计信息获取成功")
        
        except Exception as e:
            logger.error(f"❌ 获取QA统计失败: {str(e)}")
            return create_error_response(f"统计信息获取失败: {str(e)}", "internal_error", 500)
    
    return _get_stats()


# ============= 错误处理 =============

@qa_routes_bp.errorhandler(404)
def qa_not_found(error):
    """404错误处理"""
    return create_error_response("QA服务端点不存在", "not_found", 404)


@qa_routes_bp.errorhandler(500)
def qa_internal_error(error):
    """500错误处理"""
    logger.error(f"QA服务内部错误: {str(error)}")
    return create_error_response("QA服务内部错误", "internal_error", 500)


@qa_routes_bp.errorhandler(400)
def qa_bad_request(error):
    """400错误处理"""
    return create_error_response("请求参数错误", "bad_request", 400)