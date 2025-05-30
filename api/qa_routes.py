# api/qa_routes.py - 完整优化版本
from flask import Blueprint, jsonify, request
from core.orchestrator.intelligent_qa_orchestrator import get_orchestrator, ProcessingResult  # 导入ProcessingResult
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


def validate_query_request_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """验证核心问答请求数据"""
    if not data:
        raise ValueError("请求数据不能为空。请在请求体中提供JSON数据。")

    query = data.get('query', '').strip()
    if not query:
        raise ValueError("查询内容 (query) 不能为空。")

    if len(query) > 1000:  # 限制查询长度
        raise ValueError("查询内容过长，请控制在1000字符以内。")

    user_id_str = str(data.get('user_id', '1'))  # 用户ID可以是字符串或数字，统一处理
    if not user_id_str.isdigit() or int(user_id_str) < 1:
        raise ValueError("用户ID (user_id) 必须是正整数。")
    user_id = int(user_id_str)

    conversation_id = data.get('conversation_id')
    if conversation_id and (not isinstance(conversation_id, str) or len(conversation_id) > 50):
        raise ValueError("对话ID (conversation_id) 无效。")

    preferences = data.get('preferences', {})
    if not isinstance(preferences, dict):
        raise ValueError("偏好设置 (preferences) 必须是一个JSON对象。")

    return {
        'query': query,
        'user_id': user_id,
        'conversation_id': conversation_id,
        'preferences': preferences,
        'context_override': data.get('context_override', {})  # 允许前端覆盖部分上下文，高级功能
    }


def create_api_success_response(data: Dict[str, Any], message: str = "操作成功", status_code: int = 200) -> tuple:
    """创建统一的API成功响应"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'service_version': '2.0.0'  # 示例版本号
    }), status_code


def create_api_error_response(message: str, error_type: str = "processing_error",
                              status_code: int = 500, details: Optional[Dict[str, Any]] = None) -> tuple:
    """创建统一的API错误响应"""
    error_payload = {
        'success': False,
        'error': {
            'type': error_type,
            'message': message,
            'details': details or {}
        },
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(error_payload), status_code


# ============= 核心问答API =============

@qa_routes_bp.route('/ask', methods=['POST'])
@async_route
async def intelligent_Youtubeing():
    """
    🧠 智能问答主接口 - 接收用户自然语言查询并返回分析结果和洞察。
    请求体 (JSON):
    {
        "query": "用户的自然语言查询",
        "user_id": 123, // 用户ID
        "conversation_id": "optional_conversation_uuid", // 可选，用于跟踪对话上下文
        "preferences": { // 可选的用户偏好
            "response_format": "detailed" // "summary", "detailed", "bullet_points"
            "include_charts": true,
            "analysis_depth": "comprehensive" // "quick", "standard", "comprehensive"
        },
        "context_override": {} // 可选，用于高级用户或调试，以覆盖部分自动获取的上下文
    }
    """
    try:
        # 确保编排器已初始化
        if not orchestrator.initialized:
            logger.info("Orchestrator not initialized, attempting to initialize now...")
            await orchestrator.initialize()
            if not orchestrator.initialized:
                logger.error("❌ API: 智能问答系统初始化失败，服务不可用。")
                return create_api_error_response("智能问答系统暂时不可用，请稍后重试。", "system_unavailable", 503)
            logger.info("✅ Orchestrator initialized successfully for /ask route.")

        request_json = request.get_json()
        validated_data = validate_query_request_data(request_json)

        logger.info(
            f"🧠 API: 接收到智能问答请求: UserID={validated_data['user_id']}, Query='{validated_data['query'][:50]}...'")

        start_time = time.time()

        # 调用编排器的核心处理方法
        processing_result: ProcessingResult = await orchestrator.process_intelligent_query(
            user_query=validated_data['query'],
            user_id=validated_data['user_id'],
            conversation_id=validated_data['conversation_id'],
            preferences=validated_data['preferences']
            # context_override 可以在 orchestrator.process_intelligent_query 内部处理
        )

        end_time = time.time()
        total_route_processing_time = end_time - start_time

        if not processing_result.success:
            logger.warning(
                f"⚠️ API: 查询处理未成功: Query='{validated_data['query'][:50]}...', Error: {processing_result.error_info}")
            return create_api_error_response(
                processing_result.error_info.get('error_message', '查询处理失败，请稍后再试。'),
                processing_result.error_info.get('error_type', 'processing_error'),
                500,  # 或者根据error_type调整status_code
                processing_result.error_info
            )

        # 构建成功的响应体
        response_payload = {
            'query_id': processing_result.query_id,
            'session_id': processing_result.session_id,
            'conversation_id': processing_result.conversation_id,
            'answer': processing_result.response_text,
            'key_metrics': processing_result.key_metrics,
            'insights': processing_result.insights,
            'visualizations': processing_result.visualizations,  # 图表数据或配置
            'confidence': {
                'overall_score': processing_result.confidence_score,
                'data_quality_score': processing_result.data_quality_score,
                'response_completeness': processing_result.response_completeness
            },
            'processing_details': {
                'strategy_used': processing_result.processing_strategy.value,
                'total_processing_time_orchestrator': f"{processing_result.total_processing_time:.2f}s",
                'total_processing_time_route': f"{total_route_processing_time:.2f}s",
                'ai_processing_time': f"{processing_result.ai_processing_time:.2f}s",
                'data_fetching_time': f"{processing_result.data_fetching_time:.2f}s",
                'processors_involved': processing_result.processors_used,
                'ai_collaboration': processing_result.ai_collaboration_summary,
                'metadata': processing_result.processing_metadata
            }
        }

        logger.info(
            f"✅ API: 智能问答处理成功: QueryID={processing_result.query_id}, Confidence={processing_result.confidence_score:.2f}")
        return create_api_success_response(response_payload, "智能问答处理成功")

    except ValueError as ve:  # 参数验证错误
        logger.warning(f"🚫 API: 请求参数验证失败: {str(ve)}")
        return create_api_error_response(str(ve), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 智能问答处理发生意外错误: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"系统内部错误，请稍后重试: {str(e)}", "internal_server_error", 500)


# ============= 对话管理API =============

@qa_routes_bp.route('/conversations', methods=['POST'])
@async_route
async def create_new_conversation():
    """
    💬 创建一个新的对话会话
    请求体 (JSON):
    {
        "user_id": 123,
        "title": "7月资金规划与风险评估", // 可选, AI可自动生成
        "initial_context": {} // 可选, 对话的初始上下文
    }
    """
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        request_json = request.get_json()
        if not request_json:
            return create_api_error_response("请求数据不能为空", "validation_error", 400)

        user_id_str = str(request_json.get('user_id'))
        if not user_id_str.isdigit() or int(user_id_str) < 1:
            return create_api_error_response("用户ID (user_id) 必须是正整数。", "validation_error", 400)
        user_id = int(user_id_str)

        title = request_json.get('title', f"用户{user_id}的对话 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        initial_context = request_json.get('initial_context', {})

        # 通过编排器（间接）或直接使用 ConversationManager 创建对话
        # 假设编排器暴露了对话管理接口，或者 ConversationManager 是全局可访问的
        # 这里我们用编排器内建的 conversation_manager
        conversation_id = orchestrator.conversation_manager.create_conversation(
            title=title,
            user_id=user_id,
            initial_context=initial_context
        )

        logger.info(f"💬 API: 为用户 {user_id} 创建新对话成功: ID={conversation_id}")
        return create_api_success_response(
            {'conversation_id': conversation_id, 'title': title},
            "新对话创建成功"
        )
    except ValueError as ve:
        logger.warning(f"🚫 API: 创建对话参数验证失败: {str(ve)}")
        return create_api_error_response(str(ve), "validation_error", 400)
    except Exception as e:
        logger.error(f"❌ API: 创建新对话失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"创建对话失败: {str(e)}", "internal_server_error", 500)


@qa_routes_bp.route('/conversations/<string:conversation_id>', methods=['GET'])
@async_route
async def get_conversation_details(conversation_id: str):
    """
    📄 获取特定对话的详细历史记录和元数据
    """
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        if not conversation_id:
            return create_api_error_response("对话ID不能为空。", "validation_error", 400)

        # 从 ConversationManager 获取对话详情
        conversation_data = orchestrator.conversation_manager.get_conversation(
            conversation_id)  # get_conversation 现在应该接收 str

        if not conversation_data:
            return create_api_error_response(f"对话ID '{conversation_id}' 未找到。", "not_found_error", 404)

        logger.info(f"📄 API: 获取对话详情成功: ID={conversation_id}")
        return create_api_success_response(conversation_data, "对话详情获取成功")

    except Exception as e:
        logger.error(f"❌ API: 获取对话详情失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"获取对话详情失败: {str(e)}", "internal_server_error", 500)


@qa_routes_bp.route('/conversations/user/<int:user_id>', methods=['GET'])
@async_route
async def get_user_conversation_list(user_id: int):
    """
    📋 获取指定用户的所有对话列表（分页）
    查询参数:
    - limit (int, 可选, 默认20): 每页数量
    - offset (int, 可选, 默认0): 偏移量
    """
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        if user_id < 1:
            return create_api_error_response("用户ID必须是正整数。", "validation_error", 400)

        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)

        if not (1 <= limit <= 100):
            return create_api_error_response("参数 'limit' 必须在 1 到 100 之间。", "validation_error", 400)
        if offset < 0:
            return create_api_error_response("参数 'offset' 不能为负数。", "validation_error", 400)

        conversations = orchestrator.conversation_manager.get_user_conversations(
            user_id=user_id, limit=limit, offset=offset
        )

        # 通常还需要一个总数，用于前端分页，ConversationManager 可能需要一个 count_user_conversations 方法
        # total_conversations = orchestrator.conversation_manager.count_user_conversations(user_id)

        logger.info(f"📋 API: 获取用户 {user_id} 的对话列表成功 (Limit: {limit}, Offset: {offset})")
        return create_api_success_response({
            'conversations': conversations,
            'pagination': {
                'user_id': user_id,
                'limit': limit,
                'offset': offset,
                'returned_count': len(conversations),
                # 'total_count': total_conversations # 如果实现了计数方法
            }
        }, "用户对话列表获取成功")

    except Exception as e:
        logger.error(f"❌ API: 获取用户对话列表失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"获取用户对话列表失败: {str(e)}", "internal_server_error", 500)


# ============= 系统状态与辅助API =============

@qa_routes_bp.route('/system/health', methods=['GET'])
@async_route
async def qa_system_health_check():
    """
    🔍 QA服务健康检查 - 复用编排器的全面健康检查
    """
    try:
        logger.info("🔍 API: 执行QA系统健康检查...")
        if not orchestrator.initialized:
            # 尝试在健康检查时初始化，以报告更准确的状态
            try:
                await orchestrator.initialize()
                if not orchestrator.initialized:
                    raise SystemError("编排器初始化失败")
            except Exception as init_e:
                logger.error(f"❌ API: 健康检查时编排器初始化失败: {init_e}")
                return create_api_error_response(f"系统初始化失败: {init_e}", "system_unavailable", 503,
                                                 {"initialization_error": str(init_e)})

        health_status = await orchestrator.health_check()

        # 根据编排器状态决定HTTP状态码
        overall_status = health_status.get('status', 'unknown')
        http_status_code = 200 if overall_status == 'healthy' else 503 if overall_status in ['unhealthy',
                                                                                             'initializing'] else 200  # 200 for degraded/limited

        logger.info(f"✅ API: QA系统健康检查完成: Status={overall_status}")
        # 直接返回编排器的健康检查结果，因为它已经很全面了
        return jsonify(health_status), http_status_code

    except Exception as e:
        logger.error(f"❌ API: QA系统健康检查发生意外错误: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"健康检查失败: {str(e)}", "internal_server_error", 500)


@qa_routes_bp.route('/system/stats', methods=['GET'])
@async_route
async def get_qa_system_statistics():
    """
    📊 获取QA系统各项服务的统计信息
    """
    try:
        logger.info("📊 API: 请求获取QA系统统计信息...")
        if not orchestrator.initialized:
            await orchestrator.initialize()

        orchestrator_stats = orchestrator.get_orchestrator_stats()

        # 可以从编排器获取其管理的各个组件的统计信息
        parser_stats = orchestrator.query_parser.get_processing_stats() if hasattr(orchestrator,
                                                                                   'query_parser') and hasattr(
            orchestrator.query_parser, 'get_processing_stats') else {}
        analyzer_stats = orchestrator.financial_data_analyzer.get_analysis_stats() if hasattr(orchestrator,
                                                                                              'financial_data_analyzer') and hasattr(
            orchestrator.financial_data_analyzer, 'get_analysis_stats') else {}
        insight_gen_stats = orchestrator.insight_generator.get_insight_stats() if hasattr(orchestrator,
                                                                                          'insight_generator') and hasattr(
            orchestrator.insight_generator, 'get_insight_stats') else {}
        data_fetcher_stats = orchestrator.data_fetcher.get_performance_stats() if hasattr(orchestrator,
                                                                                          'data_fetcher') and hasattr(
            orchestrator.data_fetcher, 'get_performance_stats') else {}
        # ... 其他组件的统计

        response_data = {
            'overall_orchestrator_stats': orchestrator_stats,
            'module_stats': {
                'query_parser': parser_stats,
                'financial_data_analyzer': analyzer_stats,
                'insight_generator': insight_gen_stats,
                'smart_data_fetcher': data_fetcher_stats,
            },
            'service_uptime_seconds': orchestrator_stats.get('uptime', 0),  # 假设编排器跟踪了启动时间
            'cache_effectiveness': {
                'parsing_cache_hit_rate': orchestrator_stats.get('cache_hit_rate', 0),  # orchestrator 可能需要更细分的缓存统计
                # 'data_cache_hit_rate': data_fetcher_stats.get('cache_hit_rate',0)
            }
        }

        logger.info("✅ API: QA系统统计信息获取成功。")
        return create_api_success_response(response_data, "QA系统统计获取成功")

    except Exception as e:
        logger.error(f"❌ API: 获取QA系统统计失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"获取统计信息失败: {str(e)}", "internal_server_error", 500)


# ============= 蓝图级别的错误处理 =============
# 这些处理器会捕获在蓝图内未被特定try-except块处理的异常

@qa_routes_bp.app_errorhandler(400)
def handle_qa_bad_request(error):
    message = error.description if hasattr(error, 'description') and error.description else "错误的请求格式或参数。"
    logger.warning(f"🚫 QA API: 400 Bad Request - {request.path} - {message}")
    return create_api_error_response(message, "bad_request", 400)


@qa_routes_bp.app_errorhandler(404)
def handle_qa_not_found(error):
    logger.warning(f"🚫 QA API: 404 Not Found - {request.path}")
    return create_api_error_response(f"请求的QA资源 '{request.path}' 未找到。", "not_found", 404)


@qa_routes_bp.app_errorhandler(500)
def handle_qa_internal_error(error):
    logger.error(f"❌ QA API: 500 Internal Server Error - {request.path} - {str(error)}\n{traceback.format_exc()}")
    # 在生产环境中，避免将详细错误暴露给客户端
    err_msg = str(error) if orchestrator.config.get("DEBUG") else "服务器内部发生错误，请稍后重试。"
    return create_api_error_response(err_msg, "internal_server_error", 500,
                                     {"trace": traceback.format_exc() if orchestrator.config.get("DEBUG") else None})


@qa_routes_bp.app_errorhandler(ValueError)
def handle_qa_value_error(error):  # 主要捕获 validate_query_request_data 抛出的错误
    logger.warning(f"🚫 QA API: ValueError (likely validation error) - {request.path} - {str(error)}")
    return create_api_error_response(str(error), "validation_error", 400)