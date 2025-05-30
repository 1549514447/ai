# api/qa_routes.py - 完整版本
from flask import Blueprint, jsonify, request
from core.orchestrator.intelligent_qa_orchestrator import get_orchestrator, ProcessingResult
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import time
import uuid  # 用于生成唯一ID

logger = logging.getLogger(__name__)
qa_routes_bp = Blueprint('qa_routes', __name__, url_prefix='/api/qa')

# 🎯 使用单一编排器实例
orchestrator = get_orchestrator()


# ============= 工具函数 =============

def async_route(f):
    """异步路由装饰器，用于在Flask中正确运行async函数。"""

    def wrapper(*args, **kwargs):
        # 为每个请求创建一个新的事件循环，以避免在某些环境中（如某些WSGI服务器）出现问题
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()

    wrapper.__name__ = f.__name__  # 保留原函数名，有助于调试
    return wrapper


def validate_query_request_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证核心问答请求 (/ask) 的数据。
    确保请求包含必要的字段，并且格式正确。
    """
    if not data:
        raise ValueError("请求数据不能为空。请在请求体中提供JSON数据。")

    query = data.get('query', '').strip()
    if not query:
        raise ValueError("查询内容 (query) 不能为空。")

    if len(query) > 2000:  # 稍微放宽查询长度限制
        raise ValueError("查询内容过长，请控制在2000字符以内。")

    # 用户ID可以是字符串或数字，统一处理为整数
    user_id_raw = data.get('user_id')
    if user_id_raw is None:  # 允许匿名用户，但需要明确处理
        user_id = 0  # 或其他代表匿名用户的标识
        logger.info("未提供user_id，将作为匿名用户处理。")
    elif isinstance(user_id_raw, (str, int)):
        try:
            user_id = int(user_id_raw)
            if user_id < 0:  # 允许0代表匿名，但不能为负数
                raise ValueError("用户ID (user_id) 不能为负数。")
        except ValueError:
            raise ValueError("用户ID (user_id) 必须是有效的整数。")
    else:
        raise ValueError("用户ID (user_id) 类型无效。")

    conversation_id = data.get('conversation_id')
    if conversation_id is not None:  # conversation_id 是可选的
        if not isinstance(conversation_id, str) or not (30 < len(conversation_id) < 40):  # 典型的UUID长度
            # 可以使用更宽松的校验，或者在创建时就使用UUID
            try:
                uuid.UUID(conversation_id)  # 尝试将字符串转换为UUID对象以验证格式
            except ValueError:
                raise ValueError("提供的对话ID (conversation_id) 格式无效。")

    preferences = data.get('preferences', {})
    if not isinstance(preferences, dict):
        raise ValueError("偏好设置 (preferences) 必须是一个JSON对象。")

    context_override = data.get('context_override', {})
    if not isinstance(context_override, dict):
        raise ValueError("上下文覆盖 (context_override) 必须是一个JSON对象。")

    return {
        'query': query,
        'user_id': user_id,  # 注意：如果允许匿名，后续逻辑需要能处理 user_id 为 0 的情况
        'conversation_id': conversation_id,
        'preferences': preferences,
        'context_override': context_override
    }


def create_api_success_response(data: Dict[str, Any], message: str = "操作成功", status_code: int = 200) -> tuple:
    """创建统一的API成功响应"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'service_version': orchestrator.config.get('version', '2.0.0') if orchestrator and hasattr(orchestrator,
                                                                                                   'config') else '2.0.0'
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
    # 在开发模式下可以包含更详细的错误信息
    if orchestrator and hasattr(orchestrator, 'config') and orchestrator.config.get('DEBUG', False) and 'details' in \
            error_payload['error'] and isinstance(details, dict):
        error_payload['error']['details']['trace'] = traceback.format_exc()

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
        "user_id": 123, // 用户ID (整数, >0) 或 0/null 代表匿名
        "conversation_id": "optional_conversation_uuid_string", // 可选，用于跟踪对话上下文
        "preferences": { // 可选的用户偏好
            "response_format": "detailed", // "summary", "detailed", "bullet_points"
            "include_charts": true, // 是否包含图表数据
            "analysis_depth": "comprehensive" // "quick", "standard", "comprehensive"
        },
        "context_override": {} // 可选，用于高级用户或调试，以覆盖部分自动获取的上下文
    }
    """
    request_id = str(uuid.uuid4())  # 为每个请求生成唯一ID
    logger.info(f"🚀 RequestID: {request_id} - API: /ask - 收到智能问答请求...")

    try:
        # 确保编排器已初始化
        if not orchestrator.initialized:
            logger.info(f"RequestID: {request_id} - Orchestrator未初始化，尝试初始化...")
            await orchestrator.initialize()  # 这是异步方法
            if not orchestrator.initialized:
                logger.error(f"❌ RequestID: {request_id} - API: 智能问答系统初始化失败，服务不可用。")
                return create_api_error_response("智能问答系统暂时不可用，请稍后重试。", "system_unavailable", 503)
            logger.info(f"RequestID: {request_id} - ✅ Orchestrator初始化成功。")

        request_json = request.get_json()
        validated_data = validate_query_request_data(request_json)

        logger.info(
            f"RequestID: {request_id} - 🧠 API: 接收到智能问答请求: UserID={validated_data['user_id']}, ConversationID='{validated_data['conversation_id']}', Query='{validated_data['query'][:100]}...'")

        start_time = time.time()

        # 调用编排器的核心处理方法
        # 注意: orchestrator.process_intelligent_query 现在也需要是 async
        processing_result: ProcessingResult = await orchestrator.process_intelligent_query(
            user_query=validated_data['query'],
            user_id=validated_data['user_id'],  # 传递整数
            conversation_id=validated_data['conversation_id'],
            preferences=validated_data['preferences']
            # context_override 可以在 orchestrator.process_intelligent_query 内部处理
        )

        end_time = time.time()
        total_route_processing_time = end_time - start_time
        logger.info(
            f"RequestID: {request_id} - ⏱️ API路由层处理耗时: {total_route_processing_time:.2f}s (编排器耗时: {processing_result.total_processing_time:.2f}s)")

        if not processing_result.success:
            error_details = processing_result.error_info or {}
            logger.warning(
                f"⚠️ RequestID: {request_id} - API: 查询处理未成功: Query='{validated_data['query'][:100]}...', ErrorType: {error_details.get('error_type')}, Message: {error_details.get('error_message')}")
            return create_api_error_response(
                error_details.get('error_message', '查询处理失败，请稍后再试。'),
                error_details.get('error_type', 'processing_error'),
                500,  # 或根据error_type调整status_code
                error_details
            )

        # 构建成功的响应体
        response_payload = {
            'request_id': request_id,
            'query_id': processing_result.query_id,
            'session_id': processing_result.session_id,
            'conversation_id': processing_result.conversation_id,
            'answer': processing_result.response_text,
            'key_metrics': processing_result.key_metrics,
            'insights': processing_result.insights,
            'visualizations': processing_result.visualizations,  # 图表数据或配置
            'confidence': {
                'overall_score': round(processing_result.confidence_score, 3),
                'data_quality_score': round(processing_result.data_quality_score, 3),
                'response_completeness': round(processing_result.response_completeness, 3)
            },
            'processing_details': {
                'strategy_used': processing_result.processing_strategy.value,  # 使用枚举值
                'total_processing_time_orchestrator': f"{processing_result.total_processing_time:.3f}s",
                'total_processing_time_route': f"{total_route_processing_time:.3f}s",
                'ai_processing_time': f"{processing_result.ai_processing_time:.3f}s",
                'data_fetching_time': f"{processing_result.data_fetching_time:.3f}s",
                'processors_involved': processing_result.processors_used,
                'ai_collaboration_summary': processing_result.ai_collaboration_summary,
                'metadata': processing_result.processing_metadata
            }
        }

        logger.info(
            f"RequestID: {request_id} - ✅ API: 智能问答处理成功: QueryID={processing_result.query_id}, Confidence={processing_result.confidence_score:.2f}")
        return create_api_success_response(response_payload, "智能问答处理成功")

    except ValueError as ve:  # 参数验证错误
        logger.warning(f"🚫 RequestID: {request_id} - API: 请求参数验证失败: {str(ve)}")
        return create_api_error_response(str(ve), "validation_error", 400, {"request_id": request_id})
    except Exception as e:
        logger.error(f"❌ RequestID: {request_id} - API: 智能问答处理发生意外错误: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"系统内部错误，请稍后重试。", "internal_server_error", 500,
                                         {"request_id": request_id})


# ============= 对话管理API =============

@qa_routes_bp.route('/conversations', methods=['POST'])
@async_route
async def create_new_conversation():
    """
    💬 创建一个新的对话会话。
    请求体 (JSON):
    {
        "user_id": 123, // 用户ID (整数, >=0, 0代表匿名)
        "title": "7月资金规划与风险评估", // 可选, 如果不提供，AI可自动生成或使用默认标题
        "initial_context": {} // 可选, 对话的初始上下文信息
    }
    """
    request_id = str(uuid.uuid4())
    logger.info(f"💬 RequestID: {request_id} - API: /conversations - 收到创建新对话请求...")
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        request_json = request.get_json()
        if not request_json:
            return create_api_error_response("请求数据不能为空。", "validation_error", 400, {"request_id": request_id})

        user_id_raw = request_json.get('user_id')
        if user_id_raw is None:
            return create_api_error_response("用户ID (user_id) 不能为空。", "validation_error", 400,
                                             {"request_id": request_id})
        try:
            user_id = int(user_id_raw)
            if user_id < 0:
                raise ValueError("用户ID不能为负数。")
        except ValueError:
            return create_api_error_response("用户ID (user_id) 必须是有效的非负整数。", "validation_error", 400,
                                             {"request_id": request_id})

        title = request_json.get('title',
                                 f"用户{user_id}的新对话 - {datetime.now().strftime('%Y-%m-%d %H:%M')}").strip()
        if not title:  # 如果用户提供空标题，则使用默认
            title = f"用户{user_id}的对话 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if len(title) > 255:
            title = title[:252] + "..."  # 限制标题长度

        initial_context = request_json.get('initial_context', {})
        if not isinstance(initial_context, dict):
            return create_api_error_response("初始上下文 (initial_context) 必须是一个JSON对象。", "validation_error",
                                             400, {"request_id": request_id})

        # 使用编排器内建的 conversation_manager
        # ConversationManager 的 create_conversation 方法现在接收字符串类型的 conversation_id
        new_conversation_id_int = orchestrator.conversation_manager.create_conversation(  # 此方法返回 int
            title=title,
            user_id=user_id,
            initial_context=initial_context
        )

        # 通常对话ID会是UUID字符串，如果create_conversation返回的是数字ID，需要调整
        # 假设 ConversationManager.create_conversation 返回的是数据库自增ID (int)
        # 如果需要UUID，可以在这里生成或由ConversationManager内部生成并返回
        new_conversation_id_str = str(new_conversation_id_int)  # 如果只是简单转为字符串

        logger.info(f"RequestID: {request_id} - 💬 API: 为用户 {user_id} 创建新对话成功: ID={new_conversation_id_str}")
        return create_api_success_response(
            {'conversation_id': new_conversation_id_str, 'title': title},  # 返回字符串ID
            "新对话创建成功"
        )
    except ValueError as ve:
        logger.warning(f"🚫 RequestID: {request_id} - API: 创建对话参数验证失败: {str(ve)}")
        return create_api_error_response(str(ve), "validation_error", 400, {"request_id": request_id})
    except Exception as e:
        logger.error(f"❌ RequestID: {request_id} - API: 创建新对话失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"创建对话失败: {str(e)}", "internal_server_error", 500,
                                         {"request_id": request_id})


@qa_routes_bp.route('/conversations/<string:conversation_id_str>', methods=['GET'])
@async_route
async def get_conversation_details(conversation_id_str: str):
    """
    📄 获取特定对话的详细历史记录和元数据。
    对话ID应为之前创建时返回的ID。
    """
    request_id = str(uuid.uuid4())
    logger.info(f"📄 RequestID: {request_id} - API: /conversations/{conversation_id_str} - 请求获取对话详情...")
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        # 验证conversation_id_str是否可以转换为整数，因为ConversationManager内部使用int
        try:
            conversation_id_int = int(conversation_id_str)
        except ValueError:
            return create_api_error_response(f"无效的对话ID格式: '{conversation_id_str}'，应为整数。", "validation_error",
                                             400, {"request_id": request_id})

        conversation_data = orchestrator.conversation_manager.get_conversation(conversation_id_int)

        if not conversation_data:
            logger.warning(f"⚠️ RequestID: {request_id} - API: 对话ID '{conversation_id_str}' 未找到。")
            return create_api_error_response(f"对话ID '{conversation_id_str}' 未找到。", "not_found_error", 404,
                                             {"request_id": request_id})

        logger.info(f"RequestID: {request_id} - ✅ API: 获取对话详情成功: ID={conversation_id_str}")
        return create_api_success_response(conversation_data, "对话详情获取成功")

    except Exception as e:
        logger.error(f"❌ RequestID: {request_id} - API: 获取对话详情失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"获取对话详情失败: {str(e)}", "internal_server_error", 500,
                                         {"request_id": request_id})


@qa_routes_bp.route('/conversations/user/<int:user_id>', methods=['GET'])
@async_route
async def get_user_conversation_list(user_id: int):
    """
    📋 获取指定用户的所有对话列表（支持分页）。
    查询参数:
    - limit (int, 可选, 默认20, 范围 1-100): 每页数量。
    - offset (int, 可选, 默认0, >=0): 偏移量。
    """
    request_id = str(uuid.uuid4())
    logger.info(f"📋 RequestID: {request_id} - API: /conversations/user/{user_id} - 请求获取用户对话列表...")
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        if user_id < 0:  # 假设0是匿名用户，允许
            return create_api_error_response("用户ID (user_id) 不能为负数。", "validation_error", 400,
                                             {"request_id": request_id})

        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)

        if not (1 <= limit <= 100):
            return create_api_error_response("参数 'limit' 必须在 1 到 100 之间。", "validation_error", 400,
                                             {"request_id": request_id})
        if offset < 0:
            return create_api_error_response("参数 'offset' 不能为负数。", "validation_error", 400,
                                             {"request_id": request_id})

        conversations = orchestrator.conversation_manager.get_user_conversations(
            user_id=user_id, limit=limit, offset=offset
        )

        # 获取总对话数用于分页 (ConversationManager 需要一个 count_user_conversations 方法)
        # total_conversations = orchestrator.conversation_manager.count_user_conversations(user_id)
        # 模拟一个总数，实际应从DB获取
        # total_conversations = len(conversations) if offset == 0 and len(conversations) < limit else (offset + len(conversations) + (limit if len(conversations) == limit else 0) )
        # 由于无法准确获取总数，暂时不返回 total_count

        logger.info(
            f"RequestID: {request_id} - ✅ API: 获取用户 {user_id} 的对话列表成功 (Limit: {limit}, Offset: {offset}, Returned: {len(conversations)})")
        return create_api_success_response({
            'conversations': conversations,
            'pagination': {
                'user_id': user_id,
                'limit': limit,
                'offset': offset,
                'returned_count': len(conversations),
                # 'total_count': total_conversations # 当实现时取消注释
            }
        }, "用户对话列表获取成功")

    except Exception as e:
        logger.error(f"❌ RequestID: {request_id} - API: 获取用户对话列表失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"获取用户对话列表失败: {str(e)}", "internal_server_error", 500,
                                         {"request_id": request_id})


# ============= 系统状态与辅助API =============

@qa_routes_bp.route('/system/health', methods=['GET'])
@async_route
async def qa_system_health_check():
    """
    🔍 QA服务健康检查 - 调用编排器的全面健康检查。
    """
    request_id = str(uuid.uuid4())
    logger.info(f"🔍 RequestID: {request_id} - API: /system/health - 执行QA系统健康检查...")
    try:
        if not orchestrator.initialized:
            try:
                await orchestrator.initialize()
                if not orchestrator.initialized:
                    raise SystemError("编排器初始化失败，无法执行健康检查。")
            except Exception as init_e:
                logger.error(f"❌ RequestID: {request_id} - API: 健康检查时编排器初始化失败: {init_e}")
                return create_api_error_response(f"系统初始化失败: {init_e}", "system_unavailable", 503,
                                                 {"request_id": request_id, "initialization_error": str(init_e)})

        health_status = await orchestrator.health_check()

        overall_status = health_status.get('status', 'unknown')
        http_status_code = 200 if overall_status == 'healthy' else 503 if overall_status in ['unhealthy',
                                                                                             'initializing'] else 200

        logger.info(f"RequestID: {request_id} - ✅ API: QA系统健康检查完成: Status={overall_status}")
        # 添加 request_id 到响应中
        health_status_with_req_id = {"request_id": request_id, **health_status}
        return jsonify(health_status_with_req_id), http_status_code

    except Exception as e:
        logger.error(f"❌ RequestID: {request_id} - API: QA系统健康检查发生意外错误: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"健康检查失败: {str(e)}", "internal_server_error", 500,
                                         {"request_id": request_id})


@qa_routes_bp.route('/system/stats', methods=['GET'])
@async_route
async def get_qa_system_statistics():
    """
    📊 获取QA系统各项服务的统计信息。
    """
    request_id = str(uuid.uuid4())
    logger.info(f"📊 RequestID: {request_id} - API: /system/stats - 请求获取QA系统统计信息...")
    try:
        if not orchestrator.initialized: await orchestrator.initialize()

        orchestrator_stats = orchestrator.get_orchestrator_stats()

        # 动态获取各组件统计信息，如果组件和其统计方法存在
        module_stats = {}
        component_map = {
            'query_parser': 'query_parser',
            'financial_data_analyzer': 'financial_data_analyzer',
            'insight_generator': 'insight_generator',
            'smart_data_fetcher': 'data_fetcher'  # orchestrator 中 data_fetcher 才是 SmartDataFetcher 实例
        }

        for key, attr_name in component_map.items():
            component = getattr(orchestrator, attr_name, None)
            if component and hasattr(component, 'get_processing_stats'):  # SmartQueryParser
                module_stats[key] = component.get_processing_stats()
            elif component and hasattr(component, 'get_analysis_stats'):  # FinancialDataAnalyzer
                module_stats[key] = component.get_analysis_stats()
            elif component and hasattr(component, 'get_insight_stats'):  # InsightGenerator
                module_stats[key] = component.get_insight_stats()
            elif component and hasattr(component, 'get_performance_stats'):  # SmartDataFetcher
                module_stats[key] = component.get_performance_stats()
            else:
                module_stats[key] = {"status": "unavailable or no stats method"}

        response_data = {
            'request_id': request_id,
            'overall_orchestrator_stats': orchestrator_stats,
            'module_stats': module_stats,
            'service_uptime_seconds': orchestrator_stats.get('uptime', 0),
            'cache_effectiveness': {
                'parsing_cache_hit_rate': round(orchestrator_stats.get('cache_hit_rate', 0), 3),
            }
        }

        logger.info(f"RequestID: {request_id} - ✅ API: QA系统统计信息获取成功。")
        return create_api_success_response(response_data, "QA系统统计获取成功")

    except Exception as e:
        logger.error(f"❌ RequestID: {request_id} - API: 获取QA系统统计失败: {str(e)}\n{traceback.format_exc()}")
        return create_api_error_response(f"获取统计信息失败: {str(e)}", "internal_server_error", 500,
                                         {"request_id": request_id})


# ============= 蓝图级别的错误处理 =============
# 这些处理器会捕获在蓝图内未被特定try-except块处理的异常

@qa_routes_bp.app_errorhandler(ValueError)  # 通常是参数验证错误
def handle_qa_value_error(error: ValueError):
    request_id = str(uuid.uuid4())  # 为错误响应也生成ID
    logger.warning(
        f"🚫 RequestID: {request_id} - QA API: ValueError (参数验证错误) - Path: {request.path} - Error: {str(error)}")
    return create_api_error_response(str(error), "validation_error", 400, {"request_id": request_id})


@qa_routes_bp.app_errorhandler(400)  # Flask自动处理的Bad Request
def handle_qa_bad_request(error):
    request_id = str(uuid.uuid4())
    message = error.description if hasattr(error, 'description') and error.description else "错误的请求格式或参数。"
    logger.warning(f"🚫 RequestID: {request_id} - QA API: 400 Bad Request - Path: {request.path} - Message: {message}")
    return create_api_error_response(message, "bad_request_error", 400, {"request_id": request_id})


@qa_routes_bp.app_errorhandler(404)
def handle_qa_not_found(error):
    request_id = str(uuid.uuid4())
    logger.warning(f"🚫 RequestID: {request_id} - QA API: 404 Not Found - Path: {request.path}")
    return create_api_error_response(f"请求的QA资源 '{request.path}' 未找到。", "not_found_error", 404,
                                     {"request_id": request_id})


@qa_routes_bp.app_errorhandler(500)
def handle_qa_internal_error(error):
    request_id = str(uuid.uuid4())
    logger.error(
        f"❌ RequestID: {request_id} - QA API: 500 Internal Server Error - Path: {request.path} - Error: {str(error)}\n{traceback.format_exc()}")
    err_msg = str(error) if orchestrator and hasattr(orchestrator, 'config') and orchestrator.config.get("DEBUG",
                                                                                                         False) else "服务器内部发生错误，我们正在紧急处理。"
    details = {"trace": traceback.format_exc()} if orchestrator and hasattr(orchestrator,
                                                                            'config') and orchestrator.config.get(
        "DEBUG", False) else None
    details_with_req_id = {**(details or {}), "request_id": request_id}
    return create_api_error_response(err_msg, "internal_server_error", 500, details_with_req_id)