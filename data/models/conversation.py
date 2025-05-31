# data/models/conversation.py
import json
import logging
from dataclasses import dataclass, field  # field 需要从 dataclasses 导入
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from data.connectors.database_connector import DatabaseConnector  # 确保 DatabaseConnector 被正确导入

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """对话数据模型"""
    id: int
    title: str
    user_id: int
    created_at: datetime
    updated_at: datetime
    status: str = 'active'  # 保持为 str，与您的 Python 代码一致
    initial_context: Optional[Dict[str, Any]] = field(default_factory=dict)  # 确保有默认值


@dataclass
class Message:
    """消息数据模型"""
    id: int
    conversation_id: int
    is_user: bool
    content: str
    created_at: datetime
    visuals: List[Dict[str, Any]] = field(default_factory=list)  # visuals 应为列表，且有默认值

    ai_model_used: Optional[str] = None
    ai_strategy: Optional[str] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None


@dataclass
class Visual:
    """可视化元素数据模型"""
    id: int
    message_id: int
    visual_type: str
    visual_order: int
    title: str
    data: Dict[str, Any]


class ConversationManager:
    """现代化对话管理器 - 支持智能上下文和AI协作"""

    def __init__(self, database_connector: Optional[DatabaseConnector]):  # 允许 DB Connector 为 None
        self.db = database_connector
        if self.db:
            logger.info("ConversationManager initialized with DatabaseConnector.")
        else:
            logger.warning(
                "ConversationManager initialized WITHOUT DatabaseConnector. History will be in-memory or disabled if not implemented.")
            # 如果需要在无DB时有内存存储，可以在这里初始化一个字典
            self._memory_conversations: Dict[int, Dict[str, Any]] = {}
            self._memory_messages: Dict[int, List[Dict[str, Any]]] = {}
            self._next_conv_id = 1200  # 从一个基数开始，避免与DB ID冲突
            self._next_msg_id = 2000

    # ============= 基础CRUD操作 =============

    def create_conversation(self, title: str, user_id: int,
                            initial_context: Optional[Dict[str, Any]] = None) -> int:
        """创建新对话。如果无DB连接，则在内存中创建。"""
        context_json = json.dumps(initial_context or {}, ensure_ascii=False)
        if not self.db:
            conv_id = self._next_conv_id
            self._next_conv_id += 1
            new_conv_data = {
                'id': conv_id, 'title': title, 'user_id': user_id,
                'created_at': datetime.now(), 'updated_at': datetime.now(),
                'status': 1,  # 对应SQL的tinyint
                'initial_context': initial_context or {}  # 直接存字典
            }
            self._memory_conversations[conv_id] = new_conv_data
            self._memory_messages[conv_id] = []  # 初始化消息列表
            logger.info(f"In-memory conversation {conv_id} created for user {user_id}")
            # 添加欢迎消息（内存版）
            welcome_message = self._generate_welcome_message(user_id)
            self.add_message(conv_id, False, welcome_message,
                             ai_model_used="system_memory", ai_strategy="welcome")
            return conv_id

        try:
            query = """
                INSERT INTO conversations (title, user_id, initial_context, status)
                VALUES (%s, %s, %s, %s)
            """
            # SQL status 是 tinyint, 1 代表 'active'
            conversation_id = self.db.execute_insert(query, (title, user_id, context_json, 1))

            welcome_message = self._generate_welcome_message(user_id)
            self.add_message(conversation_id, False, welcome_message,
                             ai_model_used="system_db", ai_strategy="welcome")

            logger.info(f"DB: Created conversation {conversation_id} for user {user_id}")
            return conversation_id
        except Exception as e:
            logger.error(f"Failed to create DB conversation: {str(e)}")
            raise

    def add_message(self, conversation_id: int, is_user: bool, content: str,
                    ai_model_used: Optional[str] = None, ai_strategy: Optional[str] = None,
                    processing_time: Optional[float] = None, confidence_score: Optional[float] = None) -> int:
        """添加消息到对话。如果无DB连接，则添加到内存。"""
        if not self.db:
            if conversation_id not in self._memory_conversations:
                raise ValueError(f"In-memory conversation with ID {conversation_id} not found.")
            msg_id = self._next_msg_id
            self._next_msg_id += 1
            new_msg = {
                'id': msg_id, 'conversation_id': conversation_id, 'is_user': is_user, 'content': content,
                'created_at': datetime.now(), 'ai_model_used': ai_model_used, 'ai_strategy': ai_strategy,
                'processing_time': processing_time, 'confidence_score': confidence_score, 'visuals': []
            }
            self._memory_messages.setdefault(conversation_id, []).append(new_msg)
            self._memory_conversations[conversation_id]['updated_at'] = datetime.now()
            logger.debug(f"Added in-memory message {msg_id} to conversation {conversation_id}")
            return msg_id

        try:
            query = """
                INSERT INTO messages (conversation_id, is_user, content,
                                    ai_model_used, ai_strategy, processing_time, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            message_id = self.db.execute_insert(query, (
                conversation_id, 1 if is_user else 0, content,  # SQL is_user 是 tinyint
                ai_model_used, ai_strategy, processing_time, confidence_score
            ))
            self._update_conversation_timestamp(conversation_id)
            logger.debug(f"DB: Added message {message_id} to conversation {conversation_id}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to add DB message: {str(e)}")
            raise

    # ... (add_visual 方法类似地需要考虑无DB的情况，但由于其依赖messages表的外键，内存版会更复杂，暂时跳过其内存实现)
    def add_visual(self, message_id: int, visual_type: str, visual_order: int,
                   title: str, data: Dict[str, Any]) -> int:
        if not self.db:
            logger.warning("add_visual: No DB connector, visual not persisted for in-memory messages.")
            # 尝试在内存消息中附加可视化数据
            for conv_id, msgs in self._memory_messages.items():
                for msg in msgs:
                    if msg['id'] == message_id:
                        if 'visuals' not in msg or msg['visuals'] is None:  # visuals可能为None
                            msg['visuals'] = []
                        msg['visuals'].append({
                            'id': len(msg['visuals']) + 10000,  # 临时ID
                            'visual_type': visual_type,
                            'visual_order': visual_order,
                            'title': title,
                            'data': data
                        })
                        return msg['visuals'][-1]['id']
            return -1  # 表示失败

        try:
            data_json = json.dumps(data, ensure_ascii=False)
            query = """
                INSERT INTO message_visuals (message_id, visual_type, visual_order, title, data_json)
                VALUES (%s, %s, %s, %s, %s)
            """
            visual_id = self.db.execute_insert(query, (
                message_id, visual_type, visual_order, title, data_json
            ))
            logger.debug(f"DB: Added visual {visual_id} to message {message_id}")
            return visual_id
        except Exception as e:
            logger.error(f"Failed to add DB visual: {str(e)}")
            raise

    def get_conversation(self, conversation_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        获取完整对话内容。如果无DB连接，则从内存获取。
        
        Args:
            conversation_id: 对话ID，可以是整数或字符串
            
        Returns:
            Optional[Dict[str, Any]]: 对话详情，包括对话信息、消息列表和统计数据
        """
        # 确保conversation_id是整数类型
        if isinstance(conversation_id, str):
            try:
                conversation_id = int(conversation_id)
            except ValueError:
                logger.error(f"Invalid conversation_id format: {conversation_id}")
                return None
                
        if not self.db:
            if conversation_id not in self._memory_conversations:
                logger.warning(f"Conversation {conversation_id} not found in memory")
                return None

            conv_info_mem = self._memory_conversations[conversation_id].copy()
            # 内存中的 initial_context 已经是字典了，不需要 json.loads
            # status 是 int，需要转为 dataclass 定义的 str
            conv_info_mem['status'] = 'active' if conv_info_mem.get('status', 1) == 1 else 'archived'  # 示例转换

            messages_mem = self._memory_messages.get(conversation_id, [])
            # 确保消息中的is_user是布尔值
            for msg in messages_mem:
                msg['is_user'] = bool(msg.get('is_user'))

            # 内存版的visuals已经在消息中，不需要单独的 _get_message_visuals
            ai_stats = self._generate_ai_usage_stats(messages_mem)
            return {
                'conversation': conv_info_mem,
                'messages': messages_mem,
                'ai_usage_stats': ai_stats,
                'total_messages': len(messages_mem),
                'user_messages': len([m for m in messages_mem if m['is_user']]),
                'ai_messages': len([m for m in messages_mem if not m['is_user']])
            }

        try:
            conv_query = """
                SELECT id, title, user_id, created_at, updated_at, status, initial_context
                FROM conversations
                WHERE id = %s
            """
            conversation_data_list = self.db.execute_query(conv_query, (conversation_id,))
            if not conversation_data_list: return None

            conversation_db = conversation_data_list[0]
            try:  # initial_context 是 TEXT，可能是 NULL
                conversation_db['initial_context'] = json.loads(conversation_db['initial_context'] or '{}')
            except json.JSONDecodeError:
                logger.warning(f"Corrupted initial_context JSON for conversation_id {conversation_id}")
                conversation_db['initial_context'] = {}

            # SQL status 是 tinyint, 转换为 str
            conversation_db['status'] = 'active' if conversation_db.get('status', 1) == 1 else 'archived'

            msg_query = """
                SELECT id, conversation_id, is_user, content, created_at, ai_model_used,
                       ai_strategy, processing_time, confidence_score
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC
            """
            messages_db = self.db.execute_query(msg_query, (conversation_id,))
            # 转换 is_user 从 tinyint 到 bool
            for msg in messages_db:
                msg['is_user'] = bool(msg.get('is_user'))
                msg['visuals'] = self._get_message_visuals(msg['id'])  # 从DB获取visuals

            ai_stats = self._generate_ai_usage_stats(messages_db)
            return {
                'conversation': conversation_db,
                'messages': messages_db,
                'ai_usage_stats': ai_stats,
                'total_messages': len(messages_db),
                'user_messages': sum(1 for m in messages_db if m['is_user']),
                'ai_messages': sum(1 for m in messages_db if not m['is_user'])
            }
        except Exception as e:
            logger.error(f"Failed to get DB conversation {conversation_id}: {str(e)}")
            return None

    # +++++++++++++ 新增 get_context 方法 +++++++++++++
    def get_context(self, conversation_id: int, history_limit: int = 5) -> Dict[str, Any]:
        """
        获取对话的上下文信息，用于AI分析。

        Args:
            conversation_id: 对话ID (int)
            history_limit: 要包含的最近消息数量

        Returns:
            Dict: 包含对话上下文的字典。如果找不到对话或发生错误，则返回空字典。
        """
        logger.debug(f"Fetching context for conversation ID: {conversation_id} with history limit: {history_limit}")

        conversation_full_details = self.get_conversation(conversation_id)  # 复用现有方法

        if not conversation_full_details or not conversation_full_details.get('conversation'):
            logger.warning(f"Conversation with ID {conversation_id} not found for context retrieval.")
            return {}

        base_conv_info = conversation_full_details['conversation']
        all_messages = conversation_full_details.get('messages', [])

        context = {
            "conversation_id": base_conv_info.get('id'),
            "title": base_conv_info.get('title'),
            "user_id": base_conv_info.get('user_id'),
            "initial_context": base_conv_info.get('initial_context', {}),  # 确保这是字典
            "recent_history": []
        }

        # 获取最近 N 条消息
        # get_conversation 已经按 created_at ASC 排序了
        recent_raw_messages = all_messages[-(history_limit):]

        for msg_dict in recent_raw_messages:
            context["recent_history"].append({
                "role": "user" if msg_dict.get('is_user') else "assistant",
                "content": msg_dict.get('content', '')
                # 可以按需添加更多信息，如时间戳、AI模型等
            })

        logger.debug(
            f"Context for ConvID {conversation_id} generated with {len(context['recent_history'])} history messages.")
        return context

    # +++++++++++++++++++++++++++++++++++++++++++++++++

    def get_conversation_history(self, conversation_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取对话历史（用于上下文分析）。如果无DB，则从内存获取。"""
        if not self.db:
            if conversation_id not in self._memory_conversations:
                return []
            # 内存消息已按时间顺序存储，取最后N条
            return self._memory_messages.get(conversation_id, [])[-limit:]

        try:
            query = """
                SELECT id, is_user, content, created_at, ai_model_used, ai_strategy
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            messages = self.db.execute_query(query, (conversation_id, limit))
            for msg in messages: msg['is_user'] = bool(msg.get('is_user'))
            return list(reversed(messages))
        except Exception as e:
            logger.error(f"Failed to get DB conversation history for {conversation_id}: {str(e)}")
            return []

    def get_user_conversations(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取指定用户角色的对话列表，支持分页。
        
        Args:
            user_id: 用户角色ID (0=用户, 1=系统)
            limit: 每页数量
            offset: 偏移量
            
        Returns:
            List[Dict[str, Any]]: 对话列表
        """
        logger.info(f"Fetching conversations for user_id={user_id} with limit={limit}, offset={offset}")
        
        if not self.db:
            # 内存模式下，获取指定用户的对话
            user_convs = []
            for conv_id, conv_data in self._memory_conversations.items():
                if conv_data['user_id'] == user_id:
                    msgs = self._memory_messages.get(conv_id, [])
                    last_ai_msg_content = None
                    if msgs:
                        for msg in reversed(msgs):
                            if not msg['is_user']:
                                last_ai_msg_content = msg['content']
                                break
                    # 确保ID是字符串格式，并添加conversation_id字段
                    conv_id_str = str(conv_id)
                    user_convs.append({
                        'id': conv_id_str,
                        'conversation_id': conv_id_str,  # 添加前端期望的字段
                        'title': conv_data['title'],
                        'user_id': conv_data['user_id'],
                        'created_at': conv_data['created_at'], 
                        'updated_at': conv_data['updated_at'],
                        'status': 'active' if conv_data.get('status', 1) == 1 else 'archived',
                        'last_ai_message': last_ai_msg_content,
                        'message_count': len(msgs),
                        'ai_models_used': len(set(m['ai_model_used'] for m in msgs if m['ai_model_used']))
                    })
            # 内存排序和分页
            user_convs.sort(key=lambda c: c['updated_at'], reverse=True)
            return user_convs[offset: offset + limit]
            
        try:
            query = """
                SELECT c.id, c.title, c.user_id, c.created_at, c.updated_at, c.status,
                       (SELECT content FROM messages
                        WHERE conversation_id = c.id AND is_user = 0
                        ORDER BY created_at DESC LIMIT 1) as last_ai_message,
                       (SELECT COUNT(*) FROM messages
                        WHERE conversation_id = c.id) as message_count,
                       (SELECT COUNT(DISTINCT ai_model_used) FROM messages
                        WHERE conversation_id = c.id AND ai_model_used IS NOT NULL) as ai_models_used
                FROM conversations c
                WHERE c.user_id = %s
                ORDER BY c.updated_at DESC
                LIMIT %s OFFSET %s
            """
            conversations = self.db.execute_query(query, (user_id, limit, offset))
            for conv in conversations: 
                conv['status'] = 'active' if conv.get('status', 1) == 1 else 'archived'
                # 确保ID是字符串格式，并添加conversation_id字段
                conv['id'] = str(conv['id'])
                conv['conversation_id'] = conv['id']  # 添加前端期望的字段
            logger.debug(f"DB: Retrieved {len(conversations)} conversations for user_id {user_id}")
            return conversations
        except Exception as e:
            logger.error(f"Failed to get user conversations for user_id {user_id}: {str(e)}")
            return []
            
    def count_user_conversations(self, user_id: int) -> int:
        """
        获取指定用户角色的对话总数。
        
        Args:
            user_id: 用户角色ID (0=用户, 1=系统)
            
        Returns:
            int: 对话总数
        """
        if not self.db:
            # 内存模式下，计算指定用户的对话数量
            return sum(1 for conv_data in self._memory_conversations.values() if conv_data['user_id'] == user_id)
            
        try:
            query = "SELECT COUNT(*) as count FROM conversations WHERE user_id = %s"
            result = self.db.execute_query(query, (user_id,))
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count user conversations for user_id {user_id}: {str(e)}")
            return 0
            
    def count_all_conversations(self) -> int:
        """
        获取所有对话的总数。
        
        Returns:
            int: 对话总数
        """
        if not self.db:
            # 内存模式下，计算所有对话数量
            return len(self._memory_conversations)
            
        try:
            query = "SELECT COUNT(*) as count FROM conversations"
            result = self.db.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to count all conversations: {str(e)}")
            return 0
            
    def get_all_conversations(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取所有对话列表，支持分页。
        
        Args:
            limit: 每页数量
            offset: 偏移量
            
        Returns:
            List[Dict[str, Any]]: 对话列表
        """
        logger.info(f"Fetching all conversations with limit={limit}, offset={offset}")
        
        if not self.db:
            # 内存模式下，获取所有对话
            all_convs = []
            for conv_id, conv_data in self._memory_conversations.items():
                msgs = self._memory_messages.get(conv_id, [])
                last_ai_msg_content = None
                if msgs:
                    for msg in reversed(msgs):
                        if not msg['is_user']:
                            last_ai_msg_content = msg['content']
                            break
                # 确保ID是字符串格式，并添加conversation_id字段
                conv_id_str = str(conv_id)
                all_convs.append({
                    'id': conv_id_str,
                    'conversation_id': conv_id_str,  # 添加前端期望的字段
                    'title': conv_data['title'],
                    'user_id': conv_data['user_id'],
                    'created_at': conv_data['created_at'], 
                    'updated_at': conv_data['updated_at'],
                    'status': 'active' if conv_data.get('status', 1) == 1 else 'archived',
                    'last_ai_message': last_ai_msg_content,
                    'message_count': len(msgs),
                    'ai_models_used': len(set(m['ai_model_used'] for m in msgs if m['ai_model_used']))
                })
            # 内存排序和分页
            all_convs.sort(key=lambda c: c['updated_at'], reverse=True)
            return all_convs[offset: offset + limit]
            
        try:
            query = """
                SELECT c.id, c.title, c.user_id, c.created_at, c.updated_at, c.status,
                       (SELECT content FROM messages
                        WHERE conversation_id = c.id AND is_user = 0
                        ORDER BY created_at DESC LIMIT 1) as last_ai_message,
                       (SELECT COUNT(*) FROM messages
                        WHERE conversation_id = c.id) as message_count,
                       (SELECT COUNT(DISTINCT ai_model_used) FROM messages
                        WHERE conversation_id = c.id AND ai_model_used IS NOT NULL) as ai_models_used
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT %s OFFSET %s
            """
            conversations = self.db.execute_query(query, (limit, offset))
            for conv in conversations: 
                conv['status'] = 'active' if conv.get('status', 1) == 1 else 'archived'
                # 确保ID是字符串格式，并添加conversation_id字段
                conv['id'] = str(conv['id'])
                conv['conversation_id'] = conv['id']  # 添加前端期望的字段
            logger.debug(f"DB: Retrieved {len(conversations)} conversations")
            return conversations
        except Exception as e:
            logger.error(f"Failed to get all conversations: {str(e)}")
            return []

    def _get_message_visuals(self, message_id: int) -> List[Dict[str, Any]]:
        """获取消息的可视化元素。内存模式下，visuals直接在消息中。"""
        if not self.db:
            # 在内存模式下，visuals应已附加到消息对象上
            # 这个方法主要用于DB模式，或者内存模式下如果visuals分开存储
            logger.debug(
                f"_get_message_visuals called in no-DB mode for message_id {message_id}. Visuals should be on message obj.")
            return []  # 或者从内存消息对象中提取（如果设计如此）

        try:
            visual_query = """
                SELECT id, visual_type, visual_order, title, data_json
                FROM message_visuals
                WHERE message_id = %s
                ORDER BY visual_order
            """
            visuals_db = self.db.execute_query(visual_query, (message_id,))
            for visual in visuals_db:
                try:
                    visual['data'] = json.loads(visual['data_json'])
                    del visual['data_json']  # 移除原始json字符串
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse visual data_json for visual_id {visual.get('id')}: {e}")
                    visual['data'] = {}  # 出错时给个空字典
            return visuals_db
        except Exception as e:
            logger.error(f"Failed to get DB message visuals for message_id {message_id}: {str(e)}")
            return []

    def _generate_ai_usage_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # (保持原样)
        ai_messages = [m for m in messages if not m.get('is_user') and m.get('ai_model_used')]
        if not ai_messages: return {}
        model_usage: Dict[str, int] = {}
        strategy_usage: Dict[str, int] = {}
        total_processing_time = 0.0
        total_confidence = 0.0
        valid_confidence_count = 0
        valid_time_count = 0

        for msg in ai_messages:
            model = msg.get('ai_model_used')
            strategy = msg.get('ai_strategy')
            proc_time = msg.get('processing_time')
            conf_score = msg.get('confidence_score')

            if model: model_usage[model] = model_usage.get(model, 0) + 1
            if strategy: strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            if isinstance(proc_time, (int, float)): total_processing_time += proc_time; valid_time_count += 1
            if isinstance(conf_score, (int, float)): total_confidence += conf_score; valid_confidence_count += 1

        return {
            'total_ai_messages': len(ai_messages),
            'model_usage': model_usage,
            'strategy_usage': strategy_usage,
            'avg_processing_time': total_processing_time / valid_time_count if valid_time_count else 0,
            'avg_confidence': total_confidence / valid_confidence_count if valid_confidence_count else 0
        }

    def _update_conversation_timestamp(self, conversation_id: int):
        if not self.db:
            if conversation_id in self._memory_conversations:
                self._memory_conversations[conversation_id]['updated_at'] = datetime.now()
            return
        try:
            query = "UPDATE conversations SET updated_at = NOW() WHERE id = %s"
            self.db.execute_update(query, (conversation_id,))
        except Exception as e:
            logger.warning(f"Failed to update DB conversation timestamp for {conversation_id}: {str(e)}")

    def _generate_welcome_message(self, user_id: int) -> str:
        # (保持原样)
        return """您好！我是智能金融数据分析助手。

我可以帮您：
📊 分析系统运营数据和财务状况
📈 预测资金流动和用户趋势
🔍 深度解读业务指标和风险
💡 提供数据驱动的业务建议

请告诉我您想了解什么，我会为您提供专业的分析和洞察！"""


def create_conversation_manager(database_connector: Optional[DatabaseConnector]) -> ConversationManager:
    return ConversationManager(database_connector)