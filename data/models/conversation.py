# data/models/conversation.py
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from data.connectors.database_connector import DatabaseConnector

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """对话数据模型"""
    id: int
    title: str
    user_id: int
    created_at: datetime
    updated_at: datetime
    status: str = 'active'


@dataclass
class Message:
    """消息数据模型"""
    id: int
    conversation_id: int
    is_user: bool
    content: str
    created_at: datetime
    visuals: List[Dict[str, Any]] = None

    # 新增AI相关字段
    ai_model_used: Optional[str] = None
    ai_strategy: Optional[str] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None


@dataclass
class Visual:
    """可视化元素数据模型"""
    id: int
    message_id: int
    visual_type: str  # 'chart', 'table', 'stats', 'insight'
    visual_order: int
    title: str
    data: Dict[str, Any]


class ConversationManager:
    """现代化对话管理器 - 支持智能上下文和AI协作"""

    def __init__(self, database_connector: DatabaseConnector):
        self.db = database_connector
        logger.info("ConversationManager initialized")

    # ============= 基础CRUD操作 =============

    def create_conversation(self, title: str, user_id: int,
                            initial_context: Dict[str, Any] = None) -> int:
        """
        创建新对话

        Args:
            title: 对话标题
            user_id: 用户ID
            initial_context: 初始上下文信息（新增）

        Returns:
            int: 新创建对话的ID
        """
        try:
            # 创建基础对话
            query = """
                INSERT INTO conversations (title, user_id, initial_context)
                VALUES (%s, %s, %s)
            """
            context_json = json.dumps(initial_context or {}, ensure_ascii=False)
            conversation_id = self.db.execute_insert(query, (title, user_id, context_json))

            # 添加欢迎消息
            welcome_message = self._generate_welcome_message(user_id)
            self.add_message(conversation_id, False, welcome_message,
                             ai_model_used="system", ai_strategy="welcome")

            logger.info(f"Created conversation {conversation_id} for user {user_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}")
            raise

    def add_message(self, conversation_id: int, is_user: bool, content: str,
                    ai_model_used: str = None, ai_strategy: str = None,
                    processing_time: float = None, confidence_score: float = None) -> int:
        """
        添加消息到对话（增强版）

        Args:
            conversation_id: 对话ID
            is_user: 是否为用户消息
            content: 消息内容
            ai_model_used: 使用的AI模型（新增）
            ai_strategy: AI处理策略（新增）
            processing_time: 处理时间（新增）
            confidence_score: 置信度评分（新增）

        Returns:
            int: 新添加消息的ID
        """
        try:
            query = """
                INSERT INTO messages (conversation_id, is_user, content, 
                                    ai_model_used, ai_strategy, processing_time, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            message_id = self.db.execute_insert(query, (
                conversation_id, is_user, content,
                ai_model_used, ai_strategy, processing_time, confidence_score
            ))

            # 更新对话的最后更新时间
            self._update_conversation_timestamp(conversation_id)

            logger.debug(f"Added message {message_id} to conversation {conversation_id}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            raise

    def add_visual(self, message_id: int, visual_type: str, visual_order: int,
                   title: str, data: Dict[str, Any]) -> int:
        """
        为消息添加可视化元素

        Args:
            message_id: 消息ID
            visual_type: 可视化类型
            visual_order: 显示顺序
            title: 标题
            data: 可视化数据

        Returns:
            int: 新添加可视化元素的ID
        """
        try:
            data_json = json.dumps(data, ensure_ascii=False)
            query = """
                INSERT INTO message_visuals (message_id, visual_type, visual_order, title, data_json)
                VALUES (%s, %s, %s, %s, %s)
            """
            visual_id = self.db.execute_insert(query, (
                message_id, visual_type, visual_order, title, data_json
            ))

            logger.debug(f"Added visual {visual_id} to message {message_id}")
            return visual_id

        except Exception as e:
            logger.error(f"Failed to add visual: {str(e)}")
            raise

    # ============= 智能查询方法 =============

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        获取完整对话内容（增强版）

        Args:
            conversation_id: 对话ID

        Returns:
            Dict: 包含对话信息、消息历史、AI使用统计等
        """
        try:
            # 获取对话基本信息
            conv_query = """
                SELECT id, title, user_id, created_at, updated_at, status, initial_context
                FROM conversations
                WHERE id = %s
            """
            conversation_data = self.db.execute_query(conv_query, (conversation_id,))
            if not conversation_data:
                return None

            conversation = conversation_data[0]

            # 解析初始上下文
            try:
                conversation['initial_context'] = json.loads(conversation['initial_context'] or '{}')
            except json.JSONDecodeError:
                conversation['initial_context'] = {}

            # 获取所有消息（包含AI信息）
            msg_query = """
                SELECT id, is_user, content, created_at, ai_model_used, 
                       ai_strategy, processing_time, confidence_score
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at
            """
            messages = self.db.execute_query(msg_query, (conversation_id,))

            # 为每条消息获取可视化元素
            for message in messages:
                message['visuals'] = self._get_message_visuals(message['id'])

            # 生成AI使用统计
            ai_stats = self._generate_ai_usage_stats(messages)

            return {
                'conversation': conversation,
                'messages': messages,
                'ai_usage_stats': ai_stats,
                'total_messages': len(messages),
                'user_messages': len([m for m in messages if m['is_user']]),
                'ai_messages': len([m for m in messages if not m['is_user']])
            }

        except Exception as e:
            logger.error(f"Failed to get conversation: {str(e)}")
            return None

    def get_conversation_history(self, conversation_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取对话历史（用于上下文分析）

        Args:
            conversation_id: 对话ID
            limit: 返回消息数量限制

        Returns:
            List[Dict]: 最近的对话消息列表
        """
        try:
            query = """
                SELECT id, is_user, content, created_at, ai_model_used, ai_strategy
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            messages = self.db.execute_query(query, (conversation_id, limit))

            # 反转顺序，使其按时间正序
            return list(reversed(messages))

        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            return []

    def get_user_conversations(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取用户的对话列表（增强版）

        Args:
            user_id: 用户ID
            limit: 返回的最大数量
            offset: 偏移量，用于分页

        Returns:
            List[Dict]: 对话列表，包含统计信息
        """
        try:
            # 获取对话列表和统计信息
            query = """
                SELECT c.id, c.title, c.created_at, c.updated_at, c.status,
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

            logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations

        except Exception as e:
            logger.error(f"Failed to get user conversations: {str(e)}")
            return []

    # ============= 智能分析方法（为context_manager提供数据） =============

    def get_user_query_patterns(self, user_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """
        获取用户查询模式（供智能上下文分析使用）

        Args:
            user_id: 用户ID
            days: 分析天数

        Returns:
            List[Dict]: 用户查询模式数据
        """
        try:
            query = """
                SELECT m.content, m.ai_strategy, m.created_at, c.title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = %s AND m.is_user = 1 
                      AND m.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY m.created_at DESC
            """
            patterns = self.db.execute_query(query, (user_id, days))

            return patterns

        except Exception as e:
            logger.error(f"Failed to get user query patterns: {str(e)}")
            return []

    def get_related_conversations(self, user_id: int, keywords: List[str],
                                  exclude_id: int = None) -> List[Dict[str, Any]]:
        """
        获取相关对话（供智能关联分析使用）

        Args:
            user_id: 用户ID
            keywords: 关键词列表
            exclude_id: 排除的对话ID

        Returns:
            List[Dict]: 相关对话列表
        """
        try:
            # 构建关键词搜索条件
            keyword_conditions = []
            params = [user_id]

            for keyword in keywords:
                keyword_conditions.append("(c.title LIKE %s OR m.content LIKE %s)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])

            exclude_condition = ""
            if exclude_id:
                exclude_condition = "AND c.id != %s"
                params.append(exclude_id)

            query = f"""
                SELECT DISTINCT c.id, c.title, c.updated_at,
                       (SELECT content FROM messages 
                        WHERE conversation_id = c.id AND is_user = 0
                        ORDER BY created_at DESC LIMIT 1) as last_response
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = %s AND ({' OR '.join(keyword_conditions)})
                      {exclude_condition}
                ORDER BY c.updated_at DESC
                LIMIT 5
            """

            related = self.db.execute_query(query, params)
            return related

        except Exception as e:
            logger.error(f"Failed to get related conversations: {str(e)}")
            return []

    # ============= 管理操作 =============

    def update_conversation_title(self, conversation_id: int, new_title: str) -> bool:
        """更新对话标题"""
        try:
            query = """
                UPDATE conversations
                SET title = %s, updated_at = NOW()
                WHERE id = %s
            """
            affected = self.db.execute_update(query, (new_title, conversation_id))

            logger.info(f"Updated conversation {conversation_id} title to: {new_title}")
            return affected > 0

        except Exception as e:
            logger.error(f"Failed to update conversation title: {str(e)}")
            return False

    def delete_conversation(self, conversation_id: int) -> bool:
        """删除对话及其所有相关数据"""
        try:
            query = """
                DELETE FROM conversations
                WHERE id = %s
            """
            affected = self.db.execute_update(query, (conversation_id,))

            logger.info(f"Deleted conversation {conversation_id}")
            return affected > 0

        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            return False

    # ============= 私有辅助方法 =============

    def _get_message_visuals(self, message_id: int) -> List[Dict[str, Any]]:
        """获取消息的可视化元素"""
        try:
            visual_query = """
                SELECT id, visual_type, visual_order, title, data_json
                FROM message_visuals
                WHERE message_id = %s
                ORDER BY visual_order
            """
            visuals = self.db.execute_query(visual_query, (message_id,))

            # 解析JSON数据
            for visual in visuals:
                try:
                    visual['data'] = json.loads(visual['data_json'])
                    del visual['data_json']
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse visual data: {e}")
                    visual['data'] = {}

            return visuals
        except Exception as e:
            logger.error(f"Failed to get message visuals: {str(e)}")
            return []

    def _generate_ai_usage_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成AI使用统计"""
        ai_messages = [m for m in messages if not m['is_user'] and m['ai_model_used']]

        if not ai_messages:
            return {}

        # 统计AI模型使用情况
        model_usage = {}
        strategy_usage = {}
        total_processing_time = 0
        total_confidence = 0

        for msg in ai_messages:
            model = msg['ai_model_used']
            strategy = msg['ai_strategy']

            if model:
                model_usage[model] = model_usage.get(model, 0) + 1
            if strategy:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            if msg['processing_time']:
                total_processing_time += msg['processing_time']
            if msg['confidence_score']:
                total_confidence += msg['confidence_score']

        return {
            'total_ai_messages': len(ai_messages),
            'model_usage': model_usage,
            'strategy_usage': strategy_usage,
            'avg_processing_time': total_processing_time / len(ai_messages) if ai_messages else 0,
            'avg_confidence': total_confidence / len(ai_messages) if ai_messages else 0
        }

    def _update_conversation_timestamp(self, conversation_id: int):
        """更新对话的最后更新时间"""
        try:
            query = "UPDATE conversations SET updated_at = NOW() WHERE id = %s"
            self.db.execute_update(query, (conversation_id,))
        except Exception as e:
            logger.warning(f"Failed to update conversation timestamp: {str(e)}")

    def _generate_welcome_message(self, user_id: int) -> str:
        """生成欢迎消息"""
        return """您好！我是智能金融数据分析助手。

我可以帮您：
📊 分析系统运营数据和财务状况
📈 预测资金流动和用户趋势  
🔍 深度解读业务指标和风险
💡 提供数据驱动的业务建议

请告诉我您想了解什么，我会为您提供专业的分析和洞察！"""


# ============= 工厂函数 =============

def create_conversation_manager(database_connector: DatabaseConnector) -> ConversationManager:
    """
    创建对话管理器实例

    Args:
        database_connector: 数据库连接器

    Returns:
        ConversationManager实例
    """
    return ConversationManager(database_connector)