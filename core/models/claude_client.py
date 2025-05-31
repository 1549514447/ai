import anthropic
from typing import Dict, Any, Optional, List
import json
import asyncio
import logging
from datetime import datetime
import httpx
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 自定义JSON编码器，处理不可序列化的对象
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # 检查对象是否有__dict__属性（大多数自定义类都有）
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # 检查对象是否有to_dict方法
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        # 检查对象是否有__str__方法
        elif hasattr(obj, '__str__'):
            return str(obj)
        # 其他类型的对象，返回其类名
        return f"<{obj.__class__.__name__}>"

# 安全的JSON转换函数
def safe_json_dumps(obj, **kwargs):
    try:
        return json.dumps(obj, ensure_ascii=False, cls=CustomJSONEncoder, **kwargs)
    except Exception as e:
        logger.warning(f"JSON序列化失败，使用安全模式: {e}")
        # 尝试转换为字符串
        if isinstance(obj, dict):
            safe_dict = {}
            for k, v in obj.items():
                try:
                    # 递归处理嵌套字典
                    if isinstance(v, dict):
                        safe_dict[k] = safe_json_dumps(v)
                    else:
                        safe_dict[k] = str(v)
                except:
                    safe_dict[k] = "<不可序列化对象>"
            return json.dumps(safe_dict, ensure_ascii=False, **kwargs)
        else:
            return json.dumps({"data": str(obj)}, ensure_ascii=False, **kwargs)


class ClaudeClient:
    def __init__(self, api_key: str = None):
        """
        初始化Claude Sonnet 4客户端

        Args:
            api_key: API密钥，如果不传入则从环境变量CLAUDE_API_KEY获取
        """
        # 如果没有传入api_key，从环境变量获取
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('CLAUDE_API_KEY')
            if not api_key:
                raise ValueError("CLAUDE_API_KEY not found in environment variables or .env file")

        # 创建一个不使用代理的自定义 httpx 客户端
        custom_http_client = httpx.Client(
            timeout=httpx.Timeout(180.0),  # 增加超时时间到180秒
            follow_redirects=True
        )

        # 检查anthropic版本
        self.anthropic_version = getattr(anthropic, "__version__", "0.0.0")
        logger.info(f"Detected Anthropic SDK version: {self.anthropic_version}")
        
        # 初始化客户端 - 尝试多种可能的初始化方式
        try:
            # 尝试方法1: 直接初始化Anthropic
            self.client = anthropic.Anthropic(api_key=api_key, http_client=custom_http_client)
            logger.info("Initialized with anthropic.Anthropic")
        except Exception as e1:
            logger.warning(f"Failed to initialize with anthropic.Anthropic: {e1}")
            try:
                # 尝试方法2: 使用Client类
                self.client = anthropic.Client(api_key=api_key, http_client=custom_http_client)
                logger.info("Initialized with anthropic.Client")
            except Exception as e2:
                logger.warning(f"Failed to initialize with anthropic.Client: {e2}")
                try:
                    # 尝试方法3: 不带http_client参数
                    self.client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Initialized with anthropic.Anthropic without http_client")
                except Exception as e3:
                    logger.error(f"All initialization methods failed: {e3}")
                    raise ValueError("Failed to initialize Anthropic client with any method")

        # 检测可用的API方法
        self.has_messages_api = hasattr(self.client, 'messages') and hasattr(getattr(self.client, 'messages', None), 'create')
        self.has_completion_api = hasattr(self.client, 'completion')
        
        logger.info(f"API capabilities: messages_api={self.has_messages_api}, completion_api={self.has_completion_api}")
        
        if not (self.has_messages_api or self.has_completion_api):
            logger.warning("No known API methods detected. Client may not work properly.")

        # 设置模型和参数
        self.model = "claude-sonnet-4-20250514"  # 使用最新的Claude 4模型
        self.max_tokens = 8000
        self.max_retries = 1  # 最大重试次数
        self.retry_delay = 2  # 初始重试延迟（秒）
        
        logger.info(f"ClaudeClient initialized successfully")
        logger.info(f"Using model: {self.model}")

    # 1. 修复 core/models/claude_client.py 的 analyze_complex_query 方法

    async def analyze_complex_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        复杂查询深度分析 - Claude的核心优势
        """
        try:
            # 简化提示以减少处理时间
            system_prompt = """你是一个专业的金融数据分析师和智能业务顾问。
    提供简洁、专业的分析，包括业务洞察、风险评估和决策建议。"""

            # 🔥 修复：限制context大小，避免请求过大
            user_content = f"""
    用户查询：{query}

    可用数据上下文：
    {safe_json_dumps(context, indent=2) if context else "暂无具体数据"}

    请进行分析并提供专业建议。
    """

            # 🔥 新增：智能截断过大的context
            MAX_CONTEXT_SIZE = 8000  # 设置最大context大小
            if context and len(safe_json_dumps(context)) > MAX_CONTEXT_SIZE:
                logger.warning("Context过大，正在智能截断...")

                # 保留最重要的数据
                simplified_context = {}
                if 'system_data' in context:
                    simplified_context['system_data'] = context['system_data']
                if 'query_metadata' in context:
                    simplified_context['query_metadata'] = {
                        'query': context['query_metadata'].get('query', ''),
                        'data_sources_used': context['query_metadata'].get('data_sources_used', [])
                    }

                user_content = f"""
    用户查询：{query}

    可用数据上下文：(已优化显示关键信息)
    {safe_json_dumps(simplified_context, indent=2)}

    请进行分析并提供专业建议。
    """

            # 尝试不同的API调用方式
            analysis = None
            last_error = None

            # 实现自定义重试逻辑
            for attempt in range(self.max_retries):
                try:
                    # 尝试使用messages API
                    if self.has_messages_api:
                        try:
                            logger.info(f"Attempting to use messages API (attempt {attempt + 1}/{self.max_retries})")
                            message = await asyncio.to_thread(
                                self.client.messages.create,
                                model=self.model,
                                max_tokens=self.max_tokens,
                                system=system_prompt,
                                messages=[{"role": "user", "content": user_content}]
                            )

                            # 处理返回内容
                            if hasattr(message, 'content'):
                                content_list = message.content
                                if isinstance(content_list, list) and len(content_list) > 0:
                                    analysis = ""
                                    for content_item in content_list:
                                        if hasattr(content_item, 'text'):
                                            analysis += content_item.text
                                        elif isinstance(content_item, dict) and 'text' in content_item:
                                            analysis += content_item['text']

                                    if analysis:
                                        logger.info("Successfully retrieved analysis from messages API")
                                        break

                            # 处理stop_reason
                            if hasattr(message, 'stop_reason') and message.stop_reason == "refusal":
                                logger.warning("Claude refused to generate content for safety reasons")
                                return {
                                    "success": False,
                                    "error": "Content generation refused for safety reasons",
                                    "claude_failed": True,
                                    "should_fallback_to_gpt": True,
                                    "model_used": self.model,
                                    "query_type": "complex_analysis",
                                    "timestamp": datetime.now().isoformat()
                                }

                        except Exception as e:
                            last_error = e
                            logger.error(f"Messages API call failed (attempt {attempt + 1}): {e}")

                            # 🔥 新增：特殊处理余额不足错误
                            if "credit balance is too low" in str(e):
                                logger.error("Claude API余额不足，将立即降级到GPT-4o")
                                return {
                                    "success": False,
                                    "error": "Claude API余额不足",
                                    "claude_failed": True,
                                    "should_fallback_to_gpt": True,
                                    "fallback_reason": "insufficient_credits",
                                    "user_message": "🔄 Claude API暂时不可用（余额不足），系统已自动切换到GPT-4o为您提供服务",
                                    "model_used": self.model,
                                    "query_type": "complex_analysis",
                                    "timestamp": datetime.now().isoformat()
                                }

                    # 如果尝试失败，等待一段时间后重试
                    if analysis is None and attempt < self.max_retries - 1:
                        retry_delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)

                except Exception as e:
                    last_error = e
                    logger.error(f"API call attempt {attempt + 1} failed with error: {e}")
                    if attempt < self.max_retries - 1:
                        retry_delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)

            # 如果所有API调用都失败了
            if analysis is None:
                error_msg = str(last_error) if last_error else "All API call methods failed"
                logger.error(f"Claude分析失败: {error_msg}")

                # 🔥 明确的降级信息
                return {
                    "success": False,
                    "error": error_msg,
                    "claude_failed": True,
                    "should_fallback_to_gpt": True,
                    "fallback_reason": "api_failure",
                    "user_message": "🔄 Claude服务暂时不可用，系统已自动切换到GPT-4o为您提供服务",
                    "model_used": self.model,
                    "query_type": "complex_analysis",
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "success": True,
                "analysis": analysis,
                "model_used": self.model,
                "query_type": "complex_analysis",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Claude分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "claude_failed": True,
                "should_fallback_to_gpt": True,
                "fallback_reason": "exception",
                "user_message": f"🔄 Claude服务异常，系统已自动切换到GPT-4o为您提供服务",
                "model_used": self.model,
                "query_type": "complex_analysis",
                "timestamp": datetime.now().isoformat()
            }

    async def decompose_query(self, query: str) -> Dict[str, Any]:
        """
        查询分解 - 将复杂查询分解为可执行的步骤
        """
        try:
            decomposition_prompt = f"""
分析以下查询，将其分解为具体的执行步骤和数据需求：

查询："{query}"

复杂度定义：
- simple: 单一直接查询，1个明确目标，无需复杂计算
- medium: 2-3个步骤，涉及基础计算、简单预测或对比分析
- complex: 3+个步骤，多维分析，历史趋势，复杂推理，多情景预测

返回JSON格式：
{{
    "complexity": "simple/medium/complex",
    "confidence": 0.0-1.0,
    "reasoning": "判断理由",
    "key_indicators": [
        {{
            "indicator": "指标名称",
            "value": "指标值",
            "weight": "权重"
        }}
    ],
    "processing_requirements": {{
        "estimated_steps": 步骤数量,
        "data_complexity": "low/medium/high",
        "calculation_intensity": "low/medium/high",
        "analysis_depth": "surface/moderate/deep"
    }},
    "recommended_approach": "direct_query/analysis_with_calculation/multi_model_collaboration"
}}
"""

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=3500,
                messages=[{"role": "user", "content": decomposition_prompt}]
            )
            result_text = message.content[0].text

            # 解析JSON结果
            decomposition_result = json.loads(result_text)

            return {
                "success": True,
                "decomposition": decomposition_result,
                "model_used": self.model,
                "timestamp": datetime.now().isoformat()
            }

        except json.JSONDecodeError as e:
            logger.error(f"查询分解JSON解析失败: {str(e)}")
            return {
                "success": False,
                "error": f"JSON解析失败: {str(e)}",
                "raw_response": result_text if 'result_text' in locals() else None
            }
        except Exception as e:
            logger.error(f"查询分解失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }

    async def generate_business_insights(self, analysis_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        业务洞察生成 - Claude的核心价值
        """
        try:
            insight_prompt = f"""
基于以下分析数据，为用户查询生成深度的业务洞察和建议：

原始查询：{query}

分析数据：
{safe_json_dumps(analysis_data, indent=2)}

请生成包含以下内容的业务洞察：

1. 核心发现 (Key Findings)
2. 风险预警 (Risk Warnings) 
3. 机会识别 (Opportunities)
4. 具体建议 (Actionable Recommendations)
5. 后续关注点 (Follow-up Points)

返回JSON格式：
{{
    "executive_summary": "执行摘要",
    "key_findings": [
        {{
            "finding": "发现内容",
            "impact": "影响程度",
            "supporting_data": "支撑数据"
        }}
    ],
    "risk_warnings": [
        {{
            "risk": "风险描述", 
            "probability": "发生概率",
            "impact": "影响程度",
            "mitigation": "缓解措施"
        }}
    ],
    "opportunities": [
        {{
            "opportunity": "机会描述",
            "potential_value": "潜在价值",
            "implementation": "实施建议"
        }}
    ],
    "recommendations": [
        {{
            "priority": "高/中/低",
            "action": "具体行动",
            "timeline": "时间框架",
            "expected_outcome": "预期结果"
        }}
    ],
    "follow_up_questions": ["延展问题1", "延展问题2"]
}}
"""

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": insight_prompt}]
            )
            insights_text = message.content[0].text

            insights = json.loads(insights_text)

            return {
                "success": True,
                "insights": insights,
                "model_used": self.model,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"业务洞察生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }

    async def assess_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        查询复杂度评估 - 智能路由的基础
        """
        try:
            complexity_prompt = f"""
评估以下查询的复杂度和处理要求：

查询："{query}"

复杂度定义：
- simple: 单一直接查询，1个明确目标，无需复杂计算
- medium: 2-3个步骤，涉及基础计算、简单预测或对比分析
- complex: 3+个步骤，多维分析，历史趋势，复杂推理，多情景预测

返回JSON格式：
{{
    "complexity": "simple/medium/complex",
    "confidence": 0.0-1.0,
    "reasoning": "判断理由",
    "key_indicators": [
        {{
            "indicator": "指标名称",
            "value": "指标值",
            "weight": "权重"
        }}
    ],
    "processing_requirements": {{
        "estimated_steps": 步骤数量,
        "data_complexity": "low/medium/high",
        "calculation_intensity": "low/medium/high",
        "analysis_depth": "surface/moderate/deep"
    }},
    "recommended_approach": "direct_query/analysis_with_calculation/multi_model_collaboration"
}}
"""

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=3500,
                messages=[{"role": "user", "content": complexity_prompt}]
            )
            complexity_result_text = message.content[0].text

            complexity_result = json.loads(complexity_result_text)

            return {
                "success": True,
                "complexity_analysis": complexity_result,
                "model_used": self.model,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"复杂度评估失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_complexity": "medium"  # 失败时的默认复杂度
            }

    async def decompose_complex_query(self, query: str) -> Dict[str, Any]:
        """
        复杂查询分解 - 为 intelligent_router.py 提供的方法别名
        """
        return await self.decompose_query(query)

    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        查询复杂度分析 - 为 intelligent_router.py 提供的方法别名
        """
        return await self.assess_query_complexity(query)

    async def analyze_financial_complexity(self, text: str) -> Dict[str, float]:
        """
        分析文本的金融复杂度
        """
        try:
            system_prompt = """你是一个专业的金融文本分析专家。
你的任务是评估用户查询的金融复杂度，并给出0-1之间的评分。

评分标准：
- 金融专业度：文本包含的金融专业术语和概念的复杂程度
- 分析复杂度：需要的分析步骤和逻辑推理的复杂程度
- 数据需求：分析所需的数据量和数据处理复杂度
- 风险评估难度：涉及的风险评估的复杂程度
- 决策建议难度：生成决策建议的难度

你需要以JSON格式返回评分结果，格式如下：
{
  "financial_expertise": 0.7,  // 金融专业度评分
  "analysis_complexity": 0.8,  // 分析复杂度评分
  "data_requirements": 0.5,    // 数据需求评分
  "risk_assessment": 0.6,      // 风险评估难度评分
  "decision_making": 0.7,      // 决策建议难度评分
  "overall_complexity": 0.66   // 综合复杂度评分（以上各项的平均值）
}

只返回JSON格式的评分结果，不要有任何其他解释或文字。"""

            user_content = f"""
请评估以下查询的金融复杂度：

{text}
"""

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            result_text = message.content[0].text

            # 解析JSON结果
            result = json.loads(result_text)
            return result

        except Exception as e:
            logger.error(f"金融复杂度分析失败: {str(e)}")
            # 返回默认值
            return {
                "financial_expertise": 0.5,
                "analysis_complexity": 0.5,
                "data_requirements": 0.5,
                "risk_assessment": 0.5,
                "decision_making": 0.5,
                "overall_complexity": 0.5
            }

    async def detect_date_ranges(self, text: str) -> Dict[str, Any]:
        """
        检测文本中的日期范围
        """
        try:
            system_prompt = """你是一个专业的文本分析专家，专注于从文本中提取时间和日期信息。
你的任务是从用户查询中识别出所有提及的日期、时间段和时间范围。

请以JSON格式返回结果，格式如下：
{
  "date_ranges": [
    {
      "start_date": "2023-01-01",  // ISO格式的开始日期，如果没有明确提及则为null
      "end_date": "2023-03-31",    // ISO格式的结束日期，如果没有明确提及则为null
      "period_type": "quarter",    // 时间段类型：day, week, month, quarter, year, custom
      "description": "2023年第一季度"  // 对该时间段的描述
    }
  ],
  "specific_dates": [
    {
      "date": "2023-05-01",        // ISO格式的具体日期
      "description": "劳动节"       // 对该日期的描述
    }
  ],
  "relative_periods": [
    {
      "period_type": "last_month",  // 相对时间类型：yesterday, last_week, last_month, last_quarter, last_year等
      "description": "上个月"        // 对该相对时间的描述
    }
  ],
  "has_time_info": true             // 是否包含时间信息
}

只返回JSON格式的结果，不要有任何其他解释或文字。如果没有检测到任何日期或时间信息，则返回空数组并将has_time_info设为false。"""

            user_content = f"""
请从以下文本中提取所有日期和时间范围信息：

{text}
"""

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            result_text = message.content[0].text

            # 解析JSON结果
            result = json.loads(result_text)
            return result

        except Exception as e:
            logger.error(f"日期范围检测失败: {str(e)}")
            # 返回默认值
            return {
                "date_ranges": [],
                "specific_dates": [],
                "relative_periods": [],
                "has_time_info": False
            }

    async def extract_query_features(self, query: str) -> Dict[str, Any]:
        """
        提取查询特征，用于智能路由
        """
        try:
            system_prompt = """你是一个专业的查询分析专家。
你的任务是分析用户查询的特征，用于后续的智能路由。

请以JSON格式返回结果，格式如下：
{
  "query_type": "analysis",  // 查询类型：simple, analysis, prediction, recommendation, comparison
  "domain": "finance",       // 领域：finance, business, marketing, operations, general
  "complexity": 0.7,         // 复杂度评分(0-1)
  "time_sensitivity": 0.5,   // 时间敏感度(0-1)
  "requires_calculation": true,  // 是否需要计算
  "requires_visualization": false,  // 是否需要可视化
  "key_entities": ["revenue", "profit margin"],  // 关键实体
  "key_metrics": ["growth rate", "ROI"]  // 关键指标
}

只返回JSON格式的结果，不要有任何其他解释或文字。"""

            user_content = f"""
请分析以下用户查询的特征：

{query}
"""

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            result_text = message.content[0].text

            # 解析JSON结果
            result = json.loads(result_text)
            return result

        except Exception as e:
            logger.error(f"查询特征提取失败: {str(e)}")
            # 返回默认值
            return {
                "query_type": "analysis",
                "domain": "general",
                "complexity": 0.5,
                "time_sensitivity": 0.5,
                "requires_calculation": False,
                "requires_visualization": False,
                "key_entities": [],
                "key_metrics": []
            }

    async def generate_text(self, prompt: str, max_tokens: int = 1500, system_prompt: str = None) -> Dict[str, Any]:
        """
        生成文本响应 - 为IntelligentQAOrchestrator提供的通用文本生成方法
        根据可用的API方法（messages API或completion API）动态选择调用方式
        
        Args:
            prompt: 用户提示词
            max_tokens: 最大生成token数
            system_prompt: 可选的系统提示词
            
        Returns:
            包含生成文本的字典，格式为 {"success": bool, "text": str, ...}
        """
        logger.info(f"使用Claude生成文本响应，模型: {self.model}, 最大tokens: {max_tokens}")
        
        try:
            # 尝试使用messages API（优先）
            if self.has_messages_api:
                logger.debug("使用messages API生成文本")
                messages = [{"role": "user", "content": prompt}]
                
                # 如果提供了系统提示词
                message_kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": messages
                }
                
                if system_prompt:
                    message_kwargs["system"] = system_prompt
                
                # messages.create是同步的，使用asyncio.to_thread运行
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    **message_kwargs
                )
                
                # 提取文本内容
                content_text = ""
                if hasattr(response, 'content') and response.content:
                    for content_item in response.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text
                
                return {
                    "success": True,
                    "text": content_text,
                    "model_used": self.model,
                    "api_method": "messages"
                }
            
            # 尝试使用completion API（备选）
            elif self.has_completion_api:
                logger.debug("使用completion API生成文本")
                # 构建提示词
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt
                
                # completion是同步的，使用asyncio.to_thread运行
                response = await asyncio.to_thread(
                    self.client.completion,
                    prompt=full_prompt,
                    model=self.model,
                    max_tokens_to_sample=max_tokens
                )
                
                # 提取文本内容
                completion_text = response.completion if hasattr(response, 'completion') else str(response)
                
                return {
                    "success": True,
                    "text": completion_text,
                    "model_used": self.model,
                    "api_method": "completion"
                }
            
            # 两种API方法都不可用
            else:
                error_msg = "Claude客户端没有可用的API方法（messages或completion）"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "text": "AI服务暂时不可用，请稍后再试。",
                    "model_used": self.model
                }
                
        except Exception as e:
            error_msg = f"Claude生成文本时发生错误: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "text": f"AI处理遇到问题: {str(e)}",
                "model_used": self.model
            }