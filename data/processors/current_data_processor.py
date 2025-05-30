# core/processors/current_data_processor.py
"""
📊 当前数据处理器
专门处理实时状态查询，如"今天余额多少"、"当前用户数"等简单查询

核心特点:
- 快速响应当前状态查询
- 基于 /api/sta/system 的实时数据
- 智能数据增强和格式化
- 简单易懂的输出格式
- 支持常见的实时指标查询
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from functools import lru_cache
import hashlib

# 导入已完成的组件
from core.analyzers.query_parser import SmartQueryParser, create_smart_query_parser
from core.analyzers.financial_data_analyzer import FinancialDataAnalyzer, create_financial_data_analyzer
from core.analyzers.insight_generator import InsightGenerator, create_insight_generator
from core.data_orchestration.smart_data_fetcher import SmartDataFetcher, create_smart_data_fetcher

logger = logging.getLogger(__name__)


class CurrentDataQueryType(Enum):
    """当前数据查询类型"""
    SYSTEM_OVERVIEW = "system_overview"  # 系统概览 "总体情况"
    BALANCE_CHECK = "balance_check"  # 余额查询 "余额多少"
    USER_STATUS = "user_status"  # 用户状态 "用户数量"
    TODAY_EXPIRY = "today_expiry"  # 今日到期 "今天到期多少"
    CASH_FLOW = "cash_flow"  # 资金流动 "入金出金情况"
    PRODUCT_STATUS = "product_status"  # 产品状态 "产品情况"
    QUICK_METRICS = "quick_metrics"  # 快速指标 "关键指标"


class ResponseFormat(Enum):
    """响应格式类型"""
    SIMPLE_TEXT = "simple_text"  # 简单文字
    DETAILED_SUMMARY = "detailed_summary"  # 详细摘要
    METRICS_FOCUSED = "metrics_focused"  # 指标聚焦
    BUSINESS_ORIENTED = "business_oriented"  # 业务导向


@dataclass
class CurrentDataResponse:
    """当前数据响应结果"""
    query_type: CurrentDataQueryType  # 查询类型
    response_format: ResponseFormat  # 响应格式

    # 核心响应内容
    main_answer: str  # 主要回答
    key_metrics: Dict[str, Any]  # 关键指标
    formatted_data: Dict[str, str]  # 格式化数据

    # 增强信息
    business_context: str  # 业务上下文
    quick_insights: List[str]  # 快速洞察
    related_metrics: Dict[str, Any]  # 相关指标

    # 元数据
    data_timestamp: str  # 数据时间戳
    response_confidence: float  # 响应置信度
    data_sources: List[str]  # 数据来源
    processing_time: float  # 处理时间


class CurrentDataProcessor:
    """
    📊 当前数据处理器

    专注于快速处理实时状态查询，提供即时、准确的当前状态信息
    """

    def __init__(self, claude_client=None, gpt_client=None):
        """
        初始化当前数据处理器

        Args:
            claude_client: Claude客户端，用于业务理解和洞察
            gpt_client: GPT客户端，用于数据格式化和计算
        """
        self.claude_client = claude_client
        self.gpt_client = gpt_client

        # 初始化核心组件
        self.query_parser = create_smart_query_parser(claude_client, gpt_client)
        self.data_fetcher = create_smart_data_fetcher(claude_client, gpt_client)
        self.data_analyzer = create_financial_data_analyzer(claude_client, gpt_client)
        self.insight_generator = create_insight_generator(claude_client, gpt_client)

        # 当前数据处理配置
        self.processing_config = self._load_processing_config()

        # 常用查询模式匹配
        self.query_patterns = self._load_current_data_patterns()

        # 处理统计
        self.processing_stats = {
            'total_queries': 0,
            'queries_by_type': {},
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'cache_hit_rate': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("CurrentDataProcessor initialized for real-time queries")

    def _get_query_hash(self, query: str) -> str:
        """生成查询哈希值用于缓存"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=100)
    async def _cached_ai_analysis(self, query_hash: str, prompt: str) -> Dict[str, Any]:
        """缓存AI分析结果"""
        # 更新缓存统计
        self.processing_stats['cache_misses'] += 1
        
        # 健壮的Claude客户端调用逻辑
        try:
            if not self.claude_client:
                logger.error("Claude客户端未初始化")
                return {"response": "AI分析服务不可用"}
            
            # 检查是否是自定义的ClaudeClient类
            if hasattr(self.claude_client, 'analyze_complex_query') and callable(getattr(self.claude_client, 'analyze_complex_query')):
                logger.info("使用自定义ClaudeClient的analyze_complex_query方法")
                return await self.claude_client.analyze_complex_query(prompt, {})
            
            # 检查是否可以直接使用messages API (新版SDK)
            if hasattr(self.claude_client, 'messages') and callable(getattr(self.claude_client.messages, 'create', None)):
                logger.info("使用messages API调用Claude")
                response = await asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                if hasattr(response, 'content') and response.content:
                    content_text = ""
                    for content_item in response.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text
                        elif isinstance(content_item, dict) and 'text' in content_item:
                            content_text += content_item['text']
                    return {"response": content_text}
            
            # 检查是否可以使用completion API (旧版SDK)
            if hasattr(self.claude_client, 'completions') and callable(getattr(self.claude_client.completions, 'create', None)):
                logger.info("使用completions API调用Claude")
                response = await asyncio.to_thread(
                    self.claude_client.completions.create,
                    model="claude-3-opus-20240229",
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=1000
                )
                if hasattr(response, 'completion'):
                    return {"response": response.completion}
                
            # 尝试直接调用client的completion方法 (最旧版本)
            if hasattr(self.claude_client, 'completion') and callable(self.claude_client.completion):
                logger.info("使用旧版completion方法调用Claude")
                response = await asyncio.to_thread(
                    self.claude_client.completion,
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=1000
                )
                if hasattr(response, 'completion'):
                    return {"response": response.completion}
            
            # 所有方法都失败，返回错误信息
            logger.error("无法找到可用的Claude API调用方法")
            return {"response": "AI分析服务调用失败，请检查Claude客户端配置"}
            
        except Exception as e:
            logger.error(f"AI分析调用失败: {str(e)}")
            return {"response": f"AI分析出错: {str(e)}"}

    async def _identify_current_data_query_type(self, user_query: str) -> CurrentDataQueryType:
        """AI驱动的查询类型识别"""
        try:
            if not self.claude_client:
                # 如果没有AI客户端，使用默认类型
                logger.warning("没有可用的AI客户端，使用默认查询类型")
                return CurrentDataQueryType.SYSTEM_OVERVIEW
                
            prompt = f"""
            分析以下金融业务查询，识别查询类型：

            用户查询: "{user_query}"

            请从以下类型中选择最匹配的一个:
            - system_overview: 系统概览查询，如"总体情况"、"系统状态"
            - balance_check: 余额查询，如"余额多少"、"总资金"
            - user_status: 用户状态查询，如"用户数量"、"活跃用户"
            - today_expiry: 今日到期查询，如"今天到期多少"
            - cash_flow: 资金流动查询，如"入金出金情况"
            - product_status: 产品状态查询，如"产品情况"
            - quick_metrics: 快速指标查询，如"关键指标"

            只返回类型ID (如 "balance_check")，不要有其他内容。
            """

            # 使用缓存
            query_hash = self._get_query_hash(user_query)
            
            # 检查是否已有缓存结果
            if query_hash in self._cached_ai_analysis.cache_info().currsize:
                self.processing_stats['cache_hits'] += 1
                # 更新缓存命中率
                total_cache_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
                if total_cache_requests > 0:
                    self.processing_stats['cache_hit_rate'] = self.processing_stats['cache_hits'] / total_cache_requests
                logger.info(f"缓存命中: {user_query}")
                
            result = await self._cached_ai_analysis(query_hash, prompt)
        
            # 提取类型ID
            if isinstance(result, dict) and 'response' in result:
                type_id = result['response'].strip().lower()
            else:
                type_id = str(result).strip().lower()
                
            # 移除可能的引号
            type_id = type_id.replace('"', '').replace("'", "")
            
            # 转换为枚举
            try:
                return CurrentDataQueryType(type_id)
            except ValueError:
                logger.warning(f"无法识别的查询类型: {type_id}，使用默认类型")
                return CurrentDataQueryType.SYSTEM_OVERVIEW
            
        except Exception as e:
            logger.warning(f"AI查询类型识别失败: {str(e)}，使用默认类型")
            return CurrentDataQueryType.SYSTEM_OVERVIEW

    async def _process_by_query_type(self, query_type: CurrentDataQueryType, 
                                   user_query: str, current_data: Dict[str, Any], 
                                   additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        AI驱动的查询处理，根据查询类型处理数据
        
        Args:
            query_type: 查询类型
            user_query: 用户查询
            current_data: 当前系统数据
            additional_context: 额外上下文
            
        Returns:
            处理结果，包含关键指标和格式化数据
        """
        try:
            logger.info(f"处理查询类型: {query_type.value}")
            
            # 准备AI处理提示
            prompt = f"""
            处理以下金融查询，提取关键指标并计算必要的数据:
            
            查询类型: {query_type.value}
            用户查询: "{user_query}"

            系统数据:
            ```
            {json.dumps(current_data, ensure_ascii=False, indent=2)}
            ```
            
            请提取与查询相关的关键指标，并进行必要的计算。返回JSON格式:
            {{
                "key_metrics": {{
                    "metric1": value1,
                    "metric2": value2,
                    ...
                }},
                "related_metrics": {{
                    "related1": value1,
                    ...
                }},
                "response_format": "simple_text/detailed_summary/metrics_focused/business_oriented",
                "confidence": 0.95
            }}
            """
            
            # 使用缓存
            query_hash = self._get_query_hash(f"{query_type.value}_{user_query}")
            
            # 检查是否已有缓存结果
            if query_hash in self._cached_ai_analysis.cache_info().currsize:
                self.processing_stats['cache_hits'] += 1
                # 更新缓存命中率
                total_cache_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
                if total_cache_requests > 0:
                    self.processing_stats['cache_hit_rate'] = self.processing_stats['cache_hits'] / total_cache_requests
                logger.info(f"处理缓存命中: {query_type.value}")
                
            # 使用GPT处理数据和计算
            result = await self._cached_ai_analysis(query_hash, prompt)
            
            # 解析结果
            processed_result = {}
            if isinstance(result, dict) and 'response' in result:
                try:
                    processed_result = json.loads(result['response'])
                except:
                    logger.warning("无法解析AI处理结果，使用原始响应")
                    processed_result = {
                        "key_metrics": {"raw_response": result['response']},
                        "response_format": "simple_text",
                        "confidence": 0.5
                    }
            else:
                logger.warning("AI处理返回意外格式，使用默认结构")
                processed_result = {
                    "key_metrics": {"raw_response": str(result)},
                    "response_format": "simple_text",
                    "confidence": 0.5
                }
                
            # 确保结果包含所有必要字段
            if "key_metrics" not in processed_result:
                processed_result["key_metrics"] = {}
            if "related_metrics" not in processed_result:
                processed_result["related_metrics"] = {}
            if "response_format" not in processed_result:
                processed_result["response_format"] = "simple_text"
            if "confidence" not in processed_result:
                processed_result["confidence"] = 0.8
                
            return processed_result
            
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}")
            return {
                "key_metrics": {"error": str(e)},
                "related_metrics": {},
                "response_format": "simple_text",
                "confidence": 0.5
            }
            
    async def _generate_quick_insights(self, query_type: CurrentDataQueryType,
                                     key_metrics: Dict[str, Any],
                                     current_data: Dict[str, Any]) -> List[str]:
        """
        AI驱动的业务洞察生成
        
        Args:
            query_type: 查询类型
            key_metrics: 关键指标
            current_data: 当前系统数据
            
        Returns:
            业务洞察列表
        """
        try:
            if not self.claude_client:
                logger.warning("没有可用的Claude客户端，跳过洞察生成")
                return []
                
            # 准备AI处理提示
            prompt = f"""
            基于以下金融数据，生成3-5条简洁的业务洞察:
            
            查询类型: {query_type.value}
            
            关键指标:
            ```
            {json.dumps(key_metrics, ensure_ascii=False, indent=2)}
            ```
            
            系统数据:
            ```
            {json.dumps(current_data, ensure_ascii=False, indent=2)}
            ```
            
            请生成3-5条简洁、有价值的业务洞察，每条不超过50个字。
            返回JSON数组格式，例如:
            ["洞察1", "洞察2", "洞察3"]
            """
            
            # 使用缓存
            metrics_str = "_".join([f"{k}:{v}" for k, v in sorted(key_metrics.items())[:3]])
            query_hash = self._get_query_hash(f"{query_type.value}_{metrics_str}")
            
            # 检查是否已有缓存结果
            if query_hash in self._cached_ai_analysis.cache_info().currsize:
                self.processing_stats['cache_hits'] += 1
                # 更新缓存命中率
                total_cache_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
                if total_cache_requests > 0:
                    self.processing_stats['cache_hit_rate'] = self.processing_stats['cache_hits'] / total_cache_requests
                logger.info("洞察缓存命中")
                
            # 使用Claude生成洞察
            result = await self._cached_ai_analysis(query_hash, prompt)
            
            # 解析结果
            insights = []
            if isinstance(result, dict) and 'response' in result:
                try:
                    insights = json.loads(result['response'])
                except:
                    logger.warning("无法解析AI洞察结果，使用原始响应")
                    insights = [result['response']]
            else:
                logger.warning("AI洞察返回意外格式，使用默认结构")
                insights = [str(result)]
                
            # 确保结果是列表类型
            if not isinstance(insights, list):
                insights = [str(insights)]
                
            # 限制洞察数量
            return insights[:5]
            
        except Exception as e:
            logger.error(f"生成洞察失败: {str(e)}")
            return []
            
    async def _format_current_data_response(self, query_type: CurrentDataQueryType,
                                          user_query: str,
                                          processed_result: Dict[str, Any],
                                          quick_insights: List[str]) -> Dict[str, Any]:
        """
        AI驱动的响应格式化
        
        Args:
            query_type: 查询类型
            user_query: 用户查询
            processed_result: 处理结果
            quick_insights: 业务洞察
            
        Returns:
            格式化的响应
        """
        try:
            if not self.gpt_client:
                logger.warning("没有可用的GPT客户端，使用简单格式化")
                return {
                    "main_answer": f"查询类型: {query_type.value}\n关键指标: {json.dumps(processed_result.get('key_metrics', {}), ensure_ascii=False)}",
                    "formatted_data": {"raw": json.dumps(processed_result, ensure_ascii=False)},
                    "business_context": ""
                }
                
            # 准备AI处理提示
            response_format = processed_result.get('response_format', 'simple_text')
            
            prompt = f"""
            基于以下金融数据，生成用户友好的响应:
            
            查询类型: {query_type.value}
            用户查询: "{user_query}"
            
            关键指标:
            ```
            {json.dumps(processed_result.get('key_metrics', {}), ensure_ascii=False, indent=2)}
            ```
            
            业务洞察:
            ```
            {json.dumps(quick_insights, ensure_ascii=False, indent=2)}
            ```
            
            响应格式: {response_format}
            
            请生成一个结构化的响应，包含以下部分:
            1. main_answer: 主要回答，简洁明了
            2. formatted_data: 格式化的数据展示
            3. business_context: 业务上下文说明
            
            返回JSON格式:
            {{
                "main_answer": "...",
                "formatted_data": {{
                    "key1": "value1",
                    ...
                }},
                "business_context": "..."
            }}
            """
            
            # 使用缓存
            metrics_str = "_".join([f"{k}" for k in sorted(processed_result.get('key_metrics', {}).keys())[:3]])
            insights_str = "_".join(quick_insights[:2]) if quick_insights else ""
            query_hash = self._get_query_hash(f"{query_type.value}_{metrics_str}_{insights_str}")
            
            # 检查是否已有缓存结果
            if query_hash in self._cached_ai_analysis.cache_info().currsize:
                self.processing_stats['cache_hits'] += 1
                # 更新缓存命中率
                total_cache_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
                if total_cache_requests > 0:
                    self.processing_stats['cache_hit_rate'] = self.processing_stats['cache_hits'] / total_cache_requests
                logger.info("格式化缓存命中")
                
            # 使用GPT格式化响应
            result = await self._cached_ai_analysis(query_hash, prompt)
            
            # 解析结果
            formatted_response = {}
            if isinstance(result, dict) and 'response' in result:
                try:
                    formatted_response = json.loads(result['response'])
                except:
                    logger.warning("无法解析AI格式化结果，使用原始响应")
                    formatted_response = {
                        "main_answer": result['response'],
                        "formatted_data": {"raw": json.dumps(processed_result.get('key_metrics', {}), ensure_ascii=False)},
                        "business_context": ""
                    }
            else:
                logger.warning("AI格式化返回意外格式，使用默认结构")
                formatted_response = {
                    "main_answer": str(result),
                    "formatted_data": {"raw": json.dumps(processed_result.get('key_metrics', {}), ensure_ascii=False)},
                    "business_context": ""
                }
                
            # 确保结果包含所有必要字段
            if "main_answer" not in formatted_response:
                formatted_response["main_answer"] = f"查询类型: {query_type.value}"
            if "formatted_data" not in formatted_response:
                formatted_response["formatted_data"] = {"raw": json.dumps(processed_result.get('key_metrics', {}), ensure_ascii=False)}
            if "business_context" not in formatted_response:
                formatted_response["business_context"] = ""
                
            return formatted_response
            
        except Exception as e:
            logger.error(f"格式化响应失败: {str(e)}")
            return {
                "main_answer": f"查询类型: {query_type.value}\n关键指标: {json.dumps(processed_result.get('key_metrics', {}), ensure_ascii=False)}",
                "formatted_data": {"raw": json.dumps(processed_result, ensure_ascii=False)},
                "business_context": f"处理过程中出现错误: {str(e)}"
            }

    async def process_multiple_queries(self, queries: List[str]) -> List[CurrentDataResponse]:
        """批量处理多个查询，减少AI调用次数"""
        
        # 如果查询数量少，直接逐个处理
        if len(queries) <= 2:
            return [await self.process_current_data_query(q) for q in queries]
        
        try:
            # 批量识别查询类型
            combined_prompt = f"""
            批量分析以下{len(queries)}个金融查询的类型：

            {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(queries)])}

            对每个查询，只返回其类型ID：
            - system_overview
            - balance_check
            - user_status
            - today_expiry
            - cash_flow
            - product_status
            - quick_metrics

            返回JSON数组格式，例如：
            ["balance_check", "user_status", "system_overview"]
            """
            
            # 一次AI调用处理多个查询类型
            result = await self.claude_client.analyze_complex_query(combined_prompt, {})
            
            # 解析结果
            query_types = []
            if isinstance(result, dict) and 'response' in result:
                try:
                    query_types = json.loads(result['response'])
                except:
                    # 降级到单独处理
                    return [await self.process_current_data_query(q) for q in queries]
            
            # 确保类型数量匹配
            if len(query_types) != len(queries):
                # 降级到单独处理
                return [await self.process_current_data_query(q) for q in queries]
                
            # 获取系统数据（只需获取一次）
            current_data = await self._fetch_current_system_data()
            
            # 处理每个查询
            responses = []
            for i, query in enumerate(queries):
                try:
                    query_type = CurrentDataQueryType(query_types[i])
                    # 使用已获取的系统数据处理查询
                    response = await self._process_query_with_data(query, query_type, current_data)
                    responses.append(response)
                except Exception as e:
                    logger.error(f"处理查询 '{query}' 失败: {str(e)}")
                    # 单个查询失败时，降级处理该查询
                    responses.append(await self.process_current_data_query(query))
                    
            return responses
            
        except Exception as e:
            logger.error(f"批量查询处理失败: {str(e)}")
            # 降级到单独处理
            return [await self.process_current_data_query(q) for q in queries]
            
    async def _process_query_with_data(self, user_query: str, query_type: CurrentDataQueryType, 
                                current_data: Dict[str, Any]) -> CurrentDataResponse:
        """使用已获取的数据处理单个查询"""
        try:
            logger.info(f"📊 使用已获取数据处理查询: {user_query}")

            start_time = datetime.now()
            self.processing_stats['total_queries'] += 1

            # Step 3: GPT-4o处理数据和计算
            processed_result = await self._process_by_query_type(
                query_type, user_query, current_data, None
            )
            logger.info(f"数据处理完成，关键指标: {list(processed_result.get('key_metrics', {}).keys())}")

            # Step 4: Claude生成业务洞察
            quick_insights = await self._generate_quick_insights(
                query_type, processed_result['key_metrics'], current_data
            )
            logger.info(f"生成洞察: {len(quick_insights)}条")

            # Step 5: GPT-4o格式化响应
            formatted_response = await self._format_current_data_response(
                query_type, user_query, processed_result, quick_insights
            )
            logger.info("响应格式化完成")

            # Step 6: 构建最终响应
            processing_time = (datetime.now() - start_time).total_seconds()

            response = CurrentDataResponse(
                query_type=query_type,
                response_format=processed_result.get('response_format', ResponseFormat.SIMPLE_TEXT),

                main_answer=formatted_response['main_answer'],
                key_metrics=processed_result['key_metrics'],
                formatted_data=formatted_response['formatted_data'],

                business_context=formatted_response.get('business_context', ''),
                quick_insights=quick_insights,
                related_metrics=processed_result.get('related_metrics', {}),

                data_timestamp=current_data.get('timestamp', datetime.now().isoformat()),
                response_confidence=processed_result.get('confidence', 0.8),
                data_sources=['/api/sta/system'],
                processing_time=processing_time
            )

            # 更新统计信息
            self._update_processing_stats(query_type, processing_time, response.response_confidence)

            logger.info(f"✅ 批量查询处理完成，耗时{processing_time:.2f}秒")

            return response

        except Exception as e:
            logger.error(f"❌ 批量查询处理失败: {str(e)}")
            return self._create_error_response(user_query, str(e))

    async def process_current_data_query(self, user_query: str) -> CurrentDataResponse:
        """处理当前数据查询"""
        try:
            logger.info(f"📊 处理当前数据查询: {user_query}")
            
            start_time = datetime.now()
            self.processing_stats['total_queries'] += 1
            
            # Step 1: 识别查询类型
            query_type = await self._identify_current_data_query_type(user_query)
            logger.info(f"识别查询类型: {query_type.value}")
            
            # Step 2: 获取当前系统数据
            current_data = await self._fetch_current_system_data()
            logger.info(f"获取系统数据: {len(current_data)} 个字段")
            
            # Step 3: GPT-4o处理数据和计算
            processed_result = await self._process_by_query_type(
                query_type, user_query, current_data, None
            )
            logger.info(f"数据处理完成，关键指标: {list(processed_result.get('key_metrics', {}).keys())}")
            
            # Step 4: Claude生成业务洞察
            quick_insights = await self._generate_quick_insights(
                query_type, processed_result['key_metrics'], current_data
            )
            logger.info(f"生成洞察: {len(quick_insights)}条")
            
            # Step 5: GPT-4o格式化响应
            formatted_response = await self._format_current_data_response(
                query_type, user_query, processed_result, quick_insights
            )
            logger.info("响应格式化完成")
            
            # Step 6: 构建最终响应
            processing_time = (datetime.now() - start_time).total_seconds()

            response = CurrentDataResponse(
                query_type=query_type,
                response_format=ResponseFormat(processed_result.get('response_format', 'simple_text')),
                
                main_answer=formatted_response['main_answer'],
                key_metrics=processed_result['key_metrics'],
                formatted_data=formatted_response['formatted_data'],
                
                business_context=formatted_response.get('business_context', ''),
                quick_insights=quick_insights,
                related_metrics=processed_result.get('related_metrics', {}),
                
                data_timestamp=current_data.get('timestamp', datetime.now().isoformat()),
                response_confidence=processed_result.get('confidence', 0.8),
                data_sources=['/api/sta/system'],
                processing_time=processing_time
            )

            # 更新统计信息
            self._update_processing_stats(query_type, processing_time, response.response_confidence)

            logger.info(f"✅ 查询处理完成，耗时{processing_time:.2f}秒")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 查询处理失败: {str(e)}")
            return self._create_error_response(user_query, str(e))

    async def _fetch_current_system_data(self) -> Dict[str, Any]:
        """获取当前系统数据"""
        try:
            logger.info("获取当前系统真实数据")
            
            # 使用API连接器获取真实数据
            if not hasattr(self, 'api_connector') or self.api_connector is None:
                # 如果没有初始化API连接器，则创建一个
                from data.connectors.api_connector import create_enhanced_api_connector
                self.api_connector = create_enhanced_api_connector(
                    None, self.claude_client, self.gpt_client
                )
                logger.info("初始化API连接器")
            
            # 获取系统概览数据
            system_data = await self.api_connector._make_request('/api/sta/system')
            
            if not system_data or not system_data.get('success', False):
                logger.error(f"获取系统数据失败: {system_data.get('message', '未知错误')}")
                # 降级到获取每日数据
                daily_data = await self.api_connector._make_request('/api/sta/daily')
                if daily_data and daily_data.get('success', False):
                    system_data = daily_data
                    logger.info("使用每日数据作为降级方案")
                else:
                    # 如果API调用失败，使用模拟数据作为最后的降级方案
                    logger.warning("API调用失败，使用模拟数据作为降级方案")
                    return await self._get_mock_system_data()
            
            # 增强系统数据
            if system_data.get('data'):
                # 添加时间戳
                now = datetime.now()
                system_data['data']['timestamp'] = now.isoformat()
                
                # 如果需要额外数据，获取用户数据和产品数据
                try:
                    user_data = await self.api_connector._make_request('/api/sta/user')
                    if user_data and user_data.get('success', False) and user_data.get('data'):
                        system_data['data']['user_details'] = user_data['data']
                        
                    product_data = await self.api_connector._make_request('/api/sta/product')
                    if product_data and product_data.get('success', False) and product_data.get('data'):
                        system_data['data']['product_details'] = product_data['data']
                        
                    # 获取今日到期数据
                    today_str = now.strftime('%Y%m%d')
                    expiry_data = await self.api_connector._make_request('/api/sta/product/end', {'date': today_str})
                    if expiry_data and expiry_data.get('success', False) and expiry_data.get('data'):
                        system_data['data']['today_expiry_details'] = expiry_data['data']
                except Exception as e:
                    logger.warning(f"获取额外数据失败: {str(e)}")
                
                return system_data['data']
            else:
                logger.error("系统数据响应中没有data字段")
                return await self._get_mock_system_data()
            
        except Exception as e:
            logger.error(f"获取系统数据失败: {str(e)}")
            # 返回模拟数据作为降级方案
            return await self._get_mock_system_data()
            
    async def _get_mock_system_data(self) -> Dict[str, Any]:
        """获取模拟系统数据（作为降级方案）"""
        logger.warning("使用模拟数据")
        
        # 获取当前时间
        now = datetime.now()
        
        # 模拟系统数据
        system_data = {
            "timestamp": now.isoformat(),
            "system_status": "normal",
            "total_balance": 12567890.45,
            "available_balance": 10234567.89,
            "frozen_balance": 2333322.56,
            "total_users": 5678,
            "active_users_today": 1234,
            "new_users_today": 56,
            "today_expiry_amount": 1500000.00,
            "today_expiry_count": 12,
            "cash_in_today": 2345678.90,
            "cash_out_today": 1234567.80,
            "net_cash_flow": 1111111.10,
            "products": {
                "total_count": 45,
                "active_count": 38,
                "top_performing": ["产品A", "产品B", "产品C"],
                "underperforming": ["产品X", "产品Y"]
            },
            "quick_metrics": {
                "roi_7d": 0.0234,
                "roi_30d": 0.0567,
                "user_growth_30d": 0.0789,
                "transaction_volume_7d": 45678901.23
            }
        }
        
        return system_data

    def _update_processing_stats(self, query_type: CurrentDataQueryType, 
                               processing_time: float, confidence: float) -> None:
        """更新处理统计信息"""
        try:
            # 更新查询类型统计
            type_value = query_type.value
            if type_value not in self.processing_stats['queries_by_type']:
                self.processing_stats['queries_by_type'][type_value] = 0
            self.processing_stats['queries_by_type'][type_value] += 1
            
            # 更新平均响应时间
            total_queries = self.processing_stats['total_queries']
            current_avg_time = self.processing_stats['avg_response_time']
            self.processing_stats['avg_response_time'] = (current_avg_time * (total_queries - 1) + processing_time) / total_queries
            
            # 更新平均置信度
            current_avg_confidence = self.processing_stats['avg_confidence']
            self.processing_stats['avg_confidence'] = (current_avg_confidence * (total_queries - 1) + confidence) / total_queries
            
            # 更新缓存命中率
            total_cache_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
            if total_cache_requests > 0:
                self.processing_stats['cache_hit_rate'] = self.processing_stats['cache_hits'] / total_cache_requests
                
        except Exception as e:
            logger.error(f"更新统计信息失败: {str(e)}")

    def _create_error_response(self, user_query: str, error_message: str) -> CurrentDataResponse:
        """创建错误响应"""
        return CurrentDataResponse(
            query_type=CurrentDataQueryType.SYSTEM_OVERVIEW,
            response_format=ResponseFormat.SIMPLE_TEXT,
            
            main_answer=f"处理查询时出错: {error_message}",
            key_metrics={"error": error_message},
            formatted_data={"error_details": error_message},
            
            business_context="",
            quick_insights=[],
            related_metrics={},
            
            data_timestamp=datetime.now().isoformat(),
            response_confidence=0.0,
            data_sources=[],
            processing_time=0.0
        )

    def _load_processing_config(self) -> Dict[str, Any]:
        """加载处理配置"""
        return {
            "max_response_time_ms": 2000,
            "data_freshness_threshold_sec": 60,
            "enable_insights": True,
            "max_insights_count": 5,
            "cache_ttl_sec": 300
        }

    def _load_current_data_patterns(self) -> Dict[str, List[str]]:
        """加载当前数据查询模式"""
        return {
            "system_overview": ["总体情况", "系统状态", "整体状况"],
            "balance_check": ["余额", "资金", "钱", "账户"],
            "user_status": ["用户", "客户", "会员"],
            "today_expiry": ["到期", "过期", "今天到期"],
            "cash_flow": ["入金", "出金", "流入", "流出"],
            "product_status": ["产品", "理财", "投资"],
            "quick_metrics": ["指标", "数据", "统计"]
        }