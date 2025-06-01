# core/analyzers/ai_strategy_extractor.py (重构版)
import logging
import json
import re
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.detectors.query_type_detector import QueryTypeDetector, QueryType, QueryTypeResult
from core.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

@dataclass
class ExtractedStrategy:
    """提取的API策略结果"""
    success: bool
    query_analysis: Dict[str, Any]
    api_calls: List[Dict[str, Any]]
    query_type_info: Optional[QueryTypeResult] = None
    processing_time: float = 0.0
    extraction_method: str = "ai_enhanced"
    confidence: float = 0.0
    error_message: Optional[str] = None

class EnhancedAPIStrategyExtractor:
    """增强版API策略提取器 - 集成查询类型检测和提示词管理"""
    
    def __init__(self, claude_client, query_type_detector: Optional[QueryTypeDetector] = None,
                 prompt_manager: Optional[PromptManager] = None):
        self.claude_client = claude_client
        self.query_type_detector = query_type_detector or QueryTypeDetector()
        self.prompt_manager = prompt_manager or PromptManager()
        
        # 统计信息
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'query_type_distribution': {},
            'average_processing_time': 0.0
        }
        
        logger.info("增强版API策略提取器初始化完成")
    
    async def extract_strategy(self, user_query: str, 
                             context: Optional[Dict[str, Any]] = None) -> ExtractedStrategy:
        """
        增强版策略提取 - 主入口方法
        
        Args:
            user_query: 用户查询
            context: 上下文信息
            
        Returns:
            ExtractedStrategy: 提取结果
        """
        start_time = time.time()
        self.stats['total_extractions'] += 1
        
        logger.info(f"🧠 开始增强版策略提取: {user_query[:50]}...")
        
        try:
            # 步骤1: 检测查询类型
            logger.debug("🔍 执行查询类型检测...")
            query_type_result = self.query_type_detector.detect(user_query)
            
            logger.info(f"✅ 检测到查询类型: {query_type_result.type.value}, 置信度: {query_type_result.confidence:.2f}")
            
            # 更新统计
            query_type = query_type_result.type.value
            if query_type not in self.stats['query_type_distribution']:
                self.stats['query_type_distribution'][query_type] = 0
            self.stats['query_type_distribution'][query_type] += 1
            
            # 步骤2: 生成增强prompt
            logger.debug("📝 构建增强版意图分析prompt...")
            enhanced_prompt = self.prompt_manager.build_intent_analysis_prompt(
                user_query=user_query,
                query_type_result=query_type_result,
                quick_decision=context.get('quick_decision') if context else None
            )
            
            # 步骤3: 调用Claude进行策略分析
            logger.debug("🤖 调用Claude执行增强策略分析...")
            claude_result = await self._call_claude_analysis(enhanced_prompt, query_type_result)
            
            if not claude_result.get('success'):
                logger.warning("Claude分析失败，使用智能降级")
                return self._create_fallback_strategy(user_query, query_type_result, 
                                                    claude_result.get('error'), time.time() - start_time)
            
            # 步骤4: 处理和验证结果
            strategy_data = claude_result.get('analysis', {})
            processed_result = self._process_claude_result(strategy_data, query_type_result, user_query)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            processed_result.processing_time = processing_time
            processed_result.query_type_info = query_type_result
            
            # 更新统计
            if processed_result.success:
                self.stats['successful_extractions'] += 1
            else:
                self.stats['failed_extractions'] += 1
            
            self._update_average_processing_time(processing_time)
            
            logger.info(f"✅ 策略提取完成: {len(processed_result.api_calls)}个API调用, 耗时: {processing_time:.2f}s")
            return processed_result
            
        except Exception as e:
            logger.error(f"❌ 策略提取异常: {str(e)}")
            self.stats['failed_extractions'] += 1
            processing_time = time.time() - start_time
            
            return self._create_fallback_strategy(user_query, query_type_result, str(e), processing_time)
    
    async def _call_claude_analysis(self, prompt: str, 
                                  query_type_result: QueryTypeResult) -> Dict[str, Any]:
        """调用Claude执行分析"""
        
        if not self.claude_client:
            return {'success': False, 'error': 'Claude客户端不可用'}
        
        try:
            # 根据查询类型调整超时时间
            timeout = 30 if query_type_result.type == QueryType.SIMPLE_DATA else 45
            
            result = await asyncio.wait_for(
                self.claude_client.generate_text(prompt, max_tokens=5000),
                timeout=timeout
            )
            
            if result.get('success'):
                response_text = result.get('text', '{}')
                
                # 解析JSON响应
                analysis = self._parse_json_response(response_text)
                
                if analysis:
                    return {
                        'success': True,
                        'analysis': analysis,
                        'raw_response': response_text[:500]
                    }
                else:
                    return {
                        'success': False,
                        'error': 'JSON解析失败',
                        'raw_response': response_text[:200]
                    }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Claude调用失败')
                }
                
        except asyncio.TimeoutError:
            logger.error("Claude调用超时")
            return {'success': False, 'error': 'Claude响应超时'}
        except Exception as e:
            logger.error(f"Claude调用异常: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """解析Claude的JSON响应"""
        try:
            # 直接解析
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            try:
                # 提取代码块中的JSON
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # 提取大括号中的内容
                brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if brace_match:
                    return json.loads(brace_match.group())
                    
            except json.JSONDecodeError:
                pass
        
        logger.error(f"无法解析Claude响应为JSON: {response_text[:300]}")
        return None
    
    def _process_claude_result(self, strategy_data: Dict[str, Any], 
                             query_type_result: QueryTypeResult,
                             user_query: str) -> ExtractedStrategy:
        """处理Claude分析结果"""
        
        try:
            # 提取查询分析
            query_analysis = {
                'intent': strategy_data.get('intent', '数据查询'),
                'complexity': strategy_data.get('complexity', 'simple'),
                'data_needed': strategy_data.get('data_needed', []),
                'calculation_required': strategy_data.get('calculation_required', False),
                'confidence': strategy_data.get('confidence', 0.8)
            }
            
            # 提取API调用策略
            api_strategy = strategy_data.get('api_strategy', [])
            
            # 🆕 根据查询类型验证和增强API策略
            enhanced_api_calls = self._enhance_api_strategy(api_strategy, query_type_result, user_query)
            
            # 验证API调用有效性
            validation_result = self._validate_api_calls(enhanced_api_calls)
            
            if not validation_result['is_valid']:
                logger.warning(f"API调用验证失败: {validation_result['issues']}")
                # 使用智能修复
                enhanced_api_calls = self._repair_api_calls(enhanced_api_calls, query_type_result)
            
            return ExtractedStrategy(
                success=True,
                query_analysis=query_analysis,
                api_calls=enhanced_api_calls,
                confidence=query_analysis.get('confidence', 0.8),
                extraction_method='ai_enhanced_with_type_detection'
            )
            
        except Exception as e:
            logger.error(f"处理Claude结果失败: {str(e)}")
            return ExtractedStrategy(
                success=False,
                query_analysis={},
                api_calls=[],
                error_message=str(e)
            )
    
    def _enhance_api_strategy(self, api_strategy: List[Dict[str, Any]], 
                            query_type_result: QueryTypeResult,
                            user_query: str) -> List[Dict[str, Any]]:
        """根据查询类型增强API策略"""
        
        enhanced_calls = []
        
        # 🎯 特殊查询类型的API策略增强
        if query_type_result.type == QueryType.REINVESTMENT:
            enhanced_calls = self._enhance_reinvestment_strategy(api_strategy, query_type_result)
            
        elif query_type_result.type == QueryType.COMPARISON:
            enhanced_calls = self._enhance_comparison_strategy(api_strategy, query_type_result, user_query)
            
        elif query_type_result.type == QueryType.PREDICTION:
            enhanced_calls = self._enhance_prediction_strategy(api_strategy, query_type_result)
            
        elif query_type_result.type == QueryType.HISTORICAL_REVIEW:
            enhanced_calls = self._enhance_historical_strategy(api_strategy, query_type_result)
            
        else:
            # 标准策略，直接使用Claude的结果
            enhanced_calls = api_strategy
        
        # 确保每个API调用都有必要的字段
        for i, call in enumerate(enhanced_calls):
            if 'sequence' not in call:
                call['sequence'] = i + 1
            if 'priority' not in call:
                call['priority'] = 1
            if 'time_period' not in call:
                call['time_period'] = 'general'
        
        return enhanced_calls
    
    def _enhance_reinvestment_strategy(self, api_strategy: List[Dict[str, Any]], 
                                     query_type_result: QueryTypeResult) -> List[Dict[str, Any]]:
        """增强复投查询的API策略"""
        
        enhanced_calls = []
        special_req = query_type_result.special_requirements
        
        # 确保包含必要的API调用
        has_product_end = any('product_end' in call.get('method', '') for call in api_strategy)
        has_system_data = any('system_data' in call.get('method', '') for call in api_strategy)
        
        # 添加Claude建议的API调用
        for call in api_strategy:
            enhanced_calls.append(call)
        
        # 如果缺少必要的API，补充
        if not has_product_end:
            enhanced_calls.append({
                'method': 'get_product_end_interval',
                'params': {},  # 日期参数将在日期识别阶段填充
                'reason': '获取产品到期数据，用于复投计算',
                'time_period': 'target_period'
            })
        
        if not has_system_data:
            enhanced_calls.append({
                'method': 'get_system_data',
                'params': {},
                'reason': '获取当前系统余额，用于复投后余额计算',
                'time_period': 'current'
            })
        
        return enhanced_calls

    def _enhance_comparison_strategy(self, api_strategy: List[Dict[str, Any]],
                                     query_type_result: QueryTypeResult,
                                     user_query: str) -> List[Dict[str, Any]]:
        """增强对比查询的API策略 - 修复版本"""

        logger.info(f"🔍 [DEBUG] 进入对比策略增强")
        logger.info(f"🔍 [DEBUG] 查询: {user_query}")
        logger.info(f"🔍 [DEBUG] Claude原始策略: {len(api_strategy)} 个API调用")

        # 检测对比类型
        if any(keyword in user_query.lower() for keyword in ['本周', '上周']):
            logger.info("🔍 [DEBUG] 检测到周对比，生成专门的API调用序列")
            weekly_calls = self._generate_weekly_comparison_calls()

            # 🔧 添加标记，表明这是完整的日期序列
            for call in weekly_calls:
                call['_complete_date_sequence'] = True  # 标记这是完整的日期序列

            logger.info(f"✅ 生成周对比API调用: {len(weekly_calls)} 个")
            return weekly_calls

        elif any(keyword in user_query.lower() for keyword in ['今天', '昨天']):
            logger.info("🔍 [DEBUG] 检测到日对比，生成日对比API调用")
            daily_calls = self._generate_daily_comparison_calls()

            # 🔧 添加标记
            for call in daily_calls:
                call['_complete_date_sequence'] = True

            return daily_calls
        else:
            logger.info("🔍 [DEBUG] 使用Claude建议的策略")
            return api_strategy

    def _generate_weekly_comparison_calls(self) -> List[Dict[str, Any]]:
        """生成周对比的API调用 - 调试版本"""
        calls = []

        # 本周7天
        current_date = datetime.now()
        logger.info(f"🔍 [DEBUG] 当前日期: {current_date}")

        days_since_monday = current_date.weekday()
        logger.info(f"🔍 [DEBUG] 距离周一的天数: {days_since_monday}")

        current_monday = current_date - timedelta(days=days_since_monday)
        logger.info(f"🔍 [DEBUG] 本周周一: {current_monday}")

        # 生成本周数据
        logger.info("🔍 [DEBUG] 生成本周API调用:")
        for i in range(7):
            day = current_monday + timedelta(days=i)
            date_str = day.strftime('%Y%m%d')

            call = {
                'method': 'get_daily_data',
                'params': {'date': date_str},
                'reason': f'获取本周{day.strftime("%m月%d日")}数据',
                'time_period': 'current_week'
            }
            calls.append(call)
            logger.info(f"🔍 [DEBUG]   本周第{i + 1}天: {date_str} ({day.strftime('%Y-%m-%d %A')})")

        # 上周7天
        last_monday = current_monday - timedelta(days=7)
        logger.info(f"🔍 [DEBUG] 上周周一: {last_monday}")

        # 生成上周数据
        logger.info("🔍 [DEBUG] 生成上周API调用:")
        for i in range(7):
            day = last_monday + timedelta(days=i)
            date_str = day.strftime('%Y%m%d')

            call = {
                'method': 'get_daily_data',
                'params': {'date': date_str},
                'reason': f'获取上周{day.strftime("%m月%d日")}数据，用于对比',
                'time_period': 'last_week'
            }
            calls.append(call)
            logger.info(f"🔍 [DEBUG]   上周第{i + 1}天: {date_str} ({day.strftime('%Y-%m-%d %A')})")

        logger.info(f"🔍 [DEBUG] 周对比总共生成 {len(calls)} 个API调用")
        return calls

    def _generate_daily_comparison_calls(self) -> List[Dict[str, Any]]:
        """生成日对比的API调用"""
        from datetime import timedelta
        
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)
        
        return [
            {
                'method': 'get_daily_data',
                'params': {'date': current_date.strftime('%Y%m%d')},
                'reason': '获取今日数据',
                'time_period': 'today'
            },
            {
                'method': 'get_daily_data', 
                'params': {'date': yesterday.strftime('%Y%m%d')},
                'reason': '获取昨日数据，用于对比',
                'time_period': 'yesterday'
            }
        ]
    
    def _enhance_prediction_strategy(self, api_strategy: List[Dict[str, Any]], 
                                   query_type_result: QueryTypeResult) -> List[Dict[str, Any]]:
        """增强预测查询的API策略"""
        
        enhanced_calls = api_strategy.copy()
        
        # 预测需要历史数据，确保有足够的数据点
        has_historical_data = any('daily_data' in call.get('method', '') for call in api_strategy)
        
        if not has_historical_data:
            # 添加最近30天的数据采样（选择代表性日期）
            current_date = datetime.now()
            for i in [1, 7, 14, 21, 30]:  # 选择代表性日期
                target_date = current_date - timedelta(days=i)
                enhanced_calls.append({
                    'method': 'get_daily_data',
                    'params': {'date': target_date.strftime('%Y%m%d')},
                    'reason': f'获取{i}天前数据，用于趋势分析',
                    'time_period': 'historical'
                })
        
        return enhanced_calls
    
    def _enhance_historical_strategy(self, api_strategy: List[Dict[str, Any]], 
                                   query_type_result: QueryTypeResult) -> List[Dict[str, Any]]:
        """增强历史回顾查询的API策略"""
        
        # 历史回顾需要更长的时间序列
        enhanced_calls = api_strategy.copy()
        
        # 添加更多历史数据点
        current_date = datetime.now()
        for weeks_ago in [1, 2, 3, 4]:  # 最近4周的数据
            target_date = current_date - timedelta(weeks=weeks_ago)
            enhanced_calls.append({
                'method': 'get_daily_data',
                'params': {'date': target_date.strftime('%Y%m%d')},
                'reason': f'获取{weeks_ago}周前数据，用于历史分析',
                'time_period': f'week_{weeks_ago}_ago'
            })
        
        return enhanced_calls
    
    def _validate_api_calls(self, api_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证API调用的有效性"""
        
        issues = []
        valid_methods = {
            'get_system_data', 'get_daily_data', 'get_product_data',
            'get_product_end_data', 'get_product_end_interval',
            'get_user_daily_data', 'get_user_data'
        }
        
        for i, call in enumerate(api_calls):
            method = call.get('method', '')
            params = call.get('params', {})
            
            # 检查方法名
            if method not in valid_methods:
                issues.append(f"API调用{i+1}: 无效的方法名 '{method}'")
            
            # 检查日期参数格式
            for date_param in ['date', 'start_date', 'end_date']:
                if date_param in params:
                    date_value = params[date_param]
                    if date_value and not re.match(r'^\d{8}$', str(date_value)):
                        issues.append(f"API调用{i+1}: 无效的日期格式 '{date_value}'")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_calls': len(api_calls)
        }
    
    def _repair_api_calls(self, api_calls: List[Dict[str, Any]], 
                         query_type_result: QueryTypeResult) -> List[Dict[str, Any]]:
        """修复无效的API调用"""
        
        repaired_calls = []
        valid_methods = {
            'get_system_data', 'get_daily_data', 'get_product_data',
            'get_product_end_data', 'get_product_end_interval', 
            'get_user_daily_data', 'get_user_data'
        }
        
        for call in api_calls:
            method = call.get('method', '')
            params = call.get('params', {}).copy()
            reason = call.get('reason', '数据获取')
            
            # 修复方法名
            if method not in valid_methods:
                # 根据原方法名和reason推断正确的方法
                if 'product' in method.lower() or '产品' in reason:
                    if 'end' in method.lower() or '到期' in reason:
                        method = 'get_product_end_data'
                    else:
                        method = 'get_product_data'
                elif 'user' in method.lower() or '用户' in reason:
                    method = 'get_user_daily_data'
                elif 'daily' in method.lower() or '日' in reason:
                    method = 'get_daily_data'
                else:
                    method = 'get_system_data'
            
            # 修复日期参数
            current_date = datetime.now().strftime('%Y%m%d')
            for date_param in ['date', 'start_date', 'end_date']:
                if date_param in params:
                    date_value = params[date_param]
                    if not date_value or not re.match(r'^\d{8}$', str(date_value)):
                        params[date_param] = current_date
            
            repaired_calls.append({
                'method': method,
                'params': params,
                'reason': reason,
                'sequence': call.get('sequence', len(repaired_calls) + 1),
                'priority': call.get('priority', 1),
                'time_period': call.get('time_period', 'general')
            })
        
        return repaired_calls
    
    def _create_fallback_strategy(self, user_query: str, 
                                query_type_result: Optional[QueryTypeResult],
                                error_msg: str, processing_time: float) -> ExtractedStrategy:
        """创建降级策略"""
        
        logger.info(f"🔄 创建降级策略: {error_msg}")
        
        # 根据查询类型生成基础API调用
        if query_type_result and query_type_result.type != QueryType.SIMPLE_DATA:
            api_calls = self._generate_fallback_calls_by_type(query_type_result.type)
        else:
            api_calls = [{'method': 'get_system_data', 'params': {}, 'reason': '降级：获取系统概览'}]
        
        return ExtractedStrategy(
            success=True,  # 降级策略仍算成功
            query_analysis={
                'intent': '数据查询（降级）',
                'complexity': 'simple',
                'calculation_required': False,
                'confidence': 0.6
            },
            api_calls=api_calls,
            query_type_info=query_type_result,
            processing_time=processing_time,
            extraction_method='fallback',
            error_message=error_msg
        )
    
    def _generate_fallback_calls_by_type(self, query_type: QueryType) -> List[Dict[str, Any]]:
        """根据查询类型生成降级API调用"""
        
        if query_type == QueryType.REINVESTMENT:
            return [
                {'method': 'get_system_data', 'params': {}, 'reason': '降级：获取系统数据'},
                {'method': 'get_product_end_data', 'params': {}, 'reason': '降级：获取今日到期产品'}
            ]
        elif query_type == QueryType.COMPARISON:
            current_date = datetime.now()
            yesterday = current_date - timedelta(days=1)
            return [
                {'method': 'get_daily_data', 'params': {'date': current_date.strftime('%Y%m%d')}, 'reason': '降级：获取今日数据'},
                {'method': 'get_daily_data', 'params': {'date': yesterday.strftime('%Y%m%d')}, 'reason': '降级：获取昨日数据对比'}
            ]
        elif query_type == QueryType.PREDICTION:
            return [
                {'method': 'get_system_data', 'params': {}, 'reason': '降级：获取当前状态'},
                {'method': 'get_daily_data', 'params': {}, 'reason': '降级：获取最新数据'}
            ]
        else:
            return [{'method': 'get_system_data', 'params': {}, 'reason': '降级：通用数据获取'}]
    
    def _update_average_processing_time(self, new_time: float):
        """更新平均处理时间"""
        current_avg = self.stats['average_processing_time']
        total_count = self.stats['total_extractions']
        
        if total_count == 1:
            self.stats['average_processing_time'] = new_time
        else:
            self.stats['average_processing_time'] = (current_avg * (total_count - 1) + new_time) / total_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats['total_extractions']
        return {
            **self.stats,
            'success_rate': (self.stats['successful_extractions'] / total) if total > 0 else 0.0,
            'failure_rate': (self.stats['failed_extractions'] / total) if total > 0 else 0.0
        }

# 工厂函数
def create_enhanced_strategy_extractor(claude_client, 
                                     query_type_detector: Optional[QueryTypeDetector] = None,
                                     prompt_manager: Optional[PromptManager] = None) -> EnhancedAPIStrategyExtractor:
    """创建增强版API策略提取器"""
    return EnhancedAPIStrategyExtractor(claude_client, query_type_detector, prompt_manager)