# data/connectors/api_connector.py - 增强版
"""
🚀 AI驱动的智能API连接器 - 增强版
在原有功能基础上新增：
- 第8个API支持 (产品区间到期数据)
- 智能查询类型识别和API组合选择
- AI数据验证和预处理
- 为分析优化的数据格式转换

增强特点：
- 🧠 AI驱动的数据获取策略
- 🔍 智能数据质量检查
- ⚡ 优化的数据预处理流程
- 📊 分析友好的数据格式
"""

import requests
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import time
from functools import wraps
import hashlib

# 导入我们刚写的工具类
from utils.helpers.date_utils import DateUtils, create_date_utils
from utils.helpers.validation_utils import ValidationUtils, create_validation_utils, ValidationLevel

logger = logging.getLogger(__name__)


class QueryType:
    """查询类型常量"""
    REALTIME = "realtime"  # 实时状态查询
    HISTORICAL = "historical"  # 历史趋势分析
    EXPIRY = "expiry"  # 到期数据分析
    USER_ANALYSIS = "user_analysis"  # 用户行为分析
    PREDICTION = "prediction"  # 预测分析
    COMPREHENSIVE = "comprehensive"  # 综合分析


class APIConnector:
    """
    🚀 AI驱动的智能API连接器 - 增强版

    新增功能：
    1. AI驱动的查询策略
    2. 智能数据验证和预处理
    3. 分析优化的数据格式
    4. 第8个API完整支持
    """

    def __init__(self, config: Optional[Dict] = None, claude_client=None, gpt_client=None):
        """
        初始化增强版API连接器

        Args:
            config: API配置
            claude_client: Claude客户端，用于智能查询分析
            gpt_client: GPT客户端，用于数据验证
        """
        # 保留原有配置逻辑
        self.config = config or self._load_default_config()
        self.base_url = self.config.get('base_url', 'https://api2.3foxzthtdgfy.com')
        self.api_key = self.config.get('api_key', 'f22bf0ec9c61dce227d8f5d64998e883')

        # 并发控制
        self.semaphore = asyncio.Semaphore(self.config.get('max_concurrent', 10))
        self.session = None

        # 缓存管理
        self.cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟缓存

        # 熔断器状态
        self.circuit_breaker = {
            'failure_count': 0,
            'last_failure_time': None,
            'state': 'closed'  # closed, open, half_open
        }

        # 🆕 AI增强功能
        self.claude_client = claude_client
        self.gpt_client = gpt_client
        self.date_utils = create_date_utils(claude_client)
        self.validator = create_validation_utils(claude_client, gpt_client)

        # 🆕 智能查询统计
        self.query_stats = {
            'total_queries': 0,
            'ai_optimized_queries': 0,
            'cache_hits': 0,
            'validation_failures': 0
        }

        logger.info("Enhanced APIConnector initialized with AI capabilities")

    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            'base_url': 'https://api2.3foxzthtdgfy.com',
            'api_key': 'f22bf0ec9c61dce227d8f5d64998e883',
            'max_concurrent': 10,
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1.0,
            'cache_ttl': 300,
            'circuit_breaker_threshold': 5,
            'circuit_breaker_timeout': 60,
            # 🆕 AI增强配置
            'enable_ai_validation': True,
            'enable_smart_caching': True,
            'data_quality_threshold': 0.8
        }

    # ============= 保留原有核心方法 (略作增强) =============

    async def _get_session(self):
        """获取或创建aiohttp会话"""
        try:
            if self.session is None or self.session.closed:
                # 检查事件循环是否正在运行
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    raise RuntimeError("No running event loop")
                    
                timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
                self.session = aiohttp.ClientSession(timeout=timeout)
            return self.session
        except Exception as e:
            logger.error(f"Failed to get or create session: {str(e)}")
            raise

    def _generate_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """生成缓存键"""
        key_data = f"{endpoint}_{params or {} }"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """检查缓存是否有效"""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl

    def _should_bypass_circuit_breaker(self) -> bool:
        """检查熔断器状态"""
        if self.circuit_breaker['state'] == 'closed':
            return True
        elif self.circuit_breaker['state'] == 'open':
            if (time.time() - self.circuit_breaker['last_failure_time'] >
                    self.config.get('circuit_breaker_timeout', 60)):
                self.circuit_breaker['state'] = 'half_open'
                return True
            return False
        else:  # half_open
            return True

    def _update_circuit_breaker(self, success: bool):
        """更新熔断器状态"""
        if success:
            if self.circuit_breaker['state'] == 'half_open':
                self.circuit_breaker['state'] = 'closed'
            self.circuit_breaker['failure_count'] = 0
        else:
            self.circuit_breaker['failure_count'] += 1
            self.circuit_breaker['last_failure_time'] = time.time()

            threshold = self.config.get('circuit_breaker_threshold', 5)
            if self.circuit_breaker['failure_count'] >= threshold:
                self.circuit_breaker['state'] = 'open'
                logger.warning(f"Circuit breaker opened due to {threshold} failures")

    async def _make_request(self, endpoint: str, params: Dict = None,
                            use_cache: bool = True, enable_ai_validation: bool = True) -> Dict[str, Any]:
        """
        🚀 增强版HTTP请求方法
        新增AI验证和智能缓存
        """
        self.query_stats['total_queries'] += 1

        # 检查熔断器
        if not self._should_bypass_circuit_breaker():
            return {
                "success": False,
                "message": "Service temporarily unavailable (circuit breaker open)"
            }

        # 🆕 智能缓存检查
        cache_key = self._generate_cache_key(endpoint, params)
        if use_cache and self.config.get('enable_smart_caching', True):
            cached_result = await self._smart_cache_lookup(cache_key, endpoint)
            if cached_result:
                self.query_stats['cache_hits'] += 1
                logger.info(f"Smart cache hit for {endpoint}")
                return cached_result

        # 准备请求参数
        request_params = {'key': self.api_key}
        if params:
            request_params.update(params)

        url = f"{self.base_url}{endpoint}"

        # 并发控制
        async with self.semaphore:
            try:
                session = await self._get_session()
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    logger.error(f"Cannot make request to {endpoint}: Event loop is closed")
                    return {
                        "success": False,
                        "message": "Cannot process request: Event loop is closed"
                    }
                raise

            # 重试机制
            max_retries = self.config.get('max_retries', 3)
            retry_delay = self.config.get('retry_delay', 1.0)

            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Making request to {endpoint} (attempt {attempt + 1})")

                    async with session.get(url, params=request_params) as response:
                        if response.status == 200:
                            data = await response.json()

                            # 验证响应格式
                            if isinstance(data, dict) and 'result' in data:
                                result = {
                                    "success": data.get('result', False),
                                    "data": data.get('data'),
                                    "status": data.get('status', 0),
                                    "endpoint": endpoint,
                                    "request_params": request_params,
                                    "timestamp": datetime.now().isoformat()
                                }

                                if not result["success"]:
                                    result["message"] = f"API returned error: status={data.get('status', 'unknown')}"
                                else:
                                    # 🆕 AI数据验证
                                    if enable_ai_validation and self.config.get('enable_ai_validation', True):
                                        validation_result = await self._ai_validate_response(result, endpoint)
                                        result["validation"] = validation_result

                                        if not validation_result.get("is_valid", True):
                                            self.query_stats['validation_failures'] += 1
                                            logger.warning(f"Data validation failed for {endpoint}")
                            else:
                                result = {
                                    "success": False,
                                    "message": "Invalid API response format"
                                }

                            # 更新熔断器（成功）
                            self._update_circuit_breaker(True)

                            # 🆕 智能缓存
                            if result["success"] and use_cache:
                                await self._smart_cache_store(cache_key, result, endpoint)

                            return result

                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")

                except asyncio.CancelledError:
                    logger.warning(f"Request to {endpoint} was cancelled")
                    return {
                        "success": False,
                        "message": "Request cancelled"
                    }
                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        logger.error(f"Cannot continue request to {endpoint}: Event loop is closed")
                        return {
                            "success": False,
                            "message": "Cannot process request: Event loop is closed"
                        }
                    logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                except Exception as e:
                    logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")

                    # 最后一次尝试失败
                    if attempt == max_retries:
                        self._update_circuit_breaker(False)
                        return {
                            "success": False,
                            "message": f"API request failed after {max_retries + 1} attempts: {str(e)}"
                        }

                    # 等待后重试
                    try:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    except asyncio.CancelledError:
                        logger.warning(f"Sleep before retry was cancelled for {endpoint}")
                        return {
                            "success": False,
                            "message": "Request retry cancelled"
                        }

    # ============= 🆕 新增第8个API方法 =============

    async def intelligent_data_fetch_enhanced(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        🧠 增强版智能数据获取 - 基于Claude的精确分析
        """
        try:
            logger.info("🧠 开始增强版智能数据获取")

            api_calls = query_analysis.get("execution_plan", {}).get("api_calls", [])
            time_entities = query_analysis.get("query_understanding", {}).get("time_entities", [])

            # 按优先级排序API调用
            sorted_calls = sorted(api_calls, key=lambda x: x.get("priority", 999))

            # 并行执行API调用
            tasks = []
            task_metadata = []

            for api_call in sorted_calls:
                method = api_call.get("api_method")
                params = api_call.get("params", {})

                # 动态调用对应的方法
                if hasattr(self, method):
                    api_method = getattr(self, method)
                    if params:
                        task = api_method(**params)
                    else:
                        task = api_method()
                    tasks.append(task)
                    task_metadata.append({
                        "method": method,
                        "params": params,
                        "reason": api_call.get("reason", "")
                    })

            # 执行所有API调用
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 整理结果
            organized_data = {}
            for i, (result, metadata) in enumerate(zip(results, task_metadata)):
                if not isinstance(result, Exception) and result.get("success"):
                    organized_data[metadata["method"]] = {
                        "data": result.get("data"),
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.error(f"API调用失败: {metadata['method']}, 错误: {result}")

            return {
                "success": True,
                "data_type": "intelligent_package",
                "organized_data": organized_data,
                "query_analysis": query_analysis,
                "execution_summary": {
                    "total_api_calls": len(tasks),
                    "successful_calls": len(organized_data),
                    "failed_calls": len(tasks) - len(organized_data)
                },
                "package_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 增强版智能数据获取失败: {str(e)}")
            return await self._fetch_basic_data_package()
    async def get_product_end_interval(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        🆕 获取区间产品到期数据 - 第8个API

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            Dict[str, Any]: 区间到期数据
        """
        try:
            logger.info(f"🔍 获取区间到期数据: {start_date} 到 {end_date}")

            # 🆕 使用AI验证日期参数
            if self.date_utils:
                start_valid = self.date_utils.validate_api_date_format(start_date)
                end_valid = self.date_utils.validate_api_date_format(end_date)

                if not start_valid or not end_valid:
                    return {
                        "success": False,
                        "message": f"日期格式错误: start={start_date}, end={end_date}"
                    }

            params = {
                'start_date': start_date,
                'end_date': end_date
            }

            result = await self._make_request('/api/sta/product_end_interval', params)

            # 🆕 为区间数据添加统计信息
            if result.get("success") and result.get("data"):
                result["data"] = await self._enhance_interval_data(result["data"], start_date, end_date)

            return result

        except Exception as e:
            logger.error(f"❌ 区间到期数据获取失败: {str(e)}")
            return {
                "success": False,
                "message": f"区间数据获取失败: {str(e)}"
            }

    async def _enhance_interval_data(self, interval_data: Dict[str, Any],
                                     start_date: str, end_date: str) -> Dict[str, Any]:
        """增强区间数据，添加统计信息"""

        enhanced_data = interval_data.copy()

        try:
            # 计算日期范围
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            total_days = (end_dt - start_dt).days + 1

            # 添加统计信息
            total_amount = float(interval_data.get("到期金额", 0))
            total_quantity = int(interval_data.get("到期数量", 0))

            enhanced_data["interval_stats"] = {
                "total_days": total_days,
                "daily_average_amount": total_amount / total_days if total_days > 0 else 0,
                "daily_average_quantity": total_quantity / total_days if total_days > 0 else 0,
                "start_date_formatted": start_dt.strftime("%Y-%m-%d"),
                "end_date_formatted": end_dt.strftime("%Y-%m-%d")
            }

            # 🆕 如果有产品列表，计算每个产品的平均到期量
            if "产品列表" in interval_data:
                for product in enhanced_data["产品列表"]:
                    product_amount = float(product.get("到期金额", 0))
                    product_quantity = int(product.get("到期数量", 0))

                    product["daily_stats"] = {
                        "avg_daily_amount": product_amount / total_days if total_days > 0 else 0,
                        "avg_daily_quantity": product_quantity / total_days if total_days > 0 else 0
                    }

        except Exception as e:
            logger.warning(f"区间数据增强失败: {str(e)}")

        return enhanced_data

    # ============= 🆕 AI驱动的智能数据获取 =============

    async def intelligent_data_fetch(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        🧠 AI驱动的智能数据获取
        根据查询分析结果，智能选择和组合API调用

        Args:
            query_analysis: 查询分析结果 (来自query_parser)

        Returns:
            Dict[str, Any]: 智能组合的数据结果
        """
        try:
            logger.info("🧠 开始智能数据获取")
            self.query_stats['ai_optimized_queries'] += 1

            query_type = query_analysis.get("query_type", "realtime")
            data_requirements = query_analysis.get("data_requirements", {})
            time_range = query_analysis.get("time_range", {})

            # 🆕 根据查询类型选择API组合策略
            if query_type == QueryType.REALTIME:
                return await self._fetch_realtime_data_package()
            elif query_type == QueryType.HISTORICAL:
                return await self._fetch_historical_data_package(time_range)
            elif query_type == QueryType.EXPIRY:
                return await self._fetch_expiry_data_package(time_range)
            elif query_type == QueryType.USER_ANALYSIS:
                return await self._fetch_user_analysis_package(time_range)
            elif query_type == QueryType.PREDICTION:
                return await self._fetch_prediction_data_package(time_range)
            elif query_type == QueryType.COMPREHENSIVE:
                return await self._fetch_comprehensive_data_package(time_range)
            else:
                # 默认获取基础数据
                return await self._fetch_basic_data_package()

        except Exception as e:
            logger.error(f"❌ 智能数据获取失败: {str(e)}")
            return {
                "success": False,
                "message": f"智能数据获取失败: {str(e)}",
                "fallback_data": await self._fetch_basic_data_package()
            }

    async def _fetch_realtime_data_package(self) -> Dict[str, Any]:
        """获取实时数据包"""
        logger.info("📊 获取实时数据包")

        # 并行获取实时相关数据
        tasks = [
            self.get_system_data(),
            self.get_expiring_products_today(),
            self.get_expiring_products_tomorrow()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": True,
            "data_type": "realtime_package",
            "system_data": results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None,
            "today_expiry": results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None,
            "tomorrow_expiry": results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None,
            "package_timestamp": datetime.now().isoformat()
        }

    async def _fetch_historical_data_package(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """获取历史数据包"""
        logger.info(f"📈 获取历史数据包: {time_range}")

        start_date = time_range.get("start_date", "")
        end_date = time_range.get("end_date", "")

        # 转换日期格式为API格式
        if self.date_utils:
            start_api = self.date_utils.date_to_api_format(start_date)
            end_api = self.date_utils.date_to_api_format(end_date)
        else:
            start_api = start_date.replace("-", "")
            end_api = end_date.replace("-", "")

        # 并行获取历史数据
        tasks = [
            self.get_date_range_data(start_api, end_api, ["daily"]),
            self.get_date_range_data(start_api, end_api, ["user_daily"]),
            self.get_system_data()  # 当前状态作为对比
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": True,
            "data_type": "historical_package",
            "daily_data": results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None,
            "user_daily_data": results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None,
            "current_state": results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None,
            "time_range": {
                "start_date": start_date,
                "end_date": end_date,
                "api_format": {"start": start_api, "end": end_api}
            },
            "package_timestamp": datetime.now().isoformat()
        }

    async def _fetch_expiry_data_package(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """获取到期数据包"""
        logger.info(f"⏰ 获取到期数据包: {time_range}")

        start_date = time_range.get("start_date", "")
        end_date = time_range.get("end_date", "")

        # 转换日期格式
        if self.date_utils:
            start_api = self.date_utils.date_to_api_format(start_date)
            end_api = self.date_utils.date_to_api_format(end_date)
        else:
            start_api = start_date.replace("-", "")
            end_api = end_date.replace("-", "")

        # 并行获取到期相关数据
        tasks = [
            self.get_product_end_interval(start_api, end_api),  # 🆕 使用新API
            self.get_product_data(),  # 产品详情
            self.get_system_data()  # 当前状态
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": True,
            "data_type": "expiry_package",
            "interval_expiry": results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None,
            "product_details": results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None,
            "current_system": results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None,
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "package_timestamp": datetime.now().isoformat()
        }

    async def _fetch_comprehensive_data_package(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """获取综合数据包 (所有数据)"""
        logger.info(f"🎯 获取综合数据包: {time_range}")

        # 获取各种数据包
        realtime_task = self._fetch_realtime_data_package()
        historical_task = self._fetch_historical_data_package(time_range)
        expiry_task = self._fetch_expiry_data_package(time_range)

        packages = await asyncio.gather(realtime_task, historical_task, expiry_task, return_exceptions=True)

        return {
            "success": True,
            "data_type": "comprehensive_package",
            "realtime_package": packages[0] if len(packages) > 0 and not isinstance(packages[0], Exception) else None,
            "historical_package": packages[1] if len(packages) > 1 and not isinstance(packages[1], Exception) else None,
            "expiry_package": packages[2] if len(packages) > 2 and not isinstance(packages[2], Exception) else None,
            "package_metadata": {
                "total_api_calls": sum(self._count_api_calls(pkg) for pkg in packages if isinstance(pkg, dict)),
                "data_completeness": self._calculate_data_completeness(packages),
                "generation_time": datetime.now().isoformat()
            }
        }

    # ============= 🆕 AI数据验证和智能缓存 =============

    async def _ai_validate_response(self, response: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """AI验证API响应"""

        if not self.validator:
            return {"is_valid": True, "confidence": 0.5, "method": "no_validator"}

        try:
            # 根据endpoint确定验证级别
            validation_level = ValidationLevel.STANDARD
            if "product_end" in endpoint or "system" in endpoint:
                validation_level = ValidationLevel.AI_ENHANCED

            validation_result = await self.validator.validate_api_response(
                response,
                expected_fields=self._get_expected_fields(endpoint)
            )

            return {
                "is_valid": validation_result.is_valid,
                "confidence": validation_result.overall_score,
                "issues_count": len(validation_result.issues),
                "validation_level": validation_level.value,
                "method": "ai_enhanced"
            }

        except Exception as e:
            logger.error(f"AI验证失败: {str(e)}")
            return {"is_valid": True, "confidence": 0.3, "error": str(e)}

    async def _smart_cache_lookup(self, cache_key: str, endpoint: str) -> Optional[Dict[str, Any]]:
        """智能缓存查找"""

        if cache_key not in self.cache:
            return None

        cache_entry = self.cache[cache_key]

        # 🆕 根据数据类型调整缓存有效期
        dynamic_ttl = self._calculate_dynamic_ttl(endpoint)

        if time.time() - cache_entry['timestamp'] < dynamic_ttl:
            # 🆕 缓存命中时进行数据新鲜度检查
            if await self._is_cached_data_fresh(cache_entry, endpoint):
                return cache_entry['data']

        # 清除过期缓存
        del self.cache[cache_key]
        return None

    async def _smart_cache_store(self, cache_key: str, data: Dict[str, Any], endpoint: str):
        """智能缓存存储"""

        # 🆕 只缓存高质量数据
        if self._should_cache_data(data, endpoint):
            self.cache[cache_key] = {
                'data': data,
                'timestamp': time.time(),
                'endpoint': endpoint,
                'quality_score': await self._calculate_data_quality_score(data)
            }

            # 限制缓存大小
            if len(self.cache) > 1000:
                self._cleanup_cache()

    def _calculate_dynamic_ttl(self, endpoint: str) -> float:
        """根据endpoint动态计算TTL"""

        # 不同类型数据的缓存策略
        if "system" in endpoint:
            return 60  # 系统数据1分钟
        elif "day" in endpoint:
            return 300  # 每日数据5分钟
        elif "product_end" in endpoint:
            return 180  # 到期数据3分钟
        elif "product" in endpoint:
            return 600  # 产品数据10分钟
        else:
            return self.cache_ttl  # 默认TTL

    async def _is_cached_data_fresh(self, cache_entry: Dict[str, Any], endpoint: str) -> bool:
        """检查缓存数据新鲜度"""

        # 🆕 对于重要数据，可以进行轻量级验证
        if "system" in endpoint:
            # 系统数据可以快速检查时间戳
            cached_data = cache_entry.get('data', {})
            cached_timestamp = cached_data.get('timestamp', '')

            try:
                cache_time = datetime.fromisoformat(cached_timestamp.replace('Z', '+00:00'))
                now = datetime.now()

                # 如果缓存数据超过1小时，可能需要更新
                if (now - cache_time).total_seconds() > 3600:
                    return False
            except:
                pass

        return True

    def _should_cache_data(self, data: Dict[str, Any], endpoint: str) -> bool:
        """判断是否应该缓存数据"""

        # 基础检查
        if not data.get("success", False):
            return False

        # 检查数据质量阈值
        validation_info = data.get("validation", {})
        if validation_info.get("confidence", 1.0) < self.config.get('data_quality_threshold', 0.8):
            return False

        return True

    async def _calculate_data_quality_score(self, data: Dict[str, Any]) -> float:
        """计算数据质量分数"""

        score = 1.0

        # 基础质量检查
        if not data.get("success", False):
            score -= 0.5

        # 验证结果检查
        validation_info = data.get("validation", {})
        if validation_info:
            score = min(score, validation_info.get("confidence", 1.0))

        # 数据完整性检查
        data_content = data.get("data", {})
        if isinstance(data_content, dict) and len(data_content) == 0:
            score -= 0.3

        return max(0.0, score)

    # ============= 🆕 保留并增强原有方法 =============

    async def get_system_data(self) -> Dict[str, Any]:
        """获取系统概览数据 - 增强版"""
        result = await self._make_request('/api/sta/system')

        # 🆕 为系统数据添加分析友好的格式
        if result.get("success") and result.get("data"):
            result["data"] = await self._enhance_system_data(result["data"])

        return result

    async def get_daily_data(self, date: Optional[str] = None) -> Dict[str, Any]:
        """获取每日数据 - 增强版"""
        params = {}
        if date:
            params['date'] = date

        result = await self._make_request('/api/sta/day', params)

        # 🆕 为每日数据添加趋势分析准备
        if result.get("success") and result.get("data"):
            result["data"] = await self._enhance_daily_data(result["data"], date)

        return result

    async def _enhance_system_data(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强系统数据"""
        enhanced = system_data.copy()

        try:
            # 🆕 添加计算字段
            total_balance = float(enhanced.get("总余额", 0))
            total_inflow = float(enhanced.get("总入金", 0))
            total_outflow = float(enhanced.get("总出金", 0))

            enhanced["computed_metrics"] = {
                "net_flow": total_inflow - total_outflow,
                "outflow_ratio": (total_outflow / total_inflow) if total_inflow > 0 else 0,
                "balance_utilization": (total_balance / total_inflow) if total_inflow > 0 else 0
            }

            # 🆕 用户统计增强
            user_stats = enhanced.get("用户统计", {})
            total_users = user_stats.get("总用户数", 0)
            active_users = user_stats.get("活跃用户数", 0)

            if total_users > 0:
                enhanced["user_metrics"] = {
                    "activity_rate": (active_users / total_users),
                    "avg_balance_per_user": total_balance / total_users,
                    "avg_investment_per_user": float(enhanced.get("总投资金额", 0)) / total_users
                }

        except Exception as e:
            logger.warning(f"系统数据增强失败: {str(e)}")

        return enhanced

    async def _enhance_daily_data(self, daily_data: Dict[str, Any], date: Optional[str]) -> Dict[str, Any]:
        """增强每日数据"""
        enhanced = daily_data.copy()

        try:
            # 🆕 添加日期元数据
            if date and self.date_utils:
                date_info = self.date_utils.get_date_info(
                    self.date_utils.api_format_to_date(date)
                )
                enhanced["date_metadata"] = date_info

            # 🆕 添加比率计算
            inflow = float(enhanced.get("入金", 0))
            outflow = float(enhanced.get("出金", 0))

            enhanced["daily_metrics"] = {
                "net_flow": inflow - outflow,
                "flow_ratio": (inflow / outflow) if outflow > 0 else float('inf'),
                "activity_score": (float(enhanced.get("购买产品数量", 0)) +
                                   float(enhanced.get("到期产品数量", 0))) / 2
            }

        except Exception as e:
            logger.warning(f"每日数据增强失败: {str(e)}")

        return enhanced

    # ============= 🆕 便捷方法和工具函数 =============

    def _get_expected_fields(self, endpoint: str) -> List[str]:
        """获取endpoint期望的字段"""
        field_map = {
            '/api/sta/system': ['总余额', '总入金', '总出金', '用户统计'],
            '/api/sta/day': ['日期', '注册人数', '入金', '出金'],
            '/api/sta/product': ['产品总数', '产品列表'],
            '/api/sta/product_end': ['日期', '到期数量', '到期金额'],
            '/api/sta/product_end_interval': ['日期', '到期数量', '到期金额']
        }
        return field_map.get(endpoint, [])

    def _count_api_calls(self, package: Dict[str, Any]) -> int:
        """计算数据包中的API调用次数"""
        if not isinstance(package, dict):
            return 0

        count = 0
        for key, value in package.items():
            if isinstance(value, dict) and value.get("endpoint"):
                count += 1
            elif isinstance(value, dict):
                count += self._count_api_calls(value)

        return count

    def _calculate_data_completeness(self, packages: List[Any]) -> float:
        """计算数据完整性得分"""
        if not packages:
            return 0.0

        successful_packages = sum(1 for pkg in packages
                                  if isinstance(pkg, dict) and pkg.get("success", False))

        return successful_packages / len(packages)

    def _cleanup_cache(self):
        """清理缓存"""
        # 删除最旧的缓存项
        if self.cache:
            oldest_key = min(self.cache.keys(),
                             key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

    async def _fetch_basic_data_package(self) -> Dict[str, Any]:
        """获取基础数据包 (降级方案)"""
        logger.info("📦 获取基础数据包")

        system_data = await self.get_system_data()
        return {
            "success": True,
            "data_type": "basic_package",
            "system_data": system_data,
            "package_timestamp": datetime.now().isoformat()
        }

    async def _fetch_user_analysis_package(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """获取用户分析数据包"""
        logger.info("👥 获取用户分析数据包")

        tasks = [
            self.get_user_data(page=1),
            self.get_system_data()
        ]

        # 如果有时间范围，添加用户每日数据
        if time_range.get("start_date"):
            start_api = self.date_utils.date_to_api_format(time_range["start_date"])
            end_api = self.date_utils.date_to_api_format(time_range["end_date"])
            tasks.append(self.get_date_range_data(start_api, end_api, ["user_daily"]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": True,
            "data_type": "user_analysis_package",
            "user_data": results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None,
            "system_data": results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None,
            "user_daily_data": results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None,
            "package_timestamp": datetime.now().isoformat()
        }

    async def _fetch_prediction_data_package(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """获取预测分析数据包"""
        logger.info("🔮 获取预测分析数据包")

        # 预测需要更长的历史数据
        historical_package = await self._fetch_historical_data_package(time_range)
        realtime_package = await self._fetch_realtime_data_package()

        return {
            "success": True,
            "data_type": "prediction_package",
            "historical_data": historical_package,
            "current_snapshot": realtime_package,
            "prediction_metadata": {
                "historical_depth": time_range,
                "data_quality": "enhanced_for_prediction"
            },
            "package_timestamp": datetime.now().isoformat()
        }

    # ============= 保留原有方法 (其他方法保持不变) =============

    async def get_product_data(self) -> Dict[str, Any]:
        """获取产品数据"""
        return await self._make_request('/api/sta/product')

    async def get_user_daily_data(self, date: Optional[str] = None) -> Dict[str, Any]:
        """获取用户每日数据"""
        params = {}
        if date:
            params['date'] = date
        return await self._make_request('/api/sta/user_daily', params)

    async def get_user_data(self, page: int = 1) -> Dict[str, Any]:
        """获取详细用户数据"""
        params = {'page': page}
        return await self._make_request('/api/sta/user', params)

    async def get_product_end_data(self, date: str) -> Dict[str, Any]:
        """获取产品到期数据"""
        params = {'date': date}
        return await self._make_request('/api/sta/product_end', params)

    # 🆕 增强版便捷方法
    async def get_expiring_products_today(self) -> Dict[str, Any]:
        """获取今日到期产品"""
        today = datetime.now().strftime('%Y%m%d')
        return await self.get_product_end_data(today)

    async def get_expiring_products_tomorrow(self) -> Dict[str, Any]:
        """获取明日到期产品"""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
        return await self.get_product_end_data(tomorrow)

    async def get_expiring_products_week(self) -> Dict[str, Any]:
        """获取本周到期产品"""
        today = datetime.now()
        week_end = today + timedelta(days=6)
        return await self.get_product_end_interval(
            today.strftime('%Y%m%d'),
            week_end.strftime('%Y%m%d')
        )

    async def get_expiring_products_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """获取指定范围到期产品"""
        return await self.get_product_end_interval(start_date, end_date)

    # ============= 保留原有的批量数据获取方法 =============

    async def get_date_range_data(self, start_date: str, end_date: str,
                                  data_types: List[str] = None) -> Dict[str, Any]:
        """
        智能获取日期范围数据 - 保留原有逻辑，增加AI验证
        """
        logger.info(f"Getting range data from {start_date} to {end_date}")

        try:
            # 生成日期列表
            dates = self._generate_date_list(start_date, end_date)
            logger.info(f"Generated {len(dates)} dates for range query")

            # 默认数据类型
            if data_types is None:
                data_types = ['daily', 'product_end']

            # 准备批量任务
            tasks = []
            task_metadata = []

            for date in dates:
                for data_type in data_types:
                    if data_type == 'daily':
                        task = self.get_daily_data(date)
                        tasks.append(task)
                        task_metadata.append({'type': 'daily', 'date': date})
                    elif data_type == 'product_end':
                        task = self.get_product_end_data(date)
                        tasks.append(task)
                        task_metadata.append({'type': 'product_end', 'date': date})
                    elif data_type == 'user_daily':
                        task = self.get_user_daily_data(date)
                        tasks.append(task)
                        task_metadata.append({'type': 'user_daily', 'date': date})

            logger.info(f"Executing {len(tasks)} API calls concurrently")

            # 分批执行（避免过多并发）
            batch_size = self.config.get('batch_size', 20)
            all_results = []

            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_metadata = task_metadata[i:i + batch_size]

                logger.info(f"Processing batch {i // batch_size + 1}, size: {len(batch_tasks)}")

                # 执行批次
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # 处理结果
                for result, metadata in zip(batch_results, batch_metadata):
                    if isinstance(result, Exception):
                        logger.error(f"Task failed: {metadata}, error: {str(result)}")
                        all_results.append({
                            'metadata': metadata,
                            'success': False,
                            'error': str(result)
                        })
                    else:
                        all_results.append({
                            'metadata': metadata,
                            'success': result.get('success', False),
                            'data': result.get('data') if result.get('success') else None,
                            'error': result.get('message') if not result.get('success') else None
                        })

                # 批次间隔，避免API限制
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.5)

            # 整理结果
            organized_results = self._organize_range_results(all_results, dates, data_types)

            success_count = sum(1 for r in all_results if r['success'])
            total_count = len(all_results)

            return {
                "success": True,
                "data": organized_results,
                "metadata": {
                    "date_range": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_days": len(dates)
                    },
                    "data_types": data_types,
                    "execution_stats": {
                        "total_api_calls": total_count,
                        "successful_calls": success_count,
                        "success_rate": success_count / total_count if total_count > 0 else 0
                    }
                }
            }

        except Exception as e:
            logger.error(f"Date range data retrieval failed: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to retrieve date range data: {str(e)}"
            }

    def _generate_date_list(self, start_date: str, end_date: str) -> List[str]:
        """生成日期列表"""
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            dates = []
            current_dt = start_dt

            while current_dt <= end_dt:
                dates.append(current_dt.strftime('%Y%m%d'))
                current_dt += timedelta(days=1)

            return dates

        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return []

    def _organize_range_results(self, results: List[Dict], dates: List[str],
                                data_types: List[str]) -> Dict[str, Any]:
        """整理批量查询结果"""
        organized = {
            'by_date': {},
            'by_type': {dt: {} for dt in data_types},
            'summary': {
                'successful_dates': [],
                'failed_dates': [],
                'data_completeness': {}
            }
        }

        # 按日期和类型组织数据
        for result in results:
            metadata = result['metadata']
            date = metadata['date']
            data_type = metadata['type']

            # 按日期组织
            if date not in organized['by_date']:
                organized['by_date'][date] = {}

            organized['by_date'][date][data_type] = {
                'success': result['success'],
                'data': result.get('data'),
                'error': result.get('error')
            }

            # 按类型组织
            organized['by_type'][data_type][date] = {
                'success': result['success'],
                'data': result.get('data'),
                'error': result.get('error')
            }

        # 计算摘要统计
        for date in dates:
            date_results = organized['by_date'].get(date, {})
            successful_types = [dt for dt in data_types
                                if date_results.get(dt, {}).get('success', False)]

            if len(successful_types) == len(data_types):
                organized['summary']['successful_dates'].append(date)
            elif len(successful_types) == 0:
                organized['summary']['failed_dates'].append(date)

        # 数据完整性统计
        for data_type in data_types:
            successful_count = sum(1 for date in dates
                                   if organized['by_type'][data_type].get(date, {}).get('success', False))
            organized['summary']['data_completeness'][data_type] = {
                'successful_days': successful_count,
                'total_days': len(dates),
                'completeness_rate': successful_count / len(dates) if dates else 0
            }

        return organized

    # ============= 🆕 统计和监控方法 =============

    def get_connector_stats(self) -> Dict[str, Any]:
        """获取连接器统计信息"""
        return {
            "query_statistics": self.query_stats,
            "circuit_breaker_status": self.circuit_breaker,
            "cache_statistics": {
                "cache_size": len(self.cache),
                "cache_hit_rate": (self.query_stats['cache_hits'] /
                                   max(self.query_stats['total_queries'], 1)) * 100
            },
            "ai_enhancement_status": {
                "claude_available": self.claude_client is not None,
                "gpt_available": self.gpt_client is not None,
                "date_utils_available": self.date_utils is not None,
                "validator_available": self.validator is not None
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试基础API调用
            test_result = await self.get_system_data()

            return {
                "status": "healthy" if test_result.get("success") else "degraded",
                "api_connectivity": test_result.get("success", False),
                "circuit_breaker_state": self.circuit_breaker['state'],
                "ai_capabilities": {
                    "claude_ready": self.claude_client is not None,
                    "gpt_ready": self.gpt_client is not None,
                    "validation_ready": self.validator is not None
                },
                "performance_stats": self.query_stats,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # ============= 资源清理 =============

    async def close(self):
        """关闭连接器，清理资源"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Enhanced APIConnector session closed")


# ============= 工厂函数 =============

def create_enhanced_api_connector(config: Optional[Dict] = None,
                                  claude_client=None,
                                  gpt_client=None) -> APIConnector:
    """
    创建增强版API连接器实例

    Args:
        config: 配置字典
        claude_client: Claude客户端实例
        gpt_client: GPT客户端实例

    Returns:
        APIConnector: 增强版API连接器实例
    """
    return APIConnector(config, claude_client, gpt_client)


# ============= 使用示例 =============

async def main():
    """使用示例"""
    connector = create_enhanced_api_connector()

    try:
        print("=== 增强版API连接器测试 ===")

        # 1. 基础API测试
        system_data = await connector.get_system_data()
        print(f"系统数据获取: {'成功' if system_data.get('success') else '失败'}")

        # 2. 🆕 新API测试
        interval_data = await connector.get_product_end_interval("20240601", "20240630")
        print(f"区间到期数据: {'成功' if interval_data.get('success') else '失败'}")

        # 3. 🆕 智能数据包测试
        query_analysis = {
            "query_type": QueryType.REALTIME,
            "data_requirements": {},
            "time_range": {}
        }

        intelligent_data = await connector.intelligent_data_fetch(query_analysis)
        print(f"智能数据获取: {'成功' if intelligent_data.get('success') else '失败'}")

        # 4. 统计信息
        stats = connector.get_connector_stats()
        print(f"总查询次数: {stats['query_statistics']['total_queries']}")
        print(f"AI优化查询: {stats['query_statistics']['ai_optimized_queries']}")

        # 5. 健康检查
        health = await connector.health_check()
        print(f"系统状态: {health['status']}")

    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(main())