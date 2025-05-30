# data/connectors/database_connector.py
import mysql.connector
from mysql.connector import pooling
import asyncio
import aiomysql
import logging
import json
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime
import hashlib
import time

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    现代化数据库连接器 - 支持同步/异步双模式
    功能：连接池管理、事务处理、查询优化、缓存支持
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据库连接器

        Args:
            config: 数据库配置字典
        """
        self.config = config or self._load_default_config()

        # 同步连接池
        self.sync_pool = None

        # 异步连接池
        self.async_pool = None

        # 查询缓存
        self.query_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟缓存

        # 性能监控
        self.performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_query_time': 0.0,
            'slow_queries': []
        }

        # 初始化同步连接池
        self._init_sync_pool()

        logger.info("DatabaseConnector initialized successfully")

    def _load_default_config(self) -> Dict:
        """加载默认数据库配置"""
        return {
            'host': 'localhost',
            'user': 'root',
            'password': '119689',
            'database': 'jarvis',
            'port': 3306,
            'charset': 'utf8mb4',
            'pool_size': 10,
            'pool_name': 'intelligent_qa_pool',
            'cache_ttl': 300,
            'slow_query_threshold': 1.0,  # 慢查询阈值（秒）
            'connection_timeout': 30,
            'autocommit': False
        }

    def _init_sync_pool(self):
        """初始化同步连接池"""
        try:
            pool_config = {
                'pool_name': self.config['pool_name'],
                'pool_size': self.config['pool_size'],
                'host': self.config['host'],
                'user': self.config['user'],
                'password': self.config['password'],
                'database': self.config['database'],
                'port': self.config['port'],
                'charset': self.config['charset'],
                'autocommit': self.config['autocommit'],
                'connection_timeout': self.config['connection_timeout']
            }

            self.sync_pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)
            logger.info(f"Sync connection pool initialized: {self.config['pool_name']}")

        except Exception as e:
            logger.error(f"Failed to initialize sync connection pool: {str(e)}")
            raise

    async def _init_async_pool(self):
        """初始化异步连接池"""
        if self.async_pool is None:
            try:
                self.async_pool = await aiomysql.create_pool(
                    host=self.config['host'],
                    port=self.config['port'],
                    user=self.config['user'],
                    password=self.config['password'],
                    db=self.config['database'],
                    charset=self.config['charset'],
                    minsize=1,
                    maxsize=self.config['pool_size'],
                    autocommit=self.config['autocommit']
                )
                logger.info("Async connection pool initialized")

            except Exception as e:
                logger.error(f"Failed to initialize async connection pool: {str(e)}")
                raise

    # ============= 同步数据库操作 =============

    @contextmanager
    def get_connection(self):
        """获取同步数据库连接的上下文管理器"""
        conn = None
        try:
            conn = self.sync_pool.get_connection()
            yield conn
            if not self.config['autocommit']:
                conn.commit()
        except Exception as e:
            if conn and not self.config['autocommit']:
                conn.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: tuple = None,
                      use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        执行查询并返回结果

        Args:
            query: SQL查询语句
            params: 查询参数
            use_cache: 是否使用查询缓存

        Returns:
            查询结果列表
        """
        start_time = time.time()

        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(query, params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                try:
                    cursor.execute(query, params or ())
                    result = cursor.fetchall()

                    # 缓存结果
                    if use_cache:
                        self._cache_result(cache_key, result)

                    # 性能统计
                    self._update_performance_stats(query, start_time)

                    return result
                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Query execution failed: {query}, error: {str(e)}")
            raise

    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        执行更新操作

        Args:
            query: SQL更新语句
            params: 查询参数

        Returns:
            受影响的行数
        """
        start_time = time.time()

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(query, params or ())
                    affected_rows = cursor.rowcount

                    # 清除相关缓存
                    self._clear_related_cache(query)

                    # 性能统计
                    self._update_performance_stats(query, start_time)

                    return affected_rows
                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Update execution failed: {query}, error: {str(e)}")
            raise

    def execute_insert(self, query: str, params: tuple = None) -> int:
        """
        执行插入操作

        Args:
            query: SQL插入语句
            params: 查询参数

        Returns:
            新插入行的ID
        """
        start_time = time.time()

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(query, params or ())
                    new_id = cursor.lastrowid

                    # 清除相关缓存
                    self._clear_related_cache(query)

                    # 性能统计
                    self._update_performance_stats(query, start_time)

                    return new_id
                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Insert execution failed: {query}, error: {str(e)}")
            raise

    def execute_batch(self, query: str, params_list: List[tuple]) -> List[int]:
        """
        批量执行操作

        Args:
            query: SQL语句
            params_list: 参数列表

        Returns:
            操作结果列表
        """
        start_time = time.time()
        results = []

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    for params in params_list:
                        cursor.execute(query, params)
                        if query.strip().upper().startswith('INSERT'):
                            results.append(cursor.lastrowid)
                        else:
                            results.append(cursor.rowcount)

                    # 清除相关缓存
                    self._clear_related_cache(query)

                    # 性能统计
                    self._update_performance_stats(f"BATCH: {query}", start_time)

                    return results
                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Batch execution failed: {query}, error: {str(e)}")
            raise

    # ============= 异步数据库操作 =============

    @asynccontextmanager
    async def get_async_connection(self):
        """获取异步数据库连接的上下文管理器"""
        if self.async_pool is None:
            await self._init_async_pool()

        async with self.async_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                try:
                    yield cursor
                    if not self.config['autocommit']:
                        await conn.commit()
                except Exception as e:
                    if not self.config['autocommit']:
                        await conn.rollback()
                    logger.error(f"Async database operation failed: {str(e)}")
                    raise

    async def async_execute_query(self, query: str, params: tuple = None,
                                  use_cache: bool = True) -> List[Dict[str, Any]]:
        """异步执行查询"""
        start_time = time.time()

        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(query, params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result

        try:
            async with self.get_async_connection() as cursor:
                await cursor.execute(query, params or ())
                result = await cursor.fetchall()

                # 缓存结果
                if use_cache:
                    self._cache_result(cache_key, result)

                # 性能统计
                self._update_performance_stats(query, start_time)

                return result

        except Exception as e:
            logger.error(f"Async query execution failed: {query}, error: {str(e)}")
            raise

    async def async_execute_update(self, query: str, params: tuple = None) -> int:
        """异步执行更新操作"""
        start_time = time.time()

        try:
            async with self.get_async_connection() as cursor:
                await cursor.execute(query, params or ())
                affected_rows = cursor.rowcount

                # 清除相关缓存
                self._clear_related_cache(query)

                # 性能统计
                self._update_performance_stats(query, start_time)

                return affected_rows

        except Exception as e:
            logger.error(f"Async update execution failed: {query}, error: {str(e)}")
            raise

    async def async_execute_insert(self, query: str, params: tuple = None) -> int:
        """异步执行插入操作"""
        start_time = time.time()

        try:
            async with self.get_async_connection() as cursor:
                await cursor.execute(query, params or ())
                new_id = cursor.lastrowid

                # 清除相关缓存
                self._clear_related_cache(query)

                # 性能统计
                self._update_performance_stats(query, start_time)

                return new_id

        except Exception as e:
            logger.error(f"Async insert execution failed: {query}, error: {str(e)}")
            raise

    # ============= 缓存管理 =============

    def _generate_cache_key(self, query: str, params: tuple = None) -> str:
        """生成缓存键"""
        cache_data = f"{query}_{params or ()}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存结果"""
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['data']
            else:
                # 清除过期缓存
                del self.query_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """缓存查询结果"""
        self.query_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }

        # 限制缓存大小
        if len(self.query_cache) > 1000:
            # 清除最旧的缓存项
            oldest_key = min(self.query_cache.keys(),
                             key=lambda k: self.query_cache[k]['timestamp'])
            del self.query_cache[oldest_key]

    def _clear_related_cache(self, query: str):
        """清除相关缓存（在数据修改后）"""
        # 简单实现：清除所有缓存
        # 实际应用中可以根据查询类型智能清除相关缓存
        if 'INSERT' in query.upper() or 'UPDATE' in query.upper() or 'DELETE' in query.upper():
            self.query_cache.clear()

    # ============= 性能监控 =============

    def _update_performance_stats(self, query: str, start_time: float):
        """更新性能统计"""
        execution_time = time.time() - start_time

        self.performance_stats['total_queries'] += 1

        # 更新平均查询时间
        total_time = (self.performance_stats['avg_query_time'] *
                      (self.performance_stats['total_queries'] - 1) + execution_time)
        self.performance_stats['avg_query_time'] = total_time / self.performance_stats['total_queries']

        # 记录慢查询
        if execution_time > self.config['slow_query_threshold']:
            self.performance_stats['slow_queries'].append({
                'query': query,
                'execution_time': execution_time,
                'timestamp': datetime.now()
            })

            # 限制慢查询记录数量
            if len(self.performance_stats['slow_queries']) > 100:
                self.performance_stats['slow_queries'] = self.performance_stats['slow_queries'][-50:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        cache_hit_rate = (self.performance_stats['cache_hits'] /
                          max(self.performance_stats['total_queries'], 1)) * 100

        return {
            'total_queries': self.performance_stats['total_queries'],
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'avg_query_time': f"{self.performance_stats['avg_query_time']:.3f}s",
            'slow_queries_count': len(self.performance_stats['slow_queries']),
            'recent_slow_queries': self.performance_stats['slow_queries'][-5:] if self.performance_stats[
                'slow_queries'] else []
        }

    # ============= 工具方法 =============

    def test_connection(self) -> Dict[str, Any]:
        """测试数据库连接"""
        try:
            result = self.execute_query("SELECT 1 as test", use_cache=False)
            return {
                'success': True,
                'message': 'Database connection successful',
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Database connection failed: {str(e)}'
            }

    async def async_test_connection(self) -> Dict[str, Any]:
        """异步测试数据库连接"""
        try:
            result = await self.async_execute_query("SELECT 1 as test", use_cache=False)
            return {
                'success': True,
                'message': 'Async database connection successful',
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Async database connection failed: {str(e)}'
            }

    def execute_script(self, script_path: str) -> Dict[str, Any]:
        """执行SQL脚本文件"""
        try:
            with open(script_path, 'r', encoding='utf-8') as file:
                script = file.read()

            # 分割多个SQL语句
            statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]

            results = []
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    for statement in statements:
                        cursor.execute(statement)
                        if statement.upper().startswith('SELECT'):
                            results.append(cursor.fetchall())
                        else:
                            results.append(cursor.rowcount)
                finally:
                    cursor.close()

            return {
                'success': True,
                'message': f'Script executed successfully: {len(statements)} statements',
                'results': results
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Script execution failed: {str(e)}'
            }

    # ============= 资源清理 =============

    async def close(self):
        """关闭连接池"""
        try:
            if self.async_pool:
                self.async_pool.close()
                await self.async_pool.wait_closed()
                logger.info("Async connection pool closed")

            if self.sync_pool:
                # MySQL连接池没有显式关闭方法，会自动清理
                logger.info("Sync connection pool will be cleaned up automatically")

        except Exception as e:
            logger.error(f"Error closing connection pools: {str(e)}")

    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'async_pool') and self.async_pool:
                asyncio.create_task(self.close())
        except:
            pass


# ============= 工厂函数 =============

def create_database_connector(config: Optional[Dict] = None) -> DatabaseConnector:
    """
    创建数据库连接器实例

    Args:
        config: 数据库配置字典

    Returns:
        DatabaseConnector实例
    """
    return DatabaseConnector(config)


# ============= 使用示例 =============

async def main():
    """使用示例"""
    # 配置数据库
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'your_password',
        'database': 'intelligent_qa_system',
        'port': 3306,
        'pool_size': 5
    }

    # 创建数据库连接器
    db = create_database_connector(db_config)

    try:
        # 测试连接
        sync_test = db.test_connection()
        print("Sync connection test:", sync_test)

        async_test = await db.async_test_connection()
        print("Async connection test:", async_test)

        # 性能统计
        stats = db.get_performance_stats()
        print("Performance stats:", stats)

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())