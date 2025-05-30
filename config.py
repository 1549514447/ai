import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # 基础配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # AI模型配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

    # 金融数据API配置
    FINANCE_API_KEY = os.getenv('FINANCE_API_KEY')
    FINANCE_API_BASE_URL = os.getenv('FINANCE_API_BASE_URL')

    # 数据库配置
    DATABASE_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME', 'jarvis'),
        'charset': 'utf8mb4',
        'use_unicode': True
    }

    # 模型路由配置
    MODEL_ROUTING = {
        'primary_model': 'claude',  # claude 或 openai
        'fallback_model': 'openai',
        'collaboration_threshold': 0.7,  # 复杂度阈值
        'cache_ttl': 3600  # 缓存时间（秒）
    }


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}