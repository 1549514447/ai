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

    # 🆕 响应详细程度配置
    DETAILED_RESPONSE_MODE = os.getenv('DETAILED_RESPONSE_MODE', 'True').lower() == 'true'
    SHOW_CALCULATION_STEPS = os.getenv('SHOW_CALCULATION_STEPS', 'True').lower() == 'true'
    SHOW_DATA_SOURCES = os.getenv('SHOW_DATA_SOURCES', 'True').lower() == 'true'
    SHOW_VALIDATION_INFO = os.getenv('SHOW_VALIDATION_INFO', 'True').lower() == 'true'
    SHOW_BUSINESS_INSIGHTS = os.getenv('SHOW_BUSINESS_INSIGHTS', 'True').lower() == 'true'
    SHOW_DATA_QUALITY_INFO = os.getenv('SHOW_DATA_QUALITY_INFO', 'True').lower() == 'true'


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

    # 🆕 开发环境：默认启用所有详细信息
    DETAILED_RESPONSE_MODE = True
    SHOW_CALCULATION_STEPS = True
    SHOW_DATA_SOURCES = True
    SHOW_VALIDATION_INFO = True


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'

    # 🆕 生产环境：可以选择性关闭一些详细信息以提高性能
    DETAILED_RESPONSE_MODE = True
    SHOW_CALCULATION_STEPS = True
    SHOW_DATA_SOURCES = True
    SHOW_VALIDATION_INFO = False  # 生产环境可以关闭验证信息显示


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}