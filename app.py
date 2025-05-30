# app.py - 完整应用入口
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS
from api.qa_routes import qa_routes_bp
from api.data_routes import data_bp
from core.orchestrator.intelligent_qa_orchestrator import get_orchestrator
import logging
import asyncio

# 配置日志
logging.basicConfig(level=logging.INFO)


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    CORS(app)

    # 注册蓝图
    app.register_blueprint(qa_routes_bp)
    app.register_blueprint(data_bp)

    # 初始化全局编排器
    orchestrator = get_orchestrator()

    @app.route('/')
    def index():
        """系统首页"""
        return jsonify({
            'system_name': 'AI金融业务顾问',
            'version': '2.0.0',
            'status': 'operational',
            'features': [
                '🧠 双AI协作 (Claude + GPT-4o)',
                '🎯 智能路由与上下文感知',
                '📊 多维数据分析与趋势预测',
                '💡 自动业务洞察生成',
                '🎨 智能可视化图表',
                '💬 上下文感知对话管理'
            ],
            'endpoints': {
                'intelligent_qa': '/api/qa/ask',
                'data_analysis': '/api/qa/analyze',
                'insights': '/api/qa/insights',
                'system_health': '/api/qa/system/health',
                'conversations': '/api/qa/conversations/{user_id}',
                'charts': '/api/qa/charts',
                'system_data': '/api/data/system'
            }
        })

    @app.route('/health')
    def health():
        """快速健康检查"""
        return jsonify({
            'status': 'healthy',
            'orchestrator_initialized': orchestrator.initialized,
            'timestamp': datetime.now().isoformat()
        })

    return app


if __name__ == '__main__':
    app = create_app()
    print("🚀 启动AI金融业务顾问系统...")
    print("🌐 访问 http://localhost:5000 查看系统信息")
    print("🔍 访问 http://localhost:5000/health 检查系统健康状态")
    app.run(debug=True, host='0.0.0.0', port=5000)