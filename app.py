"""
404-AI: Factory Defect Recognition System
A Flask-based web application for detecting defects using AI models.
"""

import os
from flask import Flask, jsonify
import config

app = Flask(__name__)

# Load configuration based on environment
env = os.environ.get('FLASK_ENV', 'development')
# Support both a `config` dict (mapping env->obj) or the config module itself
if isinstance(config, dict):
    cfg = config.get(env, config.get('default', config))
else:
    cfg = config
app.config.from_object(cfg)


@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Welcome to 404-AI Factory Defect Recognition System',
        'status': 'running'
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    dependencies = {}
    
    # Check if dependencies can be imported
    try:
        import flask
        dependencies['flask'] = 'installed'
    except ImportError:
        dependencies['flask'] = 'not installed'
    
    try:
        import cv2
        dependencies['opencv'] = 'installed'
    except ImportError:
        dependencies['opencv'] = 'not installed'
    
    try:
        import tensorflow
        dependencies['tensorflow'] = 'installed'
    except ImportError:
        dependencies['tensorflow'] = 'not installed'
    
    try:
        import ultralytics
        dependencies['ultralytics'] = 'installed'
    except ImportError:
        dependencies['ultralytics'] = 'not installed'
    
    try:
        import sam2
        dependencies['sam2'] = 'installed'
    except ImportError:
        dependencies['sam2'] = 'not installed'
    
    all_installed = all(status == 'installed' for status in dependencies.values())
    
    return jsonify({
        'status': 'healthy' if all_installed else 'degraded',
        'dependencies': dependencies
    })


if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=debug, host=host, port=port)
