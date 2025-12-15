"""
404-AI: Factory Defect Recognition System
A Flask-based web application for detecting defects using AI models.
"""

from flask import Flask, jsonify

app = Flask(__name__)


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
    app.run(debug=True, host='0.0.0.0', port=5000)
