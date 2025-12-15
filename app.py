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
    return jsonify({
        'status': 'healthy',
        'dependencies': {
            'flask': 'installed',
            'opencv': 'installed',
            'tensorflow': 'installed',
            'ultralytics': 'installed',
            'sam2': 'installed'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
