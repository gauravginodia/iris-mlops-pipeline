"""
Flask API for IRIS Classification
Production-ready with health checks and error handling
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at startup
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.joblib')
model = None

def load_model():
    """Load ML model from disk"""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✓ Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Root endpoint - API information"""
    return jsonify({
        'service': 'IRIS Classification API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'API info (this page)',
            '/health': 'Health check for Kubernetes',
            '/predict': 'POST - Single prediction',
            '/batch': 'POST - Batch predictions'
        }
    })

@app.route('/health')
def health():
    """
    Kubernetes health check endpoint
    Returns 200 if healthy, 500 if not
    """
    if model is None:
        return jsonify({
            'status': 'unhealthy',
            'reason': 'model not loaded'
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Request body (JSON):
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    Response:
    {
        "prediction": "setosa",
        "confidence": 0.95,
        "input": {...}
    }
    """
    try:
        # Check model loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        required = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Prepare features
        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence if available
        try:
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        except:
            confidence = None
        
        # Return result
        return jsonify({
            'prediction': str(prediction),
            'confidence': confidence,
            'input': data
        }), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Request body (JSON):
    {
        "samples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.3, "sepal_width": 2.9, "petal_length": 5.6, "petal_width": 1.8}
        ]
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        # Prepare features
        features = []
        for sample in samples:
            features.append([
                float(sample['sepal_length']),
                float(sample['sepal_width']),
                float(sample['petal_length']),
                float(sample['petal_width'])
            ])
        
        features = np.array(features)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Format results
        results = [
            {
                'input': samples[i],
                'prediction': str(predictions[i])
            }
            for i in range(len(predictions))
        ]
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    if not load_model():
        logger.warning("Starting API without model - check MODEL_PATH")
    
    # Start server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)