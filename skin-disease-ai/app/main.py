import os
import sys
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.model_handler import predict, get_model_for_gradcam, load_model
from app.gradcam import generate_gradcam
from app.disease_info import get_disease_info, DISEASE_INFO
from app.utils import preprocess_image, allowed_file, validate_image_size, image_to_base64


# ─── Flask App Configuration ───────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
)

# Enable CORS for all routes
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_disease():
    """
    Predict skin disease from an uploaded image.
    
    Expects:
        - POST request with 'image' file field
        
    Returns:
        JSON response with:
        - prediction: top prediction details
        - top_3: top 3 predictions with confidence
        - disease_info: comprehensive disease information
        - gradcam: base64-encoded Grad-CAM heatmap
        - uploaded_image: base64-encoded uploaded image
    """
    # Validate request has a file
    if 'image' not in request.files:
        return jsonify({
            'error': True,
            'message': 'No image file uploaded. Please select an image.'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'error': True,
            'message': 'No file selected. Please choose an image to upload.'
        }), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': True,
            'message': 'Invalid file type. Please upload a PNG, JPG, JPEG, BMP, or WebP image.'
        }), 400
    
    try:
        # Read image bytes
        image_bytes = file.read()
        
        # Validate file size
        is_valid, error_msg = validate_image_size(image_bytes)
        if not is_valid:
            return jsonify({
                'error': True,
                'message': error_msg
            }), 400
        
        # Preprocess image for model
        img_array, original_image = preprocess_image(image_bytes)
        
        # Run prediction
        prediction_result = predict(img_array)
        
        # Get disease info for top prediction
        top_class = prediction_result['top_prediction']['class_code']
        disease_details = get_disease_info(top_class)
        
        # Generate Grad-CAM heatmap
        model, last_conv_layer = get_model_for_gradcam()
        gradcam_image = generate_gradcam(
            model=model,
            img_array=img_array,
            original_image=original_image,
            last_conv_layer_name=last_conv_layer
        )
        
        # Convert uploaded image to base64 for display
        uploaded_image_b64 = image_to_base64(
            original_image.resize((400, 400))
        )
        
        # Build response
        response = {
            'error': False,
            'prediction': prediction_result['top_prediction'],
            'top_3': prediction_result['top_3'],
            'all_probabilities': prediction_result['all_probabilities'],
            'disease_info': disease_details,
            'gradcam': gradcam_image,
            'uploaded_image': uploaded_image_b64
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({
            'error': True,
            'message': f'An error occurred during analysis: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Dermatology Assistant',
        'version': '1.0.0'
    })


@app.route('/api/diseases', methods=['GET'])
def list_diseases():
    """Return information about all supported disease classes."""
    return jsonify({
        'diseases': DISEASE_INFO,
        'total_classes': len(DISEASE_INFO)
    })


# ─── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': True,
        'message': 'File is too large. Maximum file size is 10 MB.'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': True,
        'message': 'The requested resource was not found.'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': True,
        'message': 'An internal server error occurred. Please try again.'
    }), 500


# ─── Application Entry Point ───────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AI Dermatology Assistant")
    print("  Skin Disease Detection System v1.0")
    print("=" * 60)
    
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Pre-load the model on startup
    print("\n[i] Loading ML model...")
    load_model()
    print("[✓] Model ready!\n")
    
    # Start the Flask development server
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )

