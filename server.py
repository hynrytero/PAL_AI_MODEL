from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('model', 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        if not file or not file.filename:
            return jsonify({'error': 'Invalid file'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({'error': 'Invalid file type'}), 400

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        results = model(img)
        
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                pred = {
                    'xmin': float(box.xyxy[0][0]),
                    'ymin': float(box.xyxy[0][1]),
                    'xmax': float(box.xyxy[0][2]),
                    'ymax': float(box.xyxy[0][3]),
                    'confidence': float(box.conf),
                    'class': result.names[int(box.cls)]
                }
                predictions.append(pred)
        
        print(f"Predictions: {predictions}")  # Single console output for predictions
        return jsonify({
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))