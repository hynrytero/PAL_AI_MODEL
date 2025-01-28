from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'active',
        'model': 'PAL-AI YOLOv8 Rice Leaf Disease Detection Model',
        'endpoints': {
            '/': 'GET - This help message',
            '/predict': 'POST - Submit an image for rice leaf disease detection'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # Inference
    results = model(img)
    
    # Process results (YOLOv8 format)
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
    
    return jsonify({
        'predictions': predictions
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))