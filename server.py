import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('Users/irfaa/OneDrive - Nanyang Technological University/Desktop/firstModel19/weights/best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the file as an image
        img = Image.open(file.stream)
        
        # Convert the image to a NumPy array
        img = np.array(img)
        
        # Convert RGB to BGR format (YOLO expects BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Run YOLOv8 inference
        results = model(img)  # This returns a list of Results objects
        result = results[0]   # Access the first Results object
        
        # Extract bounding boxes and labels from the results
        boxes = result.boxes  # Bounding boxes
        names = result.names  # Class names
        
        # Prepare predictions in a list of dictionaries
        predictions = []
        for box in boxes:
            predictions.append({
                "x1": box.xyxy[0][0].item(),
                "y1": box.xyxy[0][1].item(),
                "x2": box.xyxy[0][2].item(),
                "y2": box.xyxy[0][3].item(),
                "confidence": box.conf[0].item(),
                "class_id": box.cls[0].item(),
                "class_name": names[int(box.cls[0].item())]
            })
        
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
