import os
import sys
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from flask import jsonify
from http.server import BaseHTTPRequestHandler

# Dynamically add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from cataract_detection.src.model.model import CataractCNN

app = Flask(__name__)
CORS(app)

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../cataract_detection/saved_models/cataract_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CataractCNN(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# Class mapping
class_mapping = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal"
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Vercel-compatible handler
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Ensure the content type is multipart/form-data
        content_type = self.headers.get('Content-Type')
        if not content_type or 'multipart/form-data' not in content_type:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid content type")
            return

        try:
            # Parse the incoming image
            content_length = int(self.headers.get('Content-Length', 0))
            file_data = self.rfile.read(content_length)
            image = Image.open(io.BytesIO(file_data)).convert('RGB')
            
            # Process the image
            image = transform(image).unsqueeze(0).to(DEVICE)

            # Predict
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            # Get class name
            class_name = class_mapping[int(predicted.item())]

            # Respond with prediction
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({'prediction': class_name}).encode('utf-8'))

        except Exception as e:
            # Handle errors
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
