import os
import sys
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cataract_detection.src.model.model import CataractCNN

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "cataract_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CataractCNN(num_classes=4)  
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

class_mapping = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        class_name = class_mapping[int(predicted.item())]
        return jsonify({'prediction': class_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
