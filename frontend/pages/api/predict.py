import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from flask import jsonify

# Dynamically add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from cataract_detection.src.model.model import CataractCNN

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../cataract_detection/saved_models/cataract_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = CataractCNN(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class mapping
class_mapping = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal"
}

def handler(event, context):
    """Handle incoming POST requests for predictions."""
    if 'file' not in event['files']:
        return jsonify({'error': 'No file provided'}), 400

    file = event['files']['file']
    try:
        # Preprocess the image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)

        # Get predictions
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Return the prediction
        class_name = class_mapping[int(predicted.item())]
        return jsonify({'prediction': class_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
