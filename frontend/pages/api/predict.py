import os
import torch
from PIL import Image
from torchvision import transforms
from flask import jsonify
from cataract_detection.src.model.model import CataractCNN


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cataract_detection", "saved_models", "cataract_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CataractCNN(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def handler(event, context):
    if 'file' not in event['files']:
        return jsonify({'error': 'No file provided'}), 400

    file = event['files']['file']
    try:
        # Preprocess the image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)

        # Get the prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Map prediction
        class_mapping = {
            0: "Cataract",
            1: "Diabetic Retinopathy",
            2: "Glaucoma",
            3: "Normal"
        }
        return jsonify({'prediction': class_mapping[int(predicted.item())]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
