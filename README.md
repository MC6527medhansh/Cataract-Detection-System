# Cataract Detection System

## Overview
The **Cataract Detection System** is a web-based application that leverages deep learning to classify ocular diseases, including cataracts, diabetic retinopathy, glaucoma, and normal conditions. The system features a trained convolutional neural network (CNN) backend and an interactive frontend for users to upload images and view predictions.

---

## Features
- **AI-Powered Detection**:
  - Uses a trained CNN model to classify eye conditions: `Cataract`, `Diabetic Retinopathy`, `Glaucoma`, and `Normal`.
- **Interactive Frontend**:
  - Built using Next.js and Tailwind CSS.
  - Includes sections for uploading images, viewing results, and learning about the technology.
- **Flask Backend**:
  - Hosts the trained model for inference via a REST API.
- **Data Analysis and Preprocessing**:
  - Provides detailed dataset exploration and preprocessing to ensure model robustness.
- **Model Training Pipeline**:
  - Offers scripts for training, validation, and evaluation of the model.
- **Docker Support**:
  - Backend can be containerized for seamless deployment.

---

## Technologies Used

### Backend
- **Flask**: REST API for serving predictions.
- **PyTorch**: Deep learning framework for training the CNN.
- **TorchVision**: For image preprocessing and transformations.
- **Pillow**: Image manipulation library.

### Frontend
- **Next.js**: Modern React framework.
- **Tailwind CSS**: Styling framework.
- **React Dropzone**: For drag-and-drop image uploads.

### Data Analysis and Preprocessing
- **Python**: Core scripting language.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib/Seaborn**: Visualization libraries.
- **TQDM**: Progress tracking for data operations.

---

## Dataset
### Description
The dataset contains eye images categorized into four classes:
1. **Cataract**
2. **Diabetic Retinopathy**
3. **Glaucoma**
4. **Normal**

### Dataset Structure
- **Train Set**:
  - `Cataract`: 532 images
  - `Diabetic Retinopathy`: 586 images
  - `Glaucoma`: 492 images
  - `Normal`: 516 images
- **Validation Set**:
  - `Cataract`: 244 images
  - `Diabetic Retinopathy`: 300 images
  - `Glaucoma`: 258 images
  - `Normal`: 263 images
- **Test Set**:
  - `Cataract`: 261 images
  - `Diabetic Retinopathy`: 302 images
  - `Glaucoma`: 240 images
  - `Normal`: 256 images

---

## Installation and Setup

### Prerequisites
- **Node.js** (v18+)
- **Python** (v3.10+)
- **pip** and **virtualenv**
- **Docker** (optional, for containerized backend)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/cataract-detection-system.git
cd cataract-detection-system
```

### Step 2: Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   virtualenv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask server:
   ```bash
   python app.py
   ```
   The API will run at `http://127.0.0.1:5001`.

### Step 3: Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Open `http://localhost:3000` in your browser.

---

## Using the Application

### Upload Section
1. Navigate to the Upload section by clicking **"Upload"** in the header.
2. Drag and drop an image or click to select a file.
3. Click **"Predict Cataracts"** to submit the image.

### Results Section
- Displays the predicted class for the uploaded image (e.g., `Cataract`).

### About Section
- Learn about the AI technology and the project's objectives.

---

## Training and Evaluation

### Training Pipeline
1. Dataset preprocessing: Normalization, resizing, and augmentation.
2. Model architecture: Convolutional Neural Network (CataractCNN).
3. Loss function: Cross-entropy loss.
4. Optimizer: Adam.

### Evaluation
- **Validation Accuracy**: 62.35%
- **Loss**: Training loss: `2.2385`, Validation loss: `0.9049`

---

## Deployment

### Backend Deployment (Docker)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Build the Docker image:
   ```bash
   docker build -t cataract-backend .
   ```
3. Run the container:
   ```bash
   docker run -p 5001:5001 cataract-backend
   ```

### Frontend Deployment (Vercel)
1. Push the repository to GitHub.
2. Connect the repository to Vercel.
3. Deploy the app using Vercel's interface.

---

## Folder Structure
```
Cataract-Detection-System/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── styles/
│   ├── package.json
│   ├── next.config.js
├── cataract_detection/
│   ├── notebooks/
│   │   ├── data_analysis.ipynb
│   │   ├── data_exploration.ipynb
│   ├── src/
│   │   ├── model/
│   │   │   ├── model.py
│   │   ├── preprocessing/
│   │   │   ├── data_loader.py
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   ├── evaluation/
│   │   │   ├── evaluator.py
```

---

## Future Enhancements
- **Improved Model**:
  - Fine-tune the model for higher accuracy.
  - Experiment with architectures like EfficientNet or ResNet.
- **Database Integration**:
  - Store uploaded images and predictions.
- **User Authentication**:
  - Add login functionality.
- **Mobile Optimization**:
  - Ensure the application is responsive for mobile devices.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

