# Cataract Detection System

A web-based application that uses AI to detect cataracts from eye images. This system consists of a deep learning backend and an interactive frontend for users to upload images and view predictions.

---

## Features

- **AI-Powered Detection**: Uses a trained deep learning model to predict the presence of cataracts in eye images.
- **Interactive Frontend**: Built with Next.js, the frontend includes:
  - Hero section
  - Image upload section
  - Results section displaying predictions
  - About section detailing the technology.
- **Smooth Navigation**: Provides smooth transitions between sections.
- **Backend API**: Flask backend serving predictions from the trained model.

---

## Technologies Used

### Backend
- **Flask**: Serves the model as an API.
- **PyTorch**: For the deep learning model.
- **TorchVision**: Image preprocessing.
- **Pillow**: Image manipulation.

### Frontend
- **Next.js**: Modern React framework.
- **Tailwind CSS**: Styling framework.
- **React Dropzone**: For drag-and-drop image uploads.

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
5. The API will run at `http://127.0.0.1:5001`.

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
1. Navigate to the Upload section by clicking **"Upload"** in the header or the **"Get Started"** button.
2. Drag and drop an image or click to select a file.
3. Click **"Predict Cataracts"** to submit the image.

### Results Section
- After submitting an image, the app scrolls to the Results section and displays the prediction.

### About Section
- Learn about the AI technology by clicking **"About"** in the header.

---

## Smooth Navigation
- Use the header to navigate between sections: **Home**, **Upload**, **Results**, and **About**.
- The app uses smooth scrolling to transition between sections seamlessly.

---

## Deployment

### Backend
- Use Docker to containerize the backend:
  ```bash
  cd backend
  docker build -t cataract-backend .
  docker run -p 5001:5001 cataract-backend
  ```

### Frontend
- Deploy the frontend to **Vercel**:
  1. Push the repository to GitHub.
  2. Connect the repository to Vercel.
  3. Deploy the app.

---

## Future Enhancements
- **User Authentication**: Add login functionality for users.
- **Database Integration**: Store uploaded images and predictions.
- **Improved Model**: Enhance the deep learning model with a larger dataset.
- **Mobile Optimization**: Ensure the app works seamlessly on mobile devices.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

