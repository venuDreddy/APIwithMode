import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
from scipy.stats import kurtosis, skew, entropy
import base64
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Load and preprocess the dataset for authenticity detection
data = pd.read_csv('banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
target_count = data.auth.value_counts()
nb_to_delete = target_count[0] - target_count[1]
data = data[nb_to_delete:]

x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the classifier
clf = LogisticRegression(solver='lbfgs', random_state=42)
clf.fit(x_train, y_train.values.ravel())

# Define class names
class_names = ['10_rupee', '20_rupee', '50_rupee', '100_rupee', '200_rupee', '500_rupee', '2000_rupee']

# Global variables for models
MODEL_PATH = "models/final.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # Assuming 7 classes, modify if needed
        )
        
        if not os.path.exists(MODEL_PATH):
            print(f"Model file {MODEL_PATH} not found.")
            return None
            
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load model at startup
model = load_model()
model.to(device)

# After model loading
if model is None:
    raise RuntimeError("Failed to load the model. Application cannot start.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Currency Authentication and Denomination API is running",
        "status": "success"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided'
            }), 400

        uploaded_file = request.files['image']
        if not uploaded_file.mimetype.startswith('image/'):
            return jsonify({
                'error': 'Invalid file type. Please upload an image file'
            }), 400

        # Read and process image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Authenticity Check
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        norm_image = np.array(opencv_image, dtype=np.float32) / 255.0

        # Compute features for authenticity check
        var = np.var(norm_image, axis=None)
        sk = skew(norm_image, axis=None)
        kur = kurtosis(norm_image, axis=None)
        ent = entropy(norm_image, axis=None) / 100

        # Validate features
        if not np.isfinite(var) or not np.isfinite(sk) or not np.isfinite(kur) or not np.isfinite(ent):
            return jsonify({
                "success": False,
                "error": "Invalid feature values computed from image"
            }), 400

        # Predict authenticity
        auth_prediction = clf.predict(np.array([[var, sk, kur, ent]]))
        auth_proba = clf.predict_proba(np.array([[var, sk, kur, ent]]))
        authenticity = "Real Currency" if auth_prediction[0] == 0 else "Fake Currency"
        auth_confidence = float(max(auth_proba[0]))

        # Denomination Classification
        input_image = transform(image).unsqueeze(0)
        input_image = input_image.to(device)

        with torch.no_grad():
            outputs = model(input_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = class_names[predicted.item()]
            denomination_confidence = float(confidence.item())

        # Convert edge image to base64
        edge_image = cv2.Canny(opencv_image, 100, 200)
        _, buffer = cv2.imencode('.png', edge_image)
        edge_image_base64 = base64.b64encode(buffer).decode('utf-8')

        result = {
            "success": True,
            "authenticity_features": {
                "entropy": float(ent),
                "kurtosis": float(kur),
                "skew": float(sk),
                "variance": float(var)
            },
            "authenticity": {
                "prediction": authenticity,
                "confidence": auth_confidence
            },
            "denomination": {
                "prediction": predicted_class,
                "confidence": denomination_confidence
            },
            "edge_image": edge_image_base64,
            "message": f"Currency appears to be {authenticity.lower()} {predicted_class} with {auth_confidence:.2%} authenticity confidence"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to process image"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port) 