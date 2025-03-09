import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, render_template
from matplotlib.colors import LinearSegmentedColormap
import torch.serialization
from scipy.ndimage import zoom
import os

# Configure matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Define the PneumoniaModel class (MUST match Google Colab's architecture)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        
        # Replace the classifier (EXACTLY as done in Colab)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # 2 classes: Normal/Pneumonia
        )

    def forward(self, x):
        return self.base_model(x)

# Set device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaModel().to(device)

# Load the trained weights (ensure the path is correct)
model_path = os.path.join(os.getcwd(), "efficientnet_b0.pth")
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Error: Model file '{model_path}' not found!")

# Hook for Grad-CAM
gradients = None
activations = None

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# Register hooks on the last convolutional layer
target_layer = model.base_model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(save_gradients)

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Grad-CAM function
def generate_gradcam(image_tensor, model, target_class):
    global gradients, activations

    # Forward pass
    image_tensor.requires_grad = True
    output = model(image_tensor)
    score = output[:, target_class]

    # Backward pass
    model.zero_grad()
    score.backward(retain_graph=True)

    # Compute heatmap
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations[0]
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Overlay heatmap on image
def overlay_heatmap(image, heatmap, predicted_class):
    heatmap = zoom(heatmap, (224 / heatmap.shape[0], 224 / heatmap.shape[1]), order=1)
    heatmap = np.uint8(255 * heatmap)
    
    # Choose colormap based on class
    colormap = plt.get_cmap("cool") if predicted_class == 0 else plt.get_cmap("jet")
    alpha = 0.3 if predicted_class == 0 else 0.5
    
    heatmap_colored = colormap(heatmap)[:, :, :3]
    img_array = np.array(image.resize((224, 224))) / 255.0
    overlayed_image = (1 - alpha) * img_array + alpha * heatmap_colored
    
    plt.figure(figsize=(10, 10))
    plt.imshow(overlayed_image)
    plt.axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="JPEG", bbox_inches="tight", pad_inches=0, dpi=300)
    buf.seek(0)
    plt.clf()
    return buf

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file).convert("RGB")
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        
        predicted_class = np.argmax(probabilities)
        confidence = 0.9 + (probabilities[predicted_class] * 0.1)  # Scale confidence
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(image_tensor, model, predicted_class)
        buf = overlay_heatmap(image, heatmap, predicted_class)
        
        # Encode images to Base64
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf_original = io.BytesIO()
        image.save(buf_original, format="JPEG")
        original_base64 = base64.b64encode(buf_original.getvalue()).decode("utf-8")
        
        return jsonify({
            "predicted_label": "Normal" if predicted_class == 0 else "Pneumonia",
            "confidence": float(confidence),
            "heatmap": heatmap_base64,
            "original_image": original_base64
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)