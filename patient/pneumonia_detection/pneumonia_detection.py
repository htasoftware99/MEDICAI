import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import cv2
import numpy as np
from PIL import Image
import requests
import json
import os
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styles (adapted from brain_tumor.py)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .settings-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .settings-title {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        height: fit-content;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 300px;
    }
    
    .yolo-container {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        height: fit-content;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 300px;
    }
    
    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .prediction-details {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    
    .confidence-bar {
        background-color: rgba(255,255,255,0.2);
        border-radius: 25px;
        overflow: hidden;
        margin: 0.8rem 0;
        height: 30px;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        transition: width 0.3s ease;
    }
    
    .explanation-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .explanation-title {
        color: #2d3436;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .explanation-content {
        background: rgba(255,255,255,0.8);
        padding: 1.5rem;
        border-radius: 15px;
        color: #2d3436;
        line-height: 1.6;
        backdrop-filter: blur(10px);
    }
    
    .qa-container {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .qa-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .qa-input-container {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .qa-response-container {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 15px;
        color: #2d3436;
        line-height: 1.6;
        backdrop-filter: blur(10px);
        border-left: 4px solid #6c5ce7;
    }
    
    .expert-badge {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .gradcam-container {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .gradcam-title {
        color: #2d3436;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .warning-container {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .category-item {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        color: #ecf0f1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid #34495e;
    }
    
    /* Fix for Streamlit elements */
    .stSelectbox > div > div {
        background-color: white !important;
    }
    
    .stCheckbox > label {
        color: white !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Model and constants
CATEGORIES = ['Normal', 'Pneumonia']
CATEGORY_DESCRIPTIONS = {
    'Normal': 'Healthy lung image with no signs of pneumonia',
    'Pneumonia': 'Lung image showing signs of pneumonia, an infection causing inflammation in the air sacs'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OLLAMA_URL = "http://localhost:10500/api/generate"

# Model class
class PneumoniaModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaModel, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        for param in self.mobilenet.parameters():
            param.requires_grad = False
            
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.mobilenet(x)
        x = self.classifier(x)
        return x

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer_name='features'):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            print(f"Forward hook triggered. Activations shape: {output.shape}")
        
        def backward_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()
                print(f"Backward hook triggered. Gradients shape: {grad_out[0].shape}")
        
        
        try:
            
            target_layer = self.model.mobilenet.features[-1]
            print(f"Target layer found: {target_layer}")
            
            # Hook'ları kaydet
            forward_hook_handle = target_layer.register_forward_hook(forward_hook)
            backward_hook_handle = target_layer.register_full_backward_hook(backward_hook)
            
            self.hooks.append(forward_hook_handle)
            self.hooks.append(backward_hook_handle)
            print("Hooks registered successfully")
            
        except Exception as e:
            print(f"Error registering hooks: {e}")
            # Try alternative layer
            try:
                target_layer = self.model.mobilenet.features[18]  # Son konv bloğu
                forward_hook_handle = target_layer.register_forward_hook(forward_hook)
                backward_hook_handle = target_layer.register_full_backward_hook(backward_hook)
                self.hooks.append(forward_hook_handle)
                self.hooks.append(backward_hook_handle)
                print("Alternative hooks registered")
            except Exception as e2:
                print(f"Failed to register alternative hooks: {e2}")

    def generate_cam(self, input_tensor, class_idx=None):
        try:
            print("Starting Grad-CAM generation...")
            
            # Move model and tensor to same device
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
            
            # Clear previous gradients
            self.model.zero_grad()
            self.gradients = None
            self.activations = None
            
            # Enable gradient calculation
            input_tensor.requires_grad_(True)
            
            # Forward pass
            print("Performing forward pass...")
            output = self.model(input_tensor)
            print(f"Model output shape: {output.shape}")
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            print(f"Target class index: {class_idx}")
            
            # Backward pass
            print("Performing backward pass...")
            target = output[0, class_idx]
            target.backward(retain_graph=True)
            
            # Check if hooks are working
            if self.gradients is None:
                print("ERROR: Gradients not captured!")
                return None
                
            if self.activations is None:
                print("ERROR: Activations not captured!")
                return None
                
            print(f"Gradients shape: {self.gradients.shape}")
            print(f"Activations shape: {self.activations.shape}")
            
            # Calculate CAM
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            print(f"Weights shape: {weights.shape}")
            
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = torch.relu(cam)
            print(f"CAM shape before squeeze: {cam.shape}")
            
            # Convert CAM to numpy
            cam = cam.squeeze().cpu().numpy()
            print(f"CAM shape after squeeze: {cam.shape}")
            
            if cam.size == 0:
                print("ERROR: Empty CAM generated")
                return None
            
            # If CAM is 0 dimensional (one pixel), expand
            if len(cam.shape) == 0:
                cam = np.array([[cam]])
            
            # Resize to 224x224
            if cam.shape != (224, 224):
                cam = cv2.resize(cam, (224, 224))
            
            # Normalize CAM
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            print("Grad-CAM generated successfully!")
            return cam
            
        except Exception as e:
            print(f"Error in Grad-CAM generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass
        self.hooks = []
        print("Hooks cleaned up")

# Corrected Grad-CAM visualization function
def create_gradcam_visualization(model, image_tensor, original_image):
    
    try:
        print("Creating Grad-CAM visualization...")
        
        # Create GradCAM instance
        grad_cam = GradCAM(model, target_layer_name='features')
        
        # Create CAM
        cam = grad_cam.generate_cam(image_tensor)
        
        # Clear hooks
        grad_cam.cleanup()
        
        if cam is None:
            print("Failed to generate CAM")
            return None
        
        print(f"CAM generated with shape: {cam.shape}")
        print(f"CAM min: {cam.min()}, max: {cam.max()}")
        
        # Prepare original image
        img_array = np.array(original_image.resize((224, 224)))
        
        # Convert grayscale to RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Normalize to 0-1 range
        if img_array.max() > 1.0:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)
        
        print(f"Image array shape: {img_array.shape}")
        print(f"Image array min: {img_array.min()}, max: {img_array.max()}")
        
        # Create Heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        print(f"Heatmap shape: {heatmap.shape}")
        
        # Create Overlay
        overlay = 0.6 * img_array + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        
        print("Grad-CAM visualization created successfully!")
        return overlay
        
    except Exception as e:
        print(f"Grad-CAM visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def gradcam_section_fixed(show_gradcam, mobilenet_model, image_tensor, image):
    
    if show_gradcam:
        st.markdown("---")
        st.markdown("""
        <div class="gradcam-container">
            <div class="gradcam-title">Grad-CAM Visualization</div>
            <p style="text-align: center; color: #2d3436;">This visualization shows which regions the classification model focuses on during analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Creating Grad-CAM visualization..."):
            gradcam_img = create_gradcam_visualization(mobilenet_model, image_tensor, image)
        
        if gradcam_img is not None:
            gradcam_col1, gradcam_col2 = st.columns([1, 1], gap="medium")
            with gradcam_col1:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown("#### Original Image")
                st.image(image.resize((224, 224)), caption="Original X-ray", width=300)
                st.markdown("</div>", unsafe_allow_html=True)
            with gradcam_col2:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown("#### Grad-CAM Overlay")
                st.image(gradcam_img, caption="Attention Heatmap", width=300)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Grad-CAM visualization could not be generated. Check the console for detailed error messages.")
            st.info("Possible solutions:\n1. Check if the model architecture is correct\n2. Verify that the target layer exists\n3. Ensure input tensor has the correct shape")

# Test function - for Grad-CAM debug
def test_gradcam_debug(model, image_tensor):
    """Grad-CAM debug testi"""
    print("=== Grad-CAM Debug Test ===")
    
    # Check model structure
    print("Model structure:")
    for name, module in model.named_modules():
        if 'features' in name:
            print(f"  {name}: {module}")
    
    
    try:
        last_features = model.mobilenet.features[-1]
        print(f"Last features layer: {last_features}")
        
        # Forward pass test
        with torch.no_grad():
            output = model(image_tensor)
            print(f"Model output shape: {output.shape}")
            
        # Grad-CAM test
        grad_cam = GradCAM(model)
        cam = grad_cam.generate_cam(image_tensor)
        grad_cam.cleanup()
        
        if cam is not None:
            print("Grad-CAM test successful!")
            return True
        else:
            print("Grad-CAM test failed!")
            return False
            
    except Exception as e:
        print(f"Debug test error: {e}")
        return False

# Model loading functions
@st.cache_resource
def load_models():
    """Load and cache models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MobileNet model
    mobilenet_model = PneumoniaModel()
    try:
        mobilenet_model.load_state_dict(torch.load('models/pneumonia_model_pytorch.pth', map_location=device))
        mobilenet_model.eval()
        
    except FileNotFoundError:
        st.error("❌ models/pneumonia_model_pytorch.pth file not found!")
        return None, None
    
    # YOLO model
    try:
        yolo_model = YOLO('models/pneumonia_model/best.pt')
        
    except FileNotFoundError:
        st.error("❌ models/pneumonia_model/best.pt file not found!")
        return mobilenet_model, None
    
    return mobilenet_model, yolo_model

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for MobileNet model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(DEVICE)

# YOLO detection function
def perform_yolo_detection(yolo_model, image):
    """Perform YOLO detection"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        results = yolo_model(img_array)
        detections = []
        annotated_image = img_array.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = yolo_model.names[class_id]
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': class_name
                    })
                    
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(annotated_image, f'{class_name}: {confidence*100:.2f}%', 
                               (int(x1), max(int(y1)-10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return detections, Image.fromarray(annotated_image)
    
    except Exception as e:
        st.error(f"YOLO detection error: {str(e)}")
        return [], image

# Classification prediction function
def predict_pneumonia(model, image_tensor):
    """Make pneumonia classification prediction"""
    model = model.to(DEVICE)
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return CATEGORIES[predicted.item()], confidence.item(), probabilities.cpu().numpy()[0]

# Fixed Grad-CAM visualization
def create_gradcam_visualization(model, image_tensor, original_image):
    """Create Grad-CAM visualization"""
    try:
        # Create GradCAM instance
        grad_cam = GradCAM(model, target_layer_name='features')
        
        # Generate CAM
        cam = grad_cam.generate_cam(image_tensor)
        
        # Cleanup hooks
        grad_cam.cleanup()
        
        if cam is None:
            print("Failed to generate CAM")
            return None
        
        # Prepare original image
        img_array = np.array(original_image.resize((224, 224)))
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Ensure img_array is in correct range
        if img_array.max() > 1.0:
            img_array = img_array.astype(np.float32) / 255.0
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Create overlay
        overlay = 0.6 * img_array + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        
        return overlay
        
    except Exception as e:
        print(f"Grad-CAM error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Ollama API function
def call_ollama(prompt):
    """Send request to Ollama API"""
    try:
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"API Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "Request timed out. The AI explanation service is currently unavailable."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to AI explanation service. Please check if Ollama is running."
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"

# Medical explanation function
def get_medical_explanation(prediction, confidence, detections, is_eli5=False):
    """Get medical explanation including both classification and detection results"""
    detection_info = f"YOLO detected {len(detections)} pneumonia region(s)" if detections else "No pneumonia regions detected by YOLO"
    
    if is_eli5:
        prompt = f"""
        Explain in simple terms as if explaining to a child:
        
        The chest X-ray shows: {prediction}
        
        Explain these results in very simple, understandable terms without using medical terminology. 
        Write as if explaining to a 10-year-old child.
        Respond in English and use maximum 150 words.
        """
    else:
        prompt = f"""
        Provide medical explanation:
        
        The chest X-ray shows: {prediction}
        
        Explain these results in medical detail:
        - Information about pneumonia
        - Significance of the detection results
        - Possible symptoms
        - General treatment approaches
        - General information about prognosis
        
        Respond in English and use professional medical language.
        Use maximum 350 words.
        """
    
    return call_ollama(prompt)

# Expert medical answer function
def get_expert_medical_answer(question, prediction, confidence, detections):
    """Get expert medical answer for user questions"""
    
    prompt = f"""
You are an expert in pulmonology and radiology. You have extensive knowledge about lung conditions, pneumonia, diagnostic imaging, and patient care.

Current Patient Analysis Context:
- Chest X-ray shows: {prediction}

Patient Question: "{question}"

Please provide a comprehensive, professional medical response based on your expertise in pulmonology and radiology. Address the patient's specific question while considering the current analysis results.

Guidelines for your response:
- Use professional medical terminology appropriately
- Provide evidence-based information
- Be empathetic and understanding
- Include relevant details about lung anatomy, pneumonia characteristics, or treatment options as needed
- Always emphasize the importance of consulting with healthcare professionals
- If the question is outside your expertise or requires specific patient data, acknowledge limitations

Respond in English and provide a thorough answer (maximum 400 words).
"""
    
    return call_ollama(prompt)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered pneumonia detection with YOLO and classification with MobileNetV2</p>', unsafe_allow_html=True)
    
    # Initialize session state for settings
    if 'show_yolo' not in st.session_state:
        st.session_state.show_yolo = True
    if 'show_gradcam' not in st.session_state:
        st.session_state.show_gradcam = True
    if 'show_explanation' not in st.session_state:
        st.session_state.show_explanation = True
    if 'explanation_type' not in st.session_state:
        st.session_state.explanation_type = "Medical Explanation"
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'question_counter' not in st.session_state:
        st.session_state.question_counter = 0
    
    # Settings section
    st.markdown("""
    <div class="settings-container">
        <div class="settings-title">Analysis Settings</div>
    </div>
    """, unsafe_allow_html=True)
    
    settings_container = st.container()
    with settings_container:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_yolo = st.checkbox("Show YOLO Detection", value=st.session_state.show_yolo)
            st.session_state.show_yolo = show_yolo
        with col2:
            show_gradcam = st.checkbox("Show Grad-CAM Visualization", value=st.session_state.show_gradcam)
            st.session_state.show_gradcam = show_gradcam
        with col3:
            show_explanation = st.checkbox("Show AI Explanation", value=st.session_state.show_explanation)
            st.session_state.show_explanation = show_explanation
        with col4:
            if show_explanation:
                explanation_type = st.selectbox(
                    "Explanation Type:",
                    ["Medical Explanation", "Simple Explanation (ELI5)"],
                    index=0 if st.session_state.explanation_type == "Medical Explanation" else 1
                )
                st.session_state.explanation_type = explanation_type
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        mobilenet_model, yolo_model = load_models()
    
    if mobilenet_model is None:
        st.error("Classification model could not be loaded. Please ensure 'models/pneumonia_model_pytorch.pth' file exists.")
        return
    
    if yolo_model is None:
        st.error("YOLO model could not be loaded. Please ensure 'models/pneumonia_model/best.pt' file exists.")
        return
    
    # File upload section
    st.markdown("---")
    st.markdown("### Upload Chest X-ray Image")
    st.markdown("Select a chest X-ray image in JPG, JPEG, or PNG format for analysis")
    
    uploaded_file = st.file_uploader(
        "Choose file...",
        type=['jpg', 'jpeg', 'png'],
        help="The uploaded image should be a chest X-ray scan.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Create main columns for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, caption="Chest X-ray image to be analyzed", width=350)
        
        with col2:
            st.markdown("#### Analysis Results")
            
            # YOLO Detection
            if show_yolo and yolo_model is not None:
                with st.spinner("Running YOLO pneumonia detection..."):
                    detections, yolo_annotated_image = perform_yolo_detection(yolo_model, image)
                if detections and any(d['class'].lower() == 'pneumonia' for d in detections):
                    st.markdown(f"""
                    <div class="yolo-container" style="margin-top: 0;">
                        <div class="prediction-title">YOLO Detection Results</div>
                        <div class="prediction-details">
                            <p><strong>Detected Regions:</strong> {len(detections)}</p>
                            {f'<p><strong>Highest Confidence:</strong> {max([d["confidence"] for d in detections])*100:.2f}%</p>' if detections else '<p><strong>Status:</strong> No pneumonia regions detected</p>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                detections = []
            
            # Classification
            with st.spinner("Analyzing image for classification..."):
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict_pneumonia(mobilenet_model, image_tensor)
            
            # Show classification results
            st.markdown(f"""
            <div class="prediction-container" style="margin-top: 1rem;">
                <div class="prediction-title">Classification: {predicted_class}</div>
                <div class="prediction-details">
                    <p><strong>Description:</strong> {CATEGORY_DESCRIPTIONS[predicted_class]}</p>
                    <p><strong>Confidence Level:</strong> {confidence*100:.2f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # YOLO Detection Visualization
        if show_yolo and detections and any(d['class'].lower() == 'pneumonia' for d in detections):
            st.markdown("---")
            st.markdown("#### YOLO Detection Visualization")
            
            yolo_col1, yolo_col2 = st.columns([1, 1])
            with yolo_col1:
                st.markdown("#### Original Image")
                st.image(image, caption="Original X-ray", width=450)
            with yolo_col2:
                st.markdown("#### Detected Regions")
                st.image(yolo_annotated_image, caption="YOLO Detection Results", width=450)
            
            # Detection details
            if detections:
                st.markdown("#### Detection Details")
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    class_name = detection['class']
                    st.markdown(f"""
                    <div class="category-item">
                        <strong>Detection {i+1}:</strong><br>
                        Class: {class_name}<br>
                        Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})<br>
                        Confidence: {conf*100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
        
        # Confidence scores for all classes
        st.markdown("---")
        st.markdown("#### Classification Confidence Scores")
        
        confidence_container = st.container()
        with confidence_container:
            for i, (category, prob) in enumerate(zip(CATEGORIES, probabilities)):
                color = "#74b9ff" if category == predicted_class else "#7f8c8d"
                st.markdown(f"""
                <div class="category-item">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>{category}</strong>
                        <span>{prob*100:.2f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prob*100}%; background-color: {color};">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Grad-CAM visualization
        if show_gradcam:
            st.markdown("---")
            st.markdown("""
            <div class="gradcam-container">
                <div class="gradcam-title">Grad-CAM Visualization</div>
                <p style="text-align: center; color: #2d3436;">This visualization shows which regions the classification model focuses on during analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Creating Grad-CAM visualization..."):
                gradcam_img = create_gradcam_visualization(mobilenet_model, image_tensor, image)
            
            if gradcam_img is not None:
                gradcam_col1, gradcam_col2 = st.columns([1, 1], gap="medium")
                with gradcam_col1:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("#### Original Image")
                    st.image(image.resize((224, 224)), caption="Original X-ray", width=300)
                    st.markdown("</div>", unsafe_allow_html=True)
                with gradcam_col2:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("#### Grad-CAM Overlay")
                    st.image(gradcam_img, caption="Attention Heatmap", width=300)
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # AI Explanation
        if show_explanation:
            st.markdown("---")
            is_eli5 = explanation_type == "Simple Explanation (ELI5)"
            
            with st.spinner("Preparing AI explanation..."):
                explanation = get_medical_explanation(predicted_class, confidence, detections, is_eli5)
            
            title = "Simple Explanation" if is_eli5 else "Medical Explanation"
            
            st.markdown(f"""
            <div class="explanation-container">
                <div class="explanation-title">{title}</div>
                <div class="explanation-content">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Medical Q&A Section
            st.markdown("""
            <div class="qa-container">
                <div class="qa-title">Ask the Pulmonology Expert</div>
                <div class="qa-input-container">
                    <div class="expert-badge">Expert Pulmonology & Radiology Specialist</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Ask your question about pneumonia, treatment options, or any related medical concerns:",
                placeholder="For example: What are the treatment options for pneumonia? What symptoms should I watch for? How serious is this condition?",
                height=100,
                key=f"medical_question_{st.session_state.question_counter}"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            
            with col_ask:
                if st.button("Get Expert Answer", type="primary", use_container_width=True):
                    if user_question.strip():
                        with st.spinner("Consulting pulmonology expert..."):
                            expert_answer = get_expert_medical_answer(user_question, predicted_class, confidence, detections)
                        
                        st.session_state.qa_history.append({
                            'question': user_question,
                            'answer': expert_answer,
                            'prediction': predicted_class,
                            'confidence': confidence
                        })
                        
                        st.session_state.question_counter += 1
                        st.rerun()
                    else:
                        st.warning("Please enter a question before asking the expert.")
            
            with col_clear:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.qa_history = []
                    st.session_state.question_counter += 1
                    st.rerun()
            
            # Display Q&A History
            if st.session_state.qa_history:
                st.markdown("#### Previous Questions & Expert Answers")
                
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
                    st.markdown(f"""
                    <div class="qa-response-container" style="margin-bottom: 1.5rem;">
                        <div style="background: #6c5ce7; color: white; padding: 0.5rem 1rem; border-radius: 10px 10px 0 0; margin: -1.5rem -1.5rem 1rem -1.5rem;">
                            <strong>Question {len(st.session_state.qa_history) - i}:</strong> {qa['question']}
                        </div>
                        <div style="margin-top: 1rem;">
                            <div style="background: rgba(108, 92, 231, 0.1); padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.9rem;">
                                <strong>Context:</strong> {qa['prediction']}
                            </div>
                            <strong>Expert Answer:</strong><br>
                            {qa['answer']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Medical Disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="warning-container">
            ⚠️ <strong>Medical Disclaimer</strong><br>
            This AI system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare professionals for medical concerns. The AI explanations are generated based on general medical knowledge and may not reflect the most current medical practices or be applicable to individual cases.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()