import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import requests
import json
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F
import io
import base64
from ultralytics import YOLO
import os

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection & Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styles - Fixed version
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
CATEGORIES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CATEGORY_DESCRIPTIONS = {
    'Glioma': 'Tumor type originating from brain and spinal cord tissue',
    'Meningioma': 'Tumor arising from the membranes surrounding the brain',
    'No Tumor': 'Normal brain image - no tumor detected',
    'Pituitary': 'Tumor originating from the pituitary gland'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OLLAMA_URL = "http://localhost:10500/api/generate"

# Model loading functions
@st.cache_resource
def load_classification_model():
    """Load the MobileNetV2 classification model"""
    try:
        model = models.mobilenet_v2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 4)
        )
        
        # Load model weights
        model.load_state_dict(torch.load('models/best_brain_tumor_model.pth', map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        return None

@st.cache_resource
def load_yolo_model():
    """Load the YOLO detection model"""
    try:
        if os.path.exists('models/best.pt'):
            model = YOLO('models/best.pt')
            return model
        else:
            st.error("YOLO model file 'models/best.pt' not found!")
            return None
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

# Image preprocessing for classification
def preprocess_image_for_classification(image):
    """Prepare image for MobileNetV2 classification model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(DEVICE)

# YOLO detection function
def detect_tumor_with_yolo(yolo_model, image):
    """Detect tumor using YOLO model"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run YOLO inference
        results = yolo_model(img_array)
        
        # Process results
        detections = []
        annotated_image = img_array.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Store detection info
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
                    
                    # Draw bounding box on image
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add label
                    label = f'Tumor: {confidence*100:.2f}%'
                    cv2.putText(annotated_image, label, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detections, Image.fromarray(annotated_image)
    
    except Exception as e:
        st.error(f"Error in YOLO detection: {str(e)}")
        return [], image

# Classification prediction function
def predict_tumor_classification(model, image_tensor):
    """Make tumor classification prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

# Grad-CAM function
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        if self.gradients is None:
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def create_gradcam_visualization(model, image_tensor, original_image):
    """Create Grad-CAM visualization"""
    try:
        target_layer = model.features[16]  # Appropriate layer for MobileNetV2
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate_cam(image_tensor)
        
        if cam is None:
            return None
        
        # Convert original image to numpy array
        img_array = np.array(original_image.resize((224, 224)))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = 0.6 * img_array + 0.4 * heatmap
        overlay = np.uint8(overlay)
        
        return overlay
    except Exception as e:
        st.error(f"Error creating Grad-CAM: {str(e)}")
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

def get_medical_explanation(prediction, confidence, detections, is_eli5=False):
    """Get medical explanation including both classification and detection results"""
    category = CATEGORIES[prediction]
    # detection_info = f"YOLO detected {len(detections)} tumor region(s)" if detections else "No tumor regions detected by YOLO"
    
    if is_eli5:
        prompt = f"""
        Explain in simple terms as if explaining to a child:
        
        The brain image shows: {category}
        
        Explain these results in very simple, understandable terms without using medical terminology. 
        Write as if explaining to a 10-year-old child.
        Respond in English and use maximum 150 words.
        """
    else:
        prompt = f"""
        Provide medical explanation about: {category}
        
        Explain these results in medical detail:
        - Information about the tumor type
        - Significance of the detection results
        - Possible symptoms
        - General treatment approaches
        - General information about prognosis
        
        Respond in English and use professional medical language.
        Use maximum 350 words.
        """
    
    return call_ollama(prompt)

def get_expert_medical_answer(question, prediction, confidence, detections):
    """Get expert medical answer for user questions"""
    category = CATEGORIES[prediction]
    detection_info = f"YOLO detected {len(detections)} tumor region(s)" if detections else "No tumor regions detected by YOLO"
    
    prompt = f"""
You are a very expert in brain and neurosurgery. You have extensive knowledge about brain tumors, neurological conditions, surgical procedures, and patient care.

Current Patient Analysis Context:
- Brain MRI Classification Result: {category}
- Classification Confidence: {confidence*100:.1f}%
- Detection Result: {detection_info}

Patient Question: "{question}"

Please provide a comprehensive, professional medical response based on your expertise in neurosurgery and brain tumor management. Address the patient's specific question while considering the current analysis results.

Guidelines for your response:
- Use professional medical terminology appropriately
- Provide evidence-based information
- Be empathetic and understanding
- Include relevant details about brain anatomy, tumor characteristics, or treatment options as needed
- Always emphasize the importance of consulting with healthcare professionals
- If the question is outside your expertise or requires specific patient data, acknowledge limitations

Respond in English and provide a thorough answer (maximum 400 words).
"""
    
    return call_ollama(prompt)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">Brain Tumor Detection & Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered brain tumor detection with YOLO and classification with MobileNetV2</p>', unsafe_allow_html=True)
    
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
    
    # Settings section at the top
    st.markdown("""
    <div class="settings-container">
        <div class="settings-title">Analysis Settings</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use containers for better organization
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
    
    # Model loading
    classification_model = load_classification_model()
    yolo_model = load_yolo_model()
    
    if classification_model is None:
        st.error("Classification model could not be loaded. Please ensure 'best_brain_tumor_model.pth' file exists.")
        return
    
    if yolo_model is None:
        st.error("YOLO model could not be loaded. Please ensure 'best.pt' file exists.")
        return
    
    # File upload section
    st.markdown("---")  # Separator
    st.markdown("### Upload Brain MR Image")
    st.markdown("Select a brain MRI scan in JPG, JPEG or PNG format for analysis")
    
    uploaded_file = st.file_uploader(
        "Choose file...",
        type=['jpg', 'jpeg', 'png'],
        help="The uploaded image should be a brain MRI scan.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Create main columns for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, caption="MR image to be analyzed", width=350)
        
        with col2:
            st.markdown("#### Analysis Results")
            
            # Classification first to get predicted_class
            with st.spinner("Analyzing image for classification..."):
                image_tensor = preprocess_image_for_classification(image)
                prediction, confidence, probabilities = predict_tumor_classification(classification_model, image_tensor)
            
            predicted_class = CATEGORIES[prediction]
            
            # YOLO Detection (runs always but only shows results if not "No Tumor")
            if show_yolo:
                with st.spinner("Running YOLO tumor detection..."):
                    detections, yolo_annotated_image = detect_tumor_with_yolo(yolo_model, image)
                
                # Only show YOLO results if classification is not "No Tumor"
                if predicted_class != "No Tumor":
                    st.markdown(f"""
                    <div class="yolo-container" style="margin-top: 0;">
                        <div class="prediction-title">YOLO Detection Results</div>
                        <div class="prediction-details">
                            <p><strong>Detected Regions:</strong> {len(detections)}</p>
                            {f'<p><strong>Highest Confidence:</strong> {max([d["confidence"] for d in detections])*100:.2f}%</p>' if detections else '<p><strong>Status:</strong> No tumors detected</p>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                detections = []
            
            # Show classification results
            margin_top = "1rem" if (show_yolo and predicted_class != "No Tumor") else "0"
            st.markdown(f"""
            <div class="prediction-container" style="margin-top: {margin_top};">
                <div class="prediction-title">Classification: {predicted_class}</div>
                <div class="prediction-details">
                    <p><strong>Description:</strong> {CATEGORY_DESCRIPTIONS[predicted_class]}</p>
                    <p><strong>Confidence Level:</strong> {confidence*100:.2f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # st.markdown("#### Analysis Results")
            
            # # YOLO Detection
            # if show_yolo:
            #     with st.spinner("Running YOLO tumor detection..."):
            #         detections, yolo_annotated_image = detect_tumor_with_yolo(yolo_model, image)
                
            #     st.markdown(f"""
            #     <div class="yolo-container" style="margin-top: 0;">
            #         <div class="prediction-title">YOLO Detection Results</div>
            #         <div class="prediction-details">
            #             <p><strong>Detected Regions:</strong> {len(detections)}</p>
            #             {f'<p><strong>Highest Confidence:</strong> {max([d["confidence"] for d in detections])*100:.2f}%</p>' if detections else '<p><strong>Status:</strong> No tumors detected</p>'}
            #         </div>
            #     </div>
            #     """, unsafe_allow_html=True)
            # else:
            #     detections = []
            
            # # Classification
            # with st.spinner("Analyzing image for classification..."):
            #     image_tensor = preprocess_image_for_classification(image)
            #     prediction, confidence, probabilities = predict_tumor_classification(classification_model, image_tensor)
            
            # # Show classification results
            # predicted_class = CATEGORIES[prediction]
            
            # st.markdown(f"""
            # <div class="prediction-container" style="margin-top: 1rem;">
            #     <div class="prediction-title">Classification: {predicted_class}</div>
            #     <div class="prediction-details">
            #         <p><strong>Description:</strong> {CATEGORY_DESCRIPTIONS[predicted_class]}</p>
            #         <p><strong>Confidence Level:</strong> {confidence*100:.2f}%</p>
            #     </div>
            # </div>
            # """, unsafe_allow_html=True)
        
        # YOLO Detection Visualization
        if show_yolo and detections:
            st.markdown("---")
            st.markdown("#### YOLO Detection Visualization")
            
            yolo_col1, yolo_col2 = st.columns([1, 1])
            with yolo_col1:
                st.markdown("#### Original Image")
                st.image(image, caption="Original MRI", width=350)
            with yolo_col2:
                st.markdown("#### Detected Tumors")
                st.image(yolo_annotated_image, caption="YOLO Detection Results", width=350)
            
            # Detection details
            if detections:
                st.markdown("#### Detection Details")
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    st.markdown(f"""
                    <div class="category-item">
                        <strong>Detection {i+1}:</strong><br>
                        Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})<br>
                        Confidence: {conf*100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
        
        # Confidence scores for all classes
        st.markdown("---")
        st.markdown("#### Classification Confidence Scores")
        
        # Create a container for confidence scores
        confidence_container = st.container()
        
        with confidence_container:
            for i, (category, prob) in enumerate(zip(CATEGORIES, probabilities)):
                color = "#74b9ff" if i == prediction else "#7f8c8d"
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
                gradcam_img = create_gradcam_visualization(classification_model, image_tensor, image)
            
            if gradcam_img is not None:
                gradcam_col1, gradcam_col2 = st.columns([1, 1], gap="medium")
                with gradcam_col1:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("#### Original Image")
                    st.image(image.resize((224, 224)), caption="Original MRI", width=300)
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
                explanation = get_medical_explanation(prediction, confidence, detections, is_eli5)
            
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
            # st.markdown("""
            # <div class="qa-container">
            #     <div class="qa-title">🧠 Ask the Neurosurgery Expert</div>
            #     <div class="qa-input-container">
            #         <div class="expert-badge">🩺 Expert Brain & Neurosurgery Specialist</div>
            #     </div>
            # </div>
            # """, unsafe_allow_html=True)
            
            # User question input
            user_question = st.text_area(
                "Ask your question about brain tumors, treatment options, or any related medical concerns:",
                placeholder="For example: What are the treatment options for this type of tumor? What symptoms should I watch for? How serious is this condition?",
                height=100,
                key=f"medical_question_{st.session_state.question_counter}"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            
            with col_ask:
                if st.button("🔍 Get Expert Answer", type="primary", use_container_width=True):
                    if user_question.strip():
                        with st.spinner("Consulting neurosurgery expert..."):
                            expert_answer = get_expert_medical_answer(user_question, prediction, confidence, detections)
                        
                        # Store Q&A in session state
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
                
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5 Q&As
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
    

# Run the main application
if __name__ == "__main__":
    main()
                            
