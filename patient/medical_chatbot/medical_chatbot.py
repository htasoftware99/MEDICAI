import streamlit as st
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Health Assistant - Medical Consultation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Information
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DOCKER_API_URL = "http://localhost:10500/api/chat"
DOCKER_MODEL_NAME = "llama3.2:3b"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-size: 2.5rem !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
        border-radius: 15px;
        padding: 2rem;
        color: white;
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
</style>
""", unsafe_allow_html=True)

# Health-focused prompts
HEALTH_SYSTEM_PROMPTS = {
    "image_analysis": """You are a professional medical AI assistant specializing in visual health analysis. 
    Analyze the provided image carefully and provide helpful medical insights while following these guidelines:
    
    IMPORTANT DISCLAIMERS:
    - Always remind users that this is not a substitute for professional medical advice
    - Encourage consulting healthcare professionals for serious concerns
    - Do not provide specific diagnoses, only general observations and suggestions
    
    ANALYSIS APPROACH:
    - Describe what you observe in the image objectively
    - Identify potential health-related concerns if visible
    - Provide general wellness recommendations
    - Suggest when professional medical consultation is needed
    
    RESPONSE FORMAT:
    1. **Visual Observations**: What you see in the image
    2. **Potential Concerns**: Any health-related observations (if applicable)
    3. **General Recommendations**: Wellness and care suggestions
    4. **Professional Consultation**: When to seek medical help
    
    Keep responses informative yet accessible, and always prioritize user safety.""",
    
    "text_consultation": """You are a compassionate AI health assistant designed to help with daily health concerns and wellness questions.
    
    YOUR ROLE:
    - Provide helpful information about common health topics
    - Offer wellness and lifestyle recommendations
    - Guide users on when to seek professional medical care
    - Support mental and physical wellbeing
    
    IMPORTANT GUIDELINES:
    - Never provide specific medical diagnoses
    - Always recommend professional medical consultation for serious symptoms
    - Focus on general wellness, prevention, and self-care
    - Provide evidence-based health information
    - Be empathetic and supportive
    
    RESPONSE AREAS:
    ✓ General wellness tips
    ✓ Symptom awareness (when to see a doctor)
    ✓ Healthy lifestyle recommendations
    ✓ Basic first aid information
    ✓ Mental health support and resources
    ✓ Nutrition and exercise guidance
    
    Always end responses with: "Remember, this information is for educational purposes only. Please consult healthcare professionals for personalized medical advice."
    
    Be caring, informative, and always prioritize the user's safety and wellbeing."""
}

# GROQ API Key check
if not GROQ_API_KEY:
    st.error("🔑 GROQ API KEY is not set in the .env file")
    st.stop()

# Image processing function (using GROQ Vision API)
def process_image(image, query):
    try:
        image_content = image.read()
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # Visual validity check
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}
        
        # Combine system prompt with user query
        enhanced_query = f"{HEALTH_SYSTEM_PROMPTS['image_analysis']}\n\nUser Question: {query}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]
        
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.7
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed image response successfully")
            return answer
        else:
            logger.error(f"Error from GROQ API: {response.status_code} - {response.text}")
            return f"Error from GROQ API: {response.status_code} - {response.text}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

# Text processing function (Ollama in Docker uses Llama3.2:3b model)
def process_text(query):
    try:
        # Combine system prompt with user query
        enhanced_query = f"{HEALTH_SYSTEM_PROMPTS['text_consultation']}\n\nUser Question: {query}"
        
        messages = [{"role": "user", "content": enhanced_query}]
        
        response = requests.post(
            DOCKER_API_URL,
            json={
                "model": DOCKER_MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 1500
                }
            },
            headers={
                "Content-Type": "application/json"
            },
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["message"]["content"]
            logger.info(f"Processed text response successfully")
            return answer
        else:
            logger.error(f"Error from Docker API: {response.status_code} - {response.text}")
            return f"Error from Docker API: {response.status_code} - {response.text}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header"> AI Health Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main input section
    st.markdown("### 📝 How Can I Help You Today?")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📸 Upload Health-Related Image (Optional)")
        uploaded_image = st.file_uploader(
            "Choose a medical image, symptom photo, or health document",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="📷 Uploaded Image", use_container_width=True)
            st.success("✅ Image uploaded successfully!")
    
    with col2:
        st.markdown("#### 💭 Your Health Question")
        query = st.text_area(
            "Describe your symptoms, health concerns, or ask wellness questions:",
            placeholder="Example: 'I have been experiencing headaches and fatigue lately. What could be causing this?' or 'What are some healthy meal options for managing diabetes?'",
            height=150
        )
    
    # Processing section
    if query:
        st.markdown("---")
        
        if uploaded_image:
            
            if st.button("🔍 Analyze Image & Answer Question", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing your image and processing your question..."):
                    response = process_image(uploaded_image, query)
                
                st.markdown("### 🏥 AI Health Analysis Results:")
                st.markdown(response)
                
                # Log the interaction
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Image analysis completed at {timestamp}")
        else:
            
            if st.button("Get Health Guidance", type="primary", use_container_width=True):
                with st.spinner("🔄 Consulting AI health assistant..."):
                    response = process_text(query)
                
                st.markdown("### 🩺 AI Health Consultation Results:")
                st.markdown(response)
                
                # Log the interaction
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Text consultation completed at {timestamp}")
    
    # Medical Disclaimer at the bottom
    st.markdown("---")
    st.markdown("""
    <div class="warning-container">
        ⚠️ <strong>Medical Disclaimer</strong><br>
        This AI system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare professionals for medical concerns. The AI explanations are generated based on general medical knowledge and may not reflect the most current medical practices or be applicable to individual cases.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()