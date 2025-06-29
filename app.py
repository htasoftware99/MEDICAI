import streamlit as st
import sys
import os
import importlib.util

# Page configuration - MUST BE CALLED FIRST
st.set_page_config(
    page_title="MEDICAI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get current directory and add subdirectories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_dir = os.path.join(current_dir, 'patient', 'diabetes_disease_prediction')
medical_chatbot_dir = os.path.join(current_dir, 'patient', 'medical_chatbot')
heart_disease_dir = os.path.join(current_dir, 'patient', 'heart_disease_prediction')
brain_tumor_dir = os.path.join(current_dir, 'patient', 'brain_tumor_detection')
pneumonia_dir = os.path.join(current_dir, 'patient', 'pneumonia_detection')

# Add directories to Python path
sys.path.append(diabetes_dir)
sys.path.append(medical_chatbot_dir)
sys.path.append(heart_disease_dir)
sys.path.append(brain_tumor_dir)
sys.path.append(pneumonia_dir)

# Function to clear session state when switching apps
def clear_session_state():
    """Clear all session state except selected_app and previous_app"""
    keys_to_keep = ['selected_app', 'previous_app']
    keys_to_remove = []
    
    for key in st.session_state.keys():
        if key not in keys_to_keep:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]

# Sidebar navigation with collapsible menu
with st.sidebar:
    st.title("⚕️ MEDICAI")
    st.markdown("---")
    
    # Initialize session state for selected app if not exists
    if 'selected_app' not in st.session_state:
        st.session_state.selected_app = "Medical Chatbot"
    
    # Create menu buttons
    if st.button("🤖 Medical Chatbot", use_container_width=True):
        if st.session_state.selected_app != "Medical Chatbot":
            st.session_state.selected_app = "Medical Chatbot"
            clear_session_state()
            st.rerun()
    
    if st.button("🧠 Brain Tumor Detection", use_container_width=True):
        if st.session_state.selected_app != "Brain Tumor Detection":
            st.session_state.selected_app = "Brain Tumor Detection"
            clear_session_state()
            st.rerun()
    
    if st.button("🫁 Pneumonia Detection", use_container_width=True):
        if st.session_state.selected_app != "Pneumonia Detection":
            st.session_state.selected_app = "Pneumonia Detection"
            clear_session_state()
            st.rerun()
    
    if st.button("❤️ Heart Disease Prediction", use_container_width=True):
        if st.session_state.selected_app != "Heart Disease Prediction":
            st.session_state.selected_app = "Heart Disease Prediction"
            clear_session_state()
            st.rerun()
    
    if st.button("🩸 Diabetes Disease Prediction", use_container_width=True):
        if st.session_state.selected_app != "Diabetes Disease Prediction":
            st.session_state.selected_app = "Diabetes Disease Prediction"
            clear_session_state()
            st.rerun()

# Function to dynamically load and run a module
def load_and_run_module(module_name, file_path, function_name):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Module file not found: {file_path}")
            return

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            st.error(f"Could not create spec for module: {module_name}")
            return

        module = importlib.util.module_from_spec(spec)
        
        # Temporarily disable set_page_config to prevent conflicts
        original_set_page_config = st.set_page_config
        st.set_page_config = lambda **kwargs: None
        
        # Execute module
        spec.loader.exec_module(module)
        
        # Restore original set_page_config
        st.set_page_config = original_set_page_config
        
        # Run the main function of the module
        if hasattr(module, function_name):
            getattr(module, function_name)()
        else:
            st.error(f"Function {function_name} not found in {module_name}")
    except Exception as e:
        st.error(f"Could not load {module_name} module: {str(e)}")

# Run selected application based on session state
if st.session_state.selected_app == "Medical Chatbot":
    chatbot_path = os.path.join(medical_chatbot_dir, "medical_chatbot.py")
    load_and_run_module("medical_chatbot", chatbot_path, "main")
elif st.session_state.selected_app == "Brain Tumor Detection":
    brain_tumor_path = os.path.join(brain_tumor_dir, "brain_tumor.py")
    load_and_run_module("brain_tumor", brain_tumor_path, "main")
elif st.session_state.selected_app == "Pneumonia Detection":
    pneumonia_path = os.path.join(pneumonia_dir, "pneumonia_detection.py")
    load_and_run_module("pneumonia_detection", pneumonia_path, "main")
elif st.session_state.selected_app == "Heart Disease Prediction":
    heart_disease_path = os.path.join(heart_disease_dir, "heart_disease.py")
    load_and_run_module("heart_disease", heart_disease_path, "app")
elif st.session_state.selected_app == "Diabetes Disease Prediction":
    diabetes_path = os.path.join(diabetes_dir, "diabetes.py")
    load_and_run_module("diabetes", diabetes_path, "app")