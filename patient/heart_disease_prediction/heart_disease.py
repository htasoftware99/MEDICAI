import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
import time
import os
from typing import List, Dict
import tempfile

# RAG-specific imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

st.set_page_config(
    page_title="Heart Disease Prediction and Medical Report Analysis",
    page_icon="♥️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# Load the trained model and scaler
model = pickle.load(open("models/heart_disease.pkl", "rb"))
scaler = pickle.load(open("models/heart_disease_scaler.pkl", "rb"))

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize persistent ChromaDB
CHROMA_PERSIST_DIR = "patient/heart_disease_prediction/chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Initialize session state for documents
if 'report_documents' not in st.session_state:
    st.session_state.report_documents = {}

# Vector store client
vectorstore = None
try:
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    print(f"Loaded existing Chroma DB with {vectorstore._collection.count()} documents")
except Exception as e:
    print(f"Creating new Chroma DB: {e}")
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

# Request explanation from Llama3.2:3b API
def get_llama_response(prompt):
    try:
        llama_endpoint = "http://localhost:10500/api/generate"
        payload = json.dumps({"model": "llama3.2:3b", "prompt": prompt, "stream": False})
        headers = {"Content-Type": "application/json"}
        response = requests.post(llama_endpoint, data=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "No response from Llama model.")
    except requests.exceptions.RequestException as e:
        return f"Error contacting Llama API: {str(e)}"

# Enhanced function to process PDF files with better metadata
def process_pdf(pdf_file, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Extract patient name from content (improved extraction)
    patient_name = extract_patient_name(documents, file_name)
    
    # Add enhanced metadata to track the source file and patient
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "source": file_name,
            "patient_name": patient_name,
            "page": i + 1,
            "doc_type": "medical_report"
        })
    
    # Split documents into chunks with better overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add patient name to chunk metadata as well
    for chunk in chunks:
        chunk.metadata["patient_name"] = patient_name
        chunk.metadata["source"] = file_name
    
    # Store documents in session state
    st.session_state.report_documents[file_name] = {
        "chunks": chunks,
        "patient_name": patient_name,
        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to vectorstore
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    # Clean up temp file
    os.unlink(pdf_path)
    
    return len(chunks), patient_name

# Function to extract patient name from document content
def extract_patient_name(documents, file_name):
    # Common patterns for patient names in medical reports
    import re
    
    full_text = " ".join([doc.page_content for doc in documents[:2]])  # Check first 2 pages
    
    # Pattern matching for common formats
    patterns = [
        r"Patient[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
        r"Name[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
        r"Patient Name[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
        r"MR\.\s+([A-Z][a-z]+ [A-Z][a-z]+)",
        r"MS\.\s+([A-Z][a-z]+ [A-Z][a-z]+)",
        r"MRS\.\s+([A-Z][a-z]+ [A-Z][a-z]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: use filename without extension
    return file_name.replace('.pdf', '').replace('_', ' ').title()

# Enhanced function to retrieve relevant context with patient filtering
def retrieve_context(query, patient_name=None, k=3):
    if vectorstore._collection.count() == 0:
        return "No documents in database."
    
    try:
        # If patient name is specified, include it in the search
        if patient_name:
            enhanced_query = f"{query} {patient_name}"
        else:
            enhanced_query = query
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(enhanced_query)
        
        # Filter documents by patient name if specified
        if patient_name:
            filtered_docs = []
            for doc in docs:
                doc_patient = doc.metadata.get('patient_name', '').lower()
                if patient_name.lower() in doc_patient or doc_patient in patient_name.lower():
                    filtered_docs.append(doc)
            docs = filtered_docs if filtered_docs else docs[:2]  # Use first 2 if no exact match
        
        if not docs:
            return f"No relevant information found for {patient_name if patient_name else 'this query'}."
        
        context = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            patient = doc.metadata.get('patient_name', 'Unknown')
            context += f"Document {i+1} (Patient: {patient}, Source: {source}):\n{doc.page_content}\n\n"
        
        return context.strip()
        
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

# Function to extract patient name from query
def extract_patient_from_query(query):
    import re
    
    # Common name patterns
    name_patterns = [
        r"([A-Z][a-z]+ [A-Z][a-z]+)",  # First Last
        r"(Jane Smith|John Doe)",  # Specific names you mentioned
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, query)
        if matches:
            return matches[0]
    
    return None

# Enhanced compare reports function
def compare_reports(report_ids: List[str]):
    if not report_ids or len(report_ids) < 2:
        return "Please select at least two reports to compare."
    
    report_contents = {}
    patient_names = {}
    
    for report_id in report_ids:
        if report_id in st.session_state.report_documents:
            report_data = st.session_state.report_documents[report_id]
            chunks = report_data["chunks"]
            patient_names[report_id] = report_data["patient_name"]
            
            # Combine chunks into full content
            content = "\n".join([chunk.page_content for chunk in chunks])
            if len(content) > 4000:
                content = content[:4000] + "..."
            report_contents[report_id] = content
    
    if len(report_contents) < 2:
        return "Could not find enough valid reports to compare."
    
    # Create comparison prompt
    comparison_prompt = f"""As a cardiologist, compare these medical reports and highlight key differences:

"""
    
    for report_id, content in report_contents.items():
        patient_name = patient_names.get(report_id, "Unknown")
        comparison_prompt += f"Patient: {patient_name}\nReport: {content}\n\n---\n\n"
    
    comparison_prompt += "Please provide: 1) Key similarities 2) Important differences 3) Clinical insights"
    
    return get_llama_response(comparison_prompt)

def app():
    st.markdown('<h1 class="main-header">Heart Disease Prediction and Medical Report Analysis</h1>', unsafe_allow_html=True)

    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'display_answer' not in st.session_state:
        st.session_state.display_answer = False
        
    if 'form_key' not in st.session_state:
        st.session_state.form_key = 0
        
    if 'uploaded_reports' not in st.session_state:
        st.session_state.uploaded_reports = []
    
    if 'report_documents' not in st.session_state:
        st.session_state.report_documents = {}

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "PDF Reports", "Report Comparison", "Chat"])
    
    with tab1:
        st.header("Heart Disease Prediction")
        
        eli5_mode = st.checkbox("Explain like I'm 5 (Simplified)")

        with st.form(key='heart_disease_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=80, value=30)
                sex = st.selectbox("Sex", options=["Male", "Female"])
                cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
                trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
                chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
                restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2])
            
            with col2:
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
                exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
                slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
                ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4])
                thal = st.selectbox("Thalassemia (1-3)", options=[1, 2, 3])
            
            submit_button = st.form_submit_button("Predict")

        if submit_button:
            st.session_state.messages = []
            st.session_state.display_answer = False
            st.session_state.form_key += 1
            
            # Convert categorical inputs
            sex_val = 1 if sex == 'Male' else 0
            fbs_val = 1 if fbs == 'Yes' else 0
            exang_val = 1 if exang == 'Yes' else 0

            # Create input dataframe
            user_input = pd.DataFrame({
                'age': [age], 'sex': [sex_val], 'cp': [cp], 'trestbps': [trestbps],
                'chol': [chol], 'fbs': [fbs_val], 'restecg': [restecg], 
                'thalach': [thalach], 'exang': [exang_val], 'oldpeak': [oldpeak],
                'slope': [slope], 'ca': [ca], 'thal': [thal]
            })

            # Scale and predict
            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)

            # Display result
            if prediction[0] == 0:
                st.error("Result: Heart disease risk detected.")
                prompt = "Explain heart disease risk briefly." if eli5_mode else "As a cardiologist, explain the heart disease risk and its implications."
            else:
                st.success("Result: No heart disease risk detected.")
                prompt = "Explain to a 5-year-old child that their heart is healthy and strong, using simple words and happy examples." if eli5_mode else "As a cardiologist, explain why no heart disease risk was detected."

            explanation = get_llama_response(prompt)
            st.info("Medical Explanation:")
            st.write(explanation)
    
    with tab2:
        st.header("Upload Medical Reports")
        
        uploaded_files = st.file_uploader("Upload patient PDF reports", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in [report["name"] for report in st.session_state.uploaded_reports]:
                    with st.spinner(f"Processing {file.name}..."):
                        chunk_count, patient_name = process_pdf(file, file.name)
                        st.session_state.uploaded_reports.append({
                            "name": file.name,
                            "patient_name": patient_name,
                            "chunks": chunk_count,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                    #st.success(f"Processed {file.name} for patient {patient_name} - {chunk_count} chunks")
        
        # Display uploaded reports
        if st.session_state.uploaded_reports:
            st.subheader("Uploaded Reports")
            for idx, report in enumerate(st.session_state.uploaded_reports):
                st.write(f"{idx+1}. **{report['name']}**")
            
            # Select report for analysis
            report_options = [f"{report['name']} - {report['patient_name']}" for report in st.session_state.uploaded_reports]
            selected_report_display = st.selectbox("Select a report to analyze", report_options)
            
            if selected_report_display and st.button("Analyze Report"):
                # Extract the actual filename
                selected_report = selected_report_display.split(' - ')[0]
                
                with st.spinner("Analyzing report..."):
                    if selected_report in st.session_state.report_documents:
                        report_data = st.session_state.report_documents[selected_report]
                        patient_name = report_data["patient_name"]
                        chunks = report_data["chunks"]
                        
                        # Combine content
                        report_content = "\n".join([chunk.page_content for chunk in chunks])
                        if len(report_content) > 4000:
                            report_content = report_content[:4000] + "..."
                        
                        analysis_prompt = f"""As a cardiologist, analyze this medical report for {patient_name}:

{report_content}

Provide: 1) Key findings 2) Heart-related concerns 3) Recommendations"""
                        
                        analysis = get_llama_response(analysis_prompt)
                        st.markdown(f"### Analysis for {patient_name}")
                        st.write(analysis)
    
    with tab3:
        st.header("Compare Medical Reports")
        
        if len(st.session_state.uploaded_reports) >= 2:
            report_options = [f"{report['name']} - {report['patient_name']}" for report in st.session_state.uploaded_reports]
            selected_reports_display = st.multiselect("Select reports to compare", report_options)
            
            if selected_reports_display and len(selected_reports_display) >= 2 and st.button("Compare Reports"):
                # Extract actual filenames
                selected_reports = [report.split(' - ')[0] for report in selected_reports_display]
                
                with st.spinner("Comparing reports..."):
                    comparison = compare_reports(selected_reports)
                    st.markdown("### Comparison Results")
                    st.write(comparison)
        else:
            st.info("Upload at least two reports to enable comparison.")
    
    with tab4:
        st.header("Ask about Heart Disease or Reports")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Input for new questions
        query_type = st.radio("Query type:", ["General Heart Disease Question", "Question About Uploaded Reports"])
        
        with st.form(key=f"question_form_{st.session_state.form_key}"):
            question = st.text_input("Your Question:", key=f"question_text_{st.session_state.form_key}")
            ask_button = st.form_submit_button("Ask")
            
        if ask_button and question:
            current_question = question
            
            # Add question to chat
            st.session_state.messages.append({"role": "user", "content": current_question})
            
            if query_type == "General Heart Disease Question":
                prompt = f"As a cardiologist, answer this question: {current_question}"
                response = get_llama_response(prompt)
            else:
                # Enhanced RAG response with patient name extraction
                patient_name = extract_patient_from_query(current_question)
                context = retrieve_context(current_question, patient_name)
                
                prompt = f"""Based on the medical reports, answer: {current_question}

Medical Records:
{context}

Provide a focused medical response as a cardiologist."""
                
                response = get_llama_response(prompt)
            
            # Add response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.session_state.display_answer = True
            st.session_state.form_key += 1
            st.rerun()
            
        # Clear chat button
        if st.session_state.messages and st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.form_key += 1
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="warning-container">
        ⚠️ <strong>Medical Disclaimer</strong><br>
        This AI system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare professionals for medical concerns. The AI explanations are generated based on general medical knowledge and may not reflect the most current medical practices or be applicable to individual cases.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()

