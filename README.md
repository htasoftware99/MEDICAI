# Project Setup Guide

This is a Python-based Streamlit application. Follow the steps below to run the project in your local environment.

## Requirements

- Python 3.7 or higher
- Python 3.11 (recommended)
- pip (Python package manager)

## Installation Steps

### 1. Create Virtual Environment

Navigate to the project directory and create a virtual environment using Python venv:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 2. Install Required Packages

Install the required packages from the requirements file:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root directory and add the following line:

```
GROQ_API_KEY=your_groq_api_key_here
```

**Note:** Replace `your_groq_api_key_here` with your actual GROQ API key.

### 4. Run the Application

Make sure you are in the project folder and run the following command:

```bash
streamlit run app.py
```

Once the application starts successfully, it will automatically open in your browser or you can visit the local URL shown in the console (usually `http://localhost:8501`).

## Troubleshooting

- If you get a `streamlit` command not found error, make sure the virtual environment is activated
- If you encounter API key errors, check that the `.env` file is in the correct location and the API key is valid
- If you experience package installation issues, update pip: `pip install --upgrade pip`