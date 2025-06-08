# Project Setup Guide

This is a Python-based Streamlit application that uses Ollama for local AI model processing. Follow the steps below to run the project in your local environment.

## Requirements

- Python 3.7 or higher
- Python 3.11 (recommended)
- pip (Python package manager)
- Docker Desktop
- Ollama

## Installation Steps

### 1. Install Docker Desktop

Download and install Docker Desktop from the official website:
- **Windows/macOS:** [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Linux:** Follow the installation guide for your distribution at [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

Make sure Docker Desktop is running before proceeding to the next steps.

### 2. Install Ollama with Docker

Pull the official Ollama Docker image:

```bash
docker pull ollama/ollama
```

Run Ollama container:

```bash
docker run -d -v ollama:/root/.ollama -p 10500:11434 --name ollama ollama/ollama
```

**Verify Ollama container is running:**
```bash
docker ps
```

You should see the `ollama` container in the running containers list.

### 3. Install Required AI Model

Install the Llama 3.2:3b model inside the Docker container:

```bash
docker exec -it ollama ollama pull llama3.2:3b
```

**Verify model installation:**
```bash
docker exec -it ollama ollama list
```

You should see `llama3.2:3b` in the list of installed models.

### 4. Create Virtual Environment

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

### 5. Install Required Packages

Install the required packages from the requirements file:

```bash
pip install -r requirements.txt
```

### 6. Set Up Environment Variables

Create a `.env` file in the project root directory and add the following line:

```
GROQ_API_KEY=your_groq_api_key_here
```

**Note:** Replace `your_groq_api_key_here` with your actual GROQ API key.

### 7. Run the Application

Make sure all the above steps are completed successfully, then run the following command:

```bash
streamlit run app.py
```

Once the application starts successfully, it will automatically open in your browser or you can visit the local URL shown in the console (usually `http://localhost:8501`).

## Verification Checklist

Before running the application, ensure:

- [ ] Docker Desktop is installed and running
- [ ] Ollama Docker container is running (`docker ps` shows ollama container)
- [ ] Llama 3.2:3b model is downloaded and available in the Docker container
- [ ] Python virtual environment is created and activated
- [ ] All required packages are installed via pip
- [ ] Environment variables are properly configured in `.env` file

## Troubleshooting

- **Docker issues:** Make sure Docker Desktop is running and you have sufficient system resources
- **Ollama container not running:** Check container status with `docker ps` and restart if needed: `docker start ollama`
- **Ollama model not found:** Verify the model is installed with `docker exec -it ollama ollama list` command
- **Connection issues:** Ensure Ollama container is accessible on port 10500
- **Streamlit command not found:** Make sure the virtual environment is activated
- **API key errors:** Check that the `.env` file is in the correct location and the API key is valid  
- **Package installation issues:** Update pip with `pip install --upgrade pip`
- **Port conflicts:** If port 8501 is in use, Streamlit will automatically use the next available port

## System Requirements

- **RAM:** Minimum 8GB (16GB recommended for optimal performance)
- **Storage:** At least 10GB free space for models and dependencies
- **Internet:** Required for initial setup and model downloads