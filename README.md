# 🩺 Skin Disease Detection Using CNN, Generative AI & RAG

An AI-powered web application that helps users identify possible skin diseases from images and provides useful medical information using **Deep Learning, Generative AI, and Retrieval-Augmented Generation (RAG)**.

The system combines a **Convolutional Neural Network (CNN)** for image classification with **Ollama and ChromaDB** to retrieve medical knowledge and generate easy-to-understand explanations.

> **Note:** This project is designed for educational purposes and AI-assisted screening. It does not replace professional medical diagnosis.

---

## 📌 Project Overview

Skin diseases can be difficult to identify because many conditions have similar symptoms.

This project uses Artificial Intelligence to help users understand possible skin conditions by analyzing images and providing information about symptoms, causes, precautions, and general treatment options.

The application also helps users search for nearby dermatologists and hospitals.

## ✨ Key Features

- 🖼️ **Skin Disease Detection:** Analyze uploaded skin images using a CNN model.
- 🤖 **Generative AI:** Generate easy-to-understand medical explanations.
- 📚 **RAG Integration:** Retrieve relevant information from a medical knowledge base.
- 🧠 **ChromaDB:** Store and search medical information using vector embeddings.
- 💬 **Ollama LLM:** Generate explanations using a locally running language model.
- 🏥 **Dermatologist Finder:** Search for nearby dermatologists and hospitals.
- 🔐 **User Authentication:** Secure registration and login system.
- 🌐 **Web Application:** Simple interface developed using Flask.

## 🔍 Skin Diseases Covered

The system includes six skin disease categories:

1. Atopic Dermatitis
2. Contact Dermatitis
3. Eczema
4. Scabies
5. Seborrheic Dermatitis
6. Tinea Corporis

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application |
| TensorFlow / Keras | CNN model |
| CNN | Skin image classification |
| Ollama | Running local AI models |
| Llama 3.2 (3B) | Generating medical explanations |
| RAG | Retrieving relevant medical knowledge |
| ChromaDB | Vector database |
| nomic-embed-text | Text embeddings |
| SQLite | User database |
| HTML, CSS, JavaScript | Frontend development |

## ⚙️ How the System Works

**Step 1 — Upload Image**

The user selects a skin disease image through the web application.

**Step 2 — Image Analysis**

The CNN model processes the image and predicts a possible skin disease category.

**Step 3 — Medical Knowledge Retrieval**

The RAG system searches ChromaDB for relevant medical information.

**Step 4 — AI Explanation**

The Ollama language model generates an explanation using retrieved knowledge.

**Step 5 — Display Results**

The application displays information about the skin condition, including symptoms, causes, precautions, and general care.

**Step 6 — Find Dermatologists**

Users can search for nearby dermatologists and hospitals using their location or PIN code.

### System Workflow

```text
          User
           |
     Upload Skin Image
           |
      CNN Model
           |
    Disease Prediction
           |
       RAG System
           |
   ChromaDB Vector Database
           |
  Retrieve Medical Knowledge
           |
    Ollama (Llama 3.2)
           |
  Generate AI Explanation
           |
      Display Result
           |
   Find Nearby Dermatologist
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/skin-disease-genai.git
cd skin-disease-genai
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Required Libraries

Make sure Python is installed.

```bash
python -m pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from:

https://ollama.com/

### 4. Download AI Models

Open a terminal and execute:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

These models are used for text embeddings and AI-generated explanations.

### 5. Prepare the Medical Knowledge Base

The project contains medical knowledge files covering six skin diseases.

Build the vector database:

```bash
python rag/build_knowledge_base.py
```

This converts medical text into embeddings and stores them in ChromaDB.

### 6. Test the RAG System

```bash
python rag/query_knowledge_base.py
```

Enter a question to retrieve medical knowledge and generate an answer.

Type `exit` to stop.

### 7. Run the Flask Application

Ensure the trained CNN model and class labels are available in the expected `outputs` folder.

```bash
python app.py
```

Open your browser and visit:

http://127.0.0.1:5000/

## 📂 Medical Knowledge Base

The knowledge base contains medical information related to six skin diseases.

Each text file provides reference information used by the RAG system.

The `sources.csv` file records the reference pages used to prepare the starter content.

You can update the knowledge files with additional reviewed information and rebuild the vector database.

## 🧠 CNN vs RAG vs Generative AI

**CNN (Convolutional Neural Network)**

Analyzes skin images and predicts disease categories.

**RAG (Retrieval-Augmented Generation)**

Searches the medical knowledge base to retrieve relevant information.

**Generative AI (Ollama + Llama 3.2)**

Uses retrieved information to generate understandable explanations.

Together, these technologies combine image classification with knowledge-based text generation.

## 🎯 Project Objectives

- Develop an AI-assisted skin disease screening application.
- Classify skin images using deep learning.
- Integrate Generative AI for medical explanations.
- Improve information retrieval using RAG and ChromaDB.
- Provide users with understandable skin health information.
- Help users locate dermatologists for professional consultation.

## 🔮 Future Enhancements

- Support more skin disease categories.
- Add multilingual explanations.
- Improve CNN accuracy using larger, validated datasets.
- Develop a mobile application.
- Integrate verified medical knowledge sources.
- Add appointment booking functionality.

## ⚠️ Medical Disclaimer

This application is intended for educational and research purposes only.

AI predictions may be incorrect, and similar skin conditions can appear alike. The generated information should not be considered a confirmed diagnosis or prescription.

Always consult a qualified dermatologist for proper diagnosis and treatment.

---

## 📄 License

Add a license file to define how others may use, modify, and distribute this project.

## ⭐ Support

If you find this project useful, consider giving the GitHub repository a star!

**Developed using Python, CNN, Flask, Generative AI, Ollama, RAG, and ChromaDB.**
