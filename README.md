# Smart-Assistant-for-Research-Summarization
An AI-powered tool for summarizing and interacting with research documents.

## Features
**Document Upload**: Supports PDF and TXT files
**Auto Summary**: Generates concise 150-word summaries
**Ask Anything**: Free-form question answering with document references
**Challenge Me**: Generates comprehension questions and evaluates answers
**Contextual Understanding**: All answers are grounded in the source document
**Reference Tracking**: Shows supporting evidence for each answer

## Technology Stack
Streamlit
Python
PyMuPDF
HuggingFace Transformers
scikit-learn
Torch
NumPy

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 8GB+ RAM (16GB recommended for LLaMA 2 7B)

### Step-by-Step Setup
Clone the repository:
1. git clone

2.Create and activate a virtual environment: python -m venv venv 

venv\Scripts\activate.bat

3. Install dependencies: pip install -r requirements.txt
   
4.Run the app: streamlit run app.py
