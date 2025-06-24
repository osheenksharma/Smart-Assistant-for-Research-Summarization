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
| Component               | Technology                                  |
| ----------------------- | ------------------------------------------- |
| UI Framework            | Streamlit                                   |
| PDF Extraction          | `PyMuPDF` or custom utility                 |
| Summarization           | Hugging Face Transformers (e.g., BART, T5)  |
| Q\&A + Justification    | DistilBERT, Sentence-BERT                   |
| Semantic Scoring        | `sentence-transformers` (cosine similarity) |
| Knowledge Graph (2D/3D) | TF-IDF + SVD + Plotly + Graphviz            |

##Architecture Diagram Description
The system architecture consists of four main layers:

1. User Interface Layer (Streamlit Frontend)
Provides file upload interface (st.file_uploader)
Offers interaction modes:
Ask Questions: Text input and conversational history
Test Knowledge: Form-based question answering
Visualizations:
Executive Summary
Knowledge Graph (3D Concept Map + Mind Map)

2. Application Logic Layer
Manages session states (st.session_state)
Controls UI components and user interactions
Orchestrates document processing and routes to back-end logic

3. NLP Engine Layer
This layer comprises custom logic and pretrained transformer models:
PDF/Text Extractor: extract_text_from_pdf (PDF parsing)
Summarizer: summarize_text() (abstractive summarization via transformer models)
Q&A Engine (qa_engine.py)
ask_question_from_doc() → Uses DistilBERT QA pipeline
get_justification_snippet() → Uses Sentence-BERT for semantic similarity
generate_logic_questions() → Static or generative logic question creation
evaluate_user_answer() → Semantic similarity scoring with Sentence-BERT

4. Visualization Layer
Mind Map: Built using Graphviz based on high TF-IDF terms
3D Concept Map: Built with Plotly using SVD-reduced TF-IDF vectors

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
