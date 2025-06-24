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

                         ┌──────────────────────────────┐
                         │      User Interface (UI)     │
                         │        [Streamlit App]       │
                         │  - Upload Doc (PDF/TXT)      │
                         │  - Ask Questions / QA Test   │
                         │  - View Summary / Graphs     │
                         └────────────┬─────────────────┘
                                      │
                                      ▼
                     ┌────────────────────────────────────┐
                     │      Application Controller         │
                     │ - Session State Management          │
                     │ - Mode Selection Logic              │
                     └────────────┬────────────────────────┘
                                  │
    ┌─────────────────────────────┼────────────────────────────────────┐
    │                             │                                    │
    ▼                             ▼                                    ▼
┌──────────────┐     ┌────────────────────────┐            ┌─────────────────────┐
│  Extractor   │     │     Summarizer         │            │ Knowledge Graph     │
│ (PDF/Text)   │     │ summarize_text()       │            │ (TF-IDF + SVD +     │
│              │     │                        │            │ Plotly/Graphviz)    │
└──────────────┘     └────────────────────────┘            └─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │        QA Engine            │
                    │ ┌─────────────────────────┐ │
                    │ │ ask_question_from_doc() │ │
                    │ │ get_justification_snip()│ │
                    │ │ generate_logic_questions│ │
                    │ │ evaluate_user_answer()  │ │
                    │ └─────────────────────────┘ │
                    └─────────────────────────────┘


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
