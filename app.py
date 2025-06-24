import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from graphviz import Digraph
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from utils.pdf_reader import extract_text_from_pdf
from utils.summarizer import summarize_text
from utils.qa_engine import ask_question_from_doc, generate_logic_questions, evaluate_user_answer

st.set_page_config(
    page_title="NeuroScholar | Smart Assistant for Research Summarization",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --accent: #4776e6;
        --light: #f8f9fa;
        --dark: #f8f9fa;
        --card-dark: #2b2f36;
    }

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }

    .stApp {
        background: transparent;
    }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 12px !important;
        padding: 12px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        background-color: var(--card-dark) !important;
        color: white !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.3s !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }

    .stRadio>div {
        gap: 10px;
    }

    .stRadio>div>label {
        border-radius: 12px !important;
        padding: 10px 20px !important;
        background: var(--card-dark) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.3s !important;
    }

    .stRadio>div>label:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    .stRadio>div>label[data-baseweb="radio"]>div:first-child {
        border-color: var(--primary) !important;
    }

    .stRadio>div>label[data-baseweb="radio"]>div:first-child>div {
        background-color: var(--primary) !important;
    }

    .stExpander {
        background: var(--card-dark) !important;
        color: white !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        border: none !important;
    }

    .stExpander .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--light) !important;
    }

    .card {
        background: var(--card-dark) !important;
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .gradient-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)


def create_knowledge_graph(text):
    """Create a 3D knowledge graph from document text"""
    try:
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()

        if X.T.shape[1] >= 3:
            svd = TruncatedSVD(n_components=3)
            coords = svd.fit_transform(X.T)
        else:
            coords = np.random.rand(len(terms), 3)

        nodes = pd.DataFrame({
            'term': terms,
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'size': np.asarray(X.sum(axis=0)).flatten() * 50,
            'color_r': np.random.randint(50, 255, len(terms)),
            'color_g': np.random.randint(50, 255, len(terms)),
            'color_b': np.random.randint(50, 255, len(terms))
        })

        co_occurrence = (X.T * X)
        edges = []
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if co_occurrence[i, j] > 0.1:
                    edges.append({
                        'source': i,
                        'target': j,
                        'value': co_occurrence[i, j] * 10
                    })

        return nodes, pd.DataFrame(edges)

    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        return None, None


def render_knowledge_graph(nodes, edges):
    """Render interactive knowledge graph visualization"""
    if nodes is None or edges is None:
        return

    with st.spinner("Rendering knowledge graph..."):
        tab1, tab2 = st.tabs(["Mind Map", "Concept Map"])

        with tab1:

            if not nodes.empty:
                root_term = nodes.loc[nodes['size'].idxmax(), 'term']
                child_terms = nodes[nodes['term'] != root_term].sort_values(by='size', ascending=False).head(5)['term']

                dot = Digraph()

                dot.node(root_term, root_term, shape='ellipse', style='filled', color='lightblue')

                for child in child_terms:
                    dot.node(child, child)
                    dot.edge(root_term, child)

                for child in child_terms:
                    grandchildren = nodes[nodes['term'] != child].sample(2)['term']
                    for grandchild in grandchildren:
                        dot.node(grandchild, grandchild)
                        dot.edge(child, grandchild)

                st.graphviz_chart(dot)
            else:
                st.info("No data to render Mind Map.")

        with tab2:
            fig = go.Figure()

            # Add nodes
            fig.add_trace(go.Scatter3d(
                x=nodes['x'],
                y=nodes['y'],
                z=nodes['z'],
                mode='markers+text',
                marker=dict(
                    size=nodes['size'] / 10,
                    color=[f'rgb({r},{g},{b})' for r, g, b in zip(nodes['color_r'], nodes['color_g'], nodes['color_b'])],
                    opacity=0.8,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=nodes['term'],
                textposition="middle center",
                hoverinfo='text',
                hovertext=nodes['term']
            ))

            for _, edge in edges.iterrows():
                src = nodes.iloc[int(edge['source'])]
                tgt = nodes.iloc[int(edge['target'])]

                fig.add_trace(go.Scatter3d(
                    x=[src['x'], tgt['x']],
                    y=[src['y'], tgt['y']],
                    z=[src['z'], tgt['z']],
                    mode='lines',
                    line=dict(
                        color='rgba(150,150,150,0.5)',
                        width=edge['value'] / 2
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("nteractive 3D concept map showing relationships between key terms.")


def main():
    with st.container():
        st.markdown("""
        <div class="gradient-header">
            <div style="display: flex; align-items: center; gap: 20px;">
                <h1 style="margin: 0; font-size: 2.5rem;">🧠 NeuroScholar</h1>
                <div style="font-size: 1.2rem; opacity: 0.9;">AI-Powered Research Intelligence Platform</div>
            </div>
            <div style="margin-top: 10px; font-size: 1rem; opacity: 0.8;">
                Extract insights, visualize knowledge, and accelerate your research workflow
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        uploaded_file = st.file_uploader(
            "📤 Upload Research Document (PDF or TXT)", 
            type=["pdf", "txt"],
            help="Upload your academic paper, research document, or article for analysis"
        )
    
    if uploaded_file:
        if ("uploaded_file_name" not in st.session_state or 
            st.session_state.uploaded_file_name != uploaded_file.name):
            st.session_state.raw_text = None
            st.session_state.summary = None
            st.session_state.questions = None
            st.session_state.qa_history = []
            st.session_state.uploaded_file_name = uploaded_file.name
        
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully")
        
        if st.session_state.raw_text is None:
            with st.spinner("🔍 Extracting document content..."):
                if uploaded_file.type == "application/pdf":
                    st.session_state.raw_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    st.session_state.raw_text = uploaded_file.read().decode("utf-8")
        
        if st.session_state.raw_text:
            with st.expander("Document Knowledge Graph", expanded=True):
                nodes, edges = create_knowledge_graph(st.session_state.raw_text)
                render_knowledge_graph(nodes, edges)
        
        if st.session_state.summary is None:
            with st.spinner("Generating intelligent summary..."):
                st.session_state.summary = summarize_text(st.session_state.raw_text)
        
        with st.expander("Executive Summary", expanded=True):
            st.markdown(f"""
            <div class="card fade-in">
                <div style="font-size: 0.9rem; color: #555; margin-bottom: 10px;">
                    {len(st.session_state.summary.split())} words | Key Insights
                </div>
                <div style="line-height: 1.6;">
                    {st.session_state.summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        
        mode = st.radio(
            "Select Interaction Mode:",
            ["Ask Questions", "Test Knowledge"],
            horizontal=True,
            help="Choose how you want to interact with the document"
        )
        
        if mode == "Ask Questions":
            with st.container():
                st.markdown("""
                <div class="card fade-in">
                    <h3 style="margin-top: 0;">Ask About the Document</h3>
                """, unsafe_allow_html=True)
                
                user_question = st.text_input(
                    "Enter your question:", 
                    placeholder="What is the main hypothesis of this research?",
                    key="question_input"
                )
                
                if user_question:
                    with st.spinner("Analyzing document..."):
                        answer = ask_question_from_doc(user_question, st.session_state.raw_text)
                        st.session_state.qa_history.append((user_question, answer))
                    
                    st.markdown(f"""
                    <div class="card fade-in" style="background-color: #f8f9fa; border-left: 4px solid var(--primary);">
                        <div style="font-weight: 500; margin-bottom: 8px;">Answer:</div>
                        <div style="line-height: 1.6;">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.session_state.qa_history:
                    st.markdown("""
                    <div class="card fade-in">
                        <h4 style="margin-top: 0;">Conversation History</h4>
                    """, unsafe_allow_html=True)
                    
                    for q, a in reversed(st.session_state.qa_history):
                        st.markdown(f"""
                        <div style="margin-bottom: 20px;">
                            <div style="font-weight: 500; color: var(--primary);">Q: {q}</div>
                            <div style="margin-left: 15px; margin-top: 5px; color: #f8f9fa;">{a}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        else:  
            with st.container():
                st.markdown("""
                <div class="card fade-in">
                    <h3 style="margin-top: 0;">Comprehension Challenge</h3>
                    <p style="color: #555;">Test your understanding with AI-generated questions</p>
                """, unsafe_allow_html=True)
                
                if st.button("Generate Questions", key="generate_questions"):
                    with st.spinner("Crafting thought-provoking questions..."):
                        st.session_state.questions = generate_logic_questions(st.session_state.raw_text)
                
                if "questions" in st.session_state and st.session_state.questions:
                    with st.form("challenge_form"):
                        responses = {}
                        for i, q in enumerate(st.session_state.questions):
                            st.markdown(f"**Question {i+1}:** {q}")
                            responses[q] = st.text_area(
                                f"Your answer", 
                                key=f"user_ans_{i}",
                                height=100,
                                placeholder="Type your answer here..."
                            )
                        
                        submitted = st.form_submit_button("Submit Answers")
                    
                    if submitted:
                        st.markdown("""
                        <div class="card fade-in">
                            <h4 style="margin-top: 0;">Evaluation Results</h4>
                        """, unsafe_allow_html=True)
                        
                        for q, user_ans in responses.items():
                            if user_ans.strip():
                                with st.spinner(f"Evaluating: {q[:50]}..."):
                                    feedback = evaluate_user_answer(st.session_state.raw_text, q, user_ans)
                                
                                if "correct" in feedback.lower():
                                    st.markdown(f"""
                                    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4caf50;">
                                        <div style="font-weight: 500; color: #2e7d32;">✅ Correct</div>
                                        <div style="margin-top: 5px; color: #555;">{feedback}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #f44336;">
                                        <div style="font-weight: 500; color: #c62828;">⚠️ Needs Improvement</div>
                                        <div style="margin-top: 5px; color: #555;">{feedback}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 40px 20px;">
            <h3 style="margin-top: 0;">Welcome to NeuroScholar</h3>
            <p style="color: #555; margin-bottom: 25px;">
                Upload a research document to begin extracting insights, visualizing knowledge relationships, 
                and testing your understanding.
            </p>
            <div style="font-size: 3rem; margin-bottom: 20px;">📄 ➔ 🧠</div>
            <p style="color: #777;">
                Supports PDF and TXT documents. Your data is processed securely and never stored.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px 0; color: #666; font-size: 0.9rem;">
        <hr style="margin-bottom: 15px;">
        NeuroScholar AI Research Assistant | © 2023 | Ethical AI for Academic Excellence
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()