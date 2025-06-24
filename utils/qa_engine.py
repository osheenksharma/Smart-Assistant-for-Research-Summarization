from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

text_gen = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

def ask_question_from_doc(question, context):
    result = qa_pipeline(question=question, context=context)
    answer = result["answer"]
    score = result["score"]

    justification, justification_score = get_justification_snippet(answer, context)

    return (
        f"**Answer:** {answer}\n\n"
        f"**Confidence:** {round(score * 100, 2)}%\n\n"
        f"**Based on:** _{justification}_\n"
        f"**Justification Score:** {round(justification_score * 100, 2)}%"
    )

def get_justification_snippet(answer, context):
    sentences = context.split(".")
    answer_emb = semantic_model.encode(answer, convert_to_tensor=True)

    best_score = 0
    best_sent = ""

    for sent in sentences:
        sent = sent.strip()
        if sent:
            sent_emb = semantic_model.encode(sent, convert_to_tensor=True)
            score = util.pytorch_cos_sim(answer_emb, sent_emb).item()
            if score > best_score:
                best_score = score
                best_sent = sent

    if answer.lower() in best_sent.lower():
        start = best_sent.lower().find(answer.lower())
        end = start + len(answer)
        highlighted = best_sent[:start] + f"**{best_sent[start:end]}**" + best_sent[end:]
    else:
        highlighted = best_sent or "No exact supporting sentence found."

    return highlighted, best_score


def generate_logic_questions(document_text):
    trimmed_text = document_text[:1000] if len(document_text) > 1000 else document_text

    prompt = (
        "Generate 3 logic-based and comprehension questions based on the following academic text:\n\n"
        f"{trimmed_text}\n\n"
        "Questions:\n1."
    )

    try:
        result = text_gen(prompt, max_length=300, do_sample=True, top_k=50, temperature=0.8)
        raw = result[0]['generated_text'].split("Questions:")[-1]
        lines = raw.strip().split("\n")

        questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                questions.append(line.lstrip("1234567890.- ").strip())
            if len(questions) == 3:
                break

        return questions if questions else ["Unable to generate questions."]
    
    except Exception as e:
        print("Error in question generation:", str(e))
        return ["Failed to generate questions."]

def evaluate_user_answer(document_text, question, user_answer):
    expected_answer = qa_pipeline(question=question, context=document_text)["answer"]

    embedding1 = semantic_model.encode(expected_answer, convert_to_tensor=True)
    embedding2 = semantic_model.encode(user_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    if similarity > 0.7:
        return "✅ Correct! Your answer aligns well with the document."
    else:
        return (
            f"❌ Not quite. Expected something like: '{expected_answer}'. "
            "Consider reviewing that section again."
        )
