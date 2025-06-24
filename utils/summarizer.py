from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_words=150):
    if len(text.split()) > 600:
        text = " ".join(text.split()[:600])

    summary = summarizer(
        text,
        max_length=200,
        min_length=50,
        do_sample=False
    )[0]['summary_text']

    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "..."

    return summary