def summarize_text(text: str, max_words: int = 30) -> dict:
    """Simple extractive summarizer — returns first N words."""
    words = text.split()
    if len(words) <= max_words:
        return {"summary": text}
    summary = " ".join(words[:max_words]) + "..."
    return {
        "summary": summary,
        "original_word_count": len(words),
        "summary_word_count": max_words,
    }