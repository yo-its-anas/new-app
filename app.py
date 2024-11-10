import streamlit as st
from transformers import pipeline

# Streamlit app UI
st.title("Text Summarizer and Sentiment Analysis")

# Input text box
input_text = st.text_area("Enter Text for Summarization and Sentiment Analysis")

# Load transformer models for summarization and sentiment analysis
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# Button to process input text
if st.button("Process Text"):
    if input_text:
        # Summarize the text
        summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
        sentiment = sentiment_analyzer(input_text)

        # Display results
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])
        st.subheader("Sentiment Analysis:")
        st.write(f"Label: {sentiment[0]['label']} (Confidence: {sentiment[0]['score']*100:.2f}%)")
    else:
        st.warning("Please enter some text to process.")

