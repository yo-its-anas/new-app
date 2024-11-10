import streamlit as st
from transformers import pipeline

# Initialize the text summarization and sentiment analysis pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI
st.title("Text Summarizer and Sentiment Analysis App")

# Text input for summarization
input_text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if input_text:
        summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])
    else:
        st.write("Please enter some text to summarize.")

# Text input for sentiment analysis
input_text_sentiment = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if input_text_sentiment:
        sentiment = sentiment_analyzer(input_text_sentiment)
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Label: {sentiment[0]['label']}, Confidence: {sentiment[0]['score']}")
    else:
        st.write("Please enter some text for sentiment analysis.")
