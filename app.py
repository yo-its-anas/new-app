import streamlit as st
from transformers import pipeline

# Initialize pipelines without fast tokenizers
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", use_fast=False)
sentiment_analyzer = pipeline("sentiment-analysis", use_fast=False)

st.title("Text Summarizer and Sentiment Analysis")

# Input for text
text = st.text_area("Enter text here:")

if st.button("Summarize"):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    st.write("Summary:", summary[0]['summary_text'])

if st.button("Analyze Sentiment"):
    sentiment = sentiment_analyzer(text)
    st.write("Sentiment:", sentiment[0]['label'])
