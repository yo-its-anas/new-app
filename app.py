from transformers import pipeline

# Specify `use_fast=False` to avoid requiring Rust-based tokenizers
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", use_fast=False)
sentiment_analyzer = pipeline("sentiment-analysis", use_fast=False)

import streamlit as st

st.title("Text Summarizer and Sentiment Analysis")

# Input box for text
text = st.text_area("Enter text here:")

# Summarize text
if st.button("Summarize"):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    st.write("Summary:", summary[0]['summary_text'])

# Sentiment analysis
if st.button("Analyze Sentiment"):
    sentiment = sentiment_analyzer(text)
    st.write("Sentiment:", sentiment[0]['label'])
