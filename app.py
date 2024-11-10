from huggingface_hub import from_pretrained
import torch

import streamlit as st
from transformers import pipeline

# Initialize the pipelines for summarization and sentiment analysis
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# Set up the app layout
st.set_page_config(page_title="Text Summarizer & Sentiment Analyzer", layout="centered")
st.title("📄 Text Summarizer & Sentiment Analyzer")
st.write("Analyze the sentiment and get a summary of any text you enter. Powered by Hugging Face's Transformers.")

# User input section
st.subheader("Enter Your Text")
user_input = st.text_area("Paste your text here", height=200, help="Enter the text you want to analyze.")

# Check if text is provided
if user_input:
    # Sidebar options
    st.sidebar.title("🔧 Options")
    option = st.sidebar.radio("Choose Action:", ("Summarize Text", "Analyze Sentiment"))

    # Show results based on selected action
    if option == "Summarize Text":
        st.subheader("🔍 Summary")
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarizer(user_input, max_length=50, min_length=25, do_sample=False)
                st.success("Summary generated successfully!")
                st.write(summary[0]['summary_text'])

    elif option == "Analyze Sentiment":
        st.subheader("📊 Sentiment Analysis")
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                sentiment = sentiment_analyzer(user_input)
                label = sentiment[0]['label']
                score = sentiment[0]['score']
                
                # Display result with icon
                if label == "POSITIVE":
                    st.success(f"Sentiment: {label} (Confidence: {score:.2f})")
                else:
                    st.error(f"Sentiment: {label} (Confidence: {score:.2f})")

else:
    st.warning("Please enter some text to get started.")

# Footer
st.write("---")
st.caption("Developed for Hackathon - Quick and Simple Text Analysis App")
