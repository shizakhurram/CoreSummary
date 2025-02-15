import os
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import fitz  # PyMuPDF for extracting text from PDF

# 🔹 Disable Streamlit's file watcher to prevent PyTorch errors
os.environ["STREAMLIT_WATCHDOG"] = "false"

# Load model and tokenizer
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
base_model.eval()  # Ensure model is in evaluation mode

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text("text") + "\n"
    return text.strip()

# Function to summarize text
def summarize_text(text, max_length=150, min_length=50):
    if not text:
        return "No text found in the document."
    
    input_text = "summarize: " + text  # Add prefix for T5 model
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = base_model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(layout="wide")

def main():
    st.title("📄 SmartDoc: AI-Driven Document Summarization")

    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            col1, col2 = st.columns(2)
            with col1:
                st.info("Uploaded File")
                st.write(extracted_text[:1000])  # Display first 1000 characters

            with col2:
                st.info("Summarization in Progress...")
                summary = summarize_text(extracted_text)
                st.success(summary)
        else:
            st.error("No text found in the uploaded document.")

if __name__ == "__main__":
    main()
