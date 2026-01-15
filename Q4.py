# app.py
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
nltk.download("punkt")
nltk.download("punkt_tab")

# -----------------------------
# NLTK setup (Step 3)
# -----------------------------
nltk.download("punkt", quiet=True)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="PDF Text Chunking (NLTK)", layout="wide")
st.title("PDF Text Chunking Web App (NLTK Sentence Tokenizer)")
st.write("Upload a PDF → extract text → split into sentences → show sentences **index 58 to 68** → perform sentence chunking.")

# -----------------------------
# Step 1: Upload PDF + PdfReader
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is None:
    st.info("Please upload a PDF file to start.")
    st.stop()

# -----------------------------
# Step 2: Extract text from PDF
# -----------------------------
reader = PdfReader(uploaded_file)
all_text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:  # avoid None
        all_text += page_text + "\n"

if not all_text.strip():
    st.error("No text could be extracted from this PDF (it might be scanned images).")
    st.stop()

st.success("PDF text extracted successfully!")

# -----------------------------
# Step 3: Preprocess (split into sentences) + show sample 58 to 68
# -----------------------------
sentences = sent_tokenize(all_text)

st.subheader("Step 3: Total Sentences Extracted")
st.write(f"Total sentences found: **{len(sentences)}**")

start_idx, end_idx = 58, 68

st.subheader(f"Step 3: Sample Extracted Text (Sentence Index {start_idx} to {end_idx})")

if len(sentences) <= start_idx:
    st.warning(
        f"This PDF only has {len(sentences)} sentences, so it doesn't reach index {start_idx}."
    )
    st.write("Here are the last few sentences instead:")
    st.write(sentences[-10:])
else:
    sample_sentences = sentences[start_idx:end_idx + 1]
    for i, s in enumerate(sample_sentences, start=start_idx):
        st.write(f"**[{i}]** {s}")

# -----------------------------
# Step 4: Apply NLTK sentence tokenizer for semantic sentence chunking
# (Here: chunk = each sentence, grouped into blocks for readability)
# -----------------------------
st.subheader("Step 4: Semantic Sentence Chunking (Grouped Sentences)")

chunk_size = st.slider("Sentences per chunk", min_value=1, max_value=10, value=3)

chunks = []
current = []
for s in sentences:
    current.append(s)
    if len(current) == chunk_size:
        chunks.append(" ".join(current))
        current = []

if current:
    chunks.append(" ".join(current))

st.write(f"Total chunks created: **{len(chunks)}**")

# Show a few chunks (not all, to keep it clean)
show_n = st.number_input("How many chunks to display?", min_value=1, max_value=min(20, len(chunks)), value=min(5, len(chunks)))

for idx in range(int(show_n)):
    st.markdown(f"### Chunk {idx+1}")
    st.write(chunks[idx])
