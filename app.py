import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

# Function to clear memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Load the model and tokenizer once
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_path = "distilbert-base-uncased"  # Use a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)  # Load model with lower precision
    model.eval()
    return embedder, model, tokenizer

# Load the dataset
@st.cache_data
def load_data(file_path, chunk_size=1000):
    return pd.read_csv(file_path, chunksize=chunk_size)

# Initialize Faiss index
def initialize_index(dimension=384):
    return faiss.IndexFlatL2(dimension)

# Encode data and add to index
def process_chunks(df_iterator, embedder, index):
    for chunk in df_iterator:
        vectors = embedder.encode(chunk['ayah_en'].tolist(), convert_to_numpy=True, show_progress_bar=False)
        vectors = vectors.astype(np.float32)
        index.add(vectors)
        clear_memory()

# Streamlit app
def main():
    st.title("Quranic Ayah Retrieval and Response Generation")

    # Load models
    embedder, model, tokenizer = load_models()

    # Path to your CSV file
    csv_file_path = "quran_dataset.csv"

    # Load data
    df_iterator = load_data(csv_file_path)

    # Initialize Faiss index
    index = initialize_index()

    # Process data in chunks
    with st.spinner("Processing data..."):
        process_chunks(df_iterator, embedder, index)
        st.success("Data processed and indexed successfully!")

    # Input text from the user
    input_text = st.text_input("Enter your query about the Quranic Ayahs")

    if st.button("Search"):
        if input_text:
            input_vector = embedder.encode([input_text], convert_to_numpy=True, show_progress_bar=False)
            k = 5
            distances, indices = index.search(input_vector.astype(np.float32), k)

            retrieved_ayahs = []
            for idx in indices[0]:
                for chunk in load_data(csv_file_path):
                    if idx < len(chunk):
                        retrieved_ayahs.append(chunk.iloc[idx])
                        break
                    idx -= len(chunk)

            st.write("Retrieved Ayahs:")
            for i, ayah in enumerate(retrieved_ayahs, 1):
                st.write(f"\n**Ayah {i}:**")
                for column, value in ayah.items():
                    st.write(f"{column}: {value}")

            retrieved_texts = "\n".join([ayah['ayah_en'] for ayah in retrieved_ayahs])

            input_tokens = tokenizer(retrieved_texts, return_tensors="pt", truncation=True, max_length=512)
            input_tokens = {key: val.to("cpu") for key, val in input_tokens.items()}

            with torch.no_grad():
                output = model.generate(
                    **input_tokens,
                    max_length=512,
                    pad_token_id=tokenizer.pad_token_id,
             
