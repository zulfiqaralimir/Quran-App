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
def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_path = "ibm-granite/granite-3b-code-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float32)
    model.eval()
    return embedder, model, tokenizer

# Load the dataset
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

# Main application
def main():
    st.title("Quranic Ayah Retrieval and Response Generation")

    # Load models
    embedder, model, tokenizer = load_models()

    # Path to your CSV file
    csv_file_path = "/content/quran_dataset.csv"

    # Load data
    df_iterator = load_data(csv_file_path)

    # Initialize Faiss index
    index = initialize_index()

    # Process data in chunks
    process_chunks(df_iterator, embedder, index)

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
            for i in input_tokens:
                input_tokens[i] = input_tokens[i].to("cpu")

            with torch.no_grad():
                output = model.generate(
                    **input_tokens,
                    max_length=512,
                    pad_token_id=tokenizer.pad_token_id,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            st.write("**Generated Response:**")
            st.write(generated_text)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
