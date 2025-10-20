# app.py
import os
import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer

# Title
st.set_page_config(page_title="AI Issue Search", layout="wide")
st.title("🧠 Intelligent Issue Search")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Connect to Neon DB using Streamlit Secrets
def get_db_connection():
    conn = psycopg2.connect(os.environ["NEON_CONN"])
    return conn

conn = get_db_connection()
cur = conn.cursor()

# User input
query = st.text_input("Describe your issue:")

if query:
    # Generate embedding
    q_emb = model.encode([query])[0].tolist()
    
    # Search for similar issues
    cur.execute("""
        SELECT title, solution, 1 - (embedding <=> %s::vector) AS similarity
        FROM issues
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """, (q_emb, q_emb))
    
    results = cur.fetchall()
    
    if results:
        st.subheader("🔍 Similar Issues Found:")
        for title, solution, sim in results:
            st.markdown(f"**Issue:** {title}\n\n**Solution:** {solution}\n\n_Similarity: {sim:.2f}_\n---")
    else:
        st.info("No similar issues found. Try a different description!")

# Optional: show a little footer
st.markdown(
    "<sub>Prototype AI Issue Management System • Powered by Streamlit & Neon</sub>", 
    unsafe_allow_html=True
)
