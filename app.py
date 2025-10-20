import os
import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer

st.title("🧠 Intelligent Issue Search")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Neon using env variable
conn = psycopg2.connect(os.environ["NEON_CONN"])
cur = conn.cursor()

query = st.text_input("Describe your issue:")
if query:
    q_emb = model.encode([query])[0].tolist()
    cur.execute("""
        SELECT title, solution, 1 - (embedding <=> %s::vector) AS sim
        FROM issues ORDER BY embedding <=> %s::vector LIMIT 5;
    """, (q_emb, q_emb))
    for t, s, sc in cur.fetchall():
        st.markdown(f"**Issue:** {t}\n\n**Solution:** {s}\n\n_Similarity: {sc:.2f}_\n---")
