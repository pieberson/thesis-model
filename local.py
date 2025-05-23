import streamlit as st
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

# Load sample dataset
with open("sample_papers.json", "r") as f:
    papers = json.load(f)

# Initialize models (you can download once, then cache if needed)
bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

st.title("Local Academic Search Engine")

query = st.text_input("Enter your academic query:", "transformer models for academic search")

if query:
    paper_texts = [p['title'] + ". " + p['abstract'] for p in papers]
    
    # Embed papers and query
    paper_embeddings = bi_encoder.encode(paper_texts, convert_to_tensor=True)
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    
    # Compute semantic similarity scores
    similarities = util.pytorch_cos_sim(query_embedding, paper_embeddings)[0]
    
    top_k = min(5, len(papers))
    top_results = torch.topk(similarities, k=top_k)
    
    # Re-rank with cross-encoder
    cross_inp = [(query, paper_texts[i]) for i in top_results.indices]
    cross_scores = cross_encoder.predict(cross_inp)
    
    ranked = sorted(zip(top_results.indices.tolist(), cross_scores), key=lambda x: x[1], reverse=True)
    
    st.subheader("Top Papers")
    for rank, (idx, score) in enumerate(ranked, start=1):
        paper = papers[idx]
        st.markdown(f"""
**{rank}.** _Score: {score:.4f}_  
**Title**: [{paper['title']}]({paper['url']})  
**Citations**: {paper['citationCount']}  
**Abstract**: {paper['abstract']}  
""")
