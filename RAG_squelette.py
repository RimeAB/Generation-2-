import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------
# 1. Chargement documents
# -----------------------

documents = open("corpus.txt").read().split("\n\n")  # chunk simple

# -----------------------
# 2. Embeddings
# -----------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -----------------------
# 3. Modèle génératif (Instruct)
# -----------------------

model_name = "mistralai/Ministral-8B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# -----------------------
# 4. Pipeline RAG
# -----------------------

def retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

def generate_answer(query):
    context = "\n\n".join(retrieve(query))
    
    prompt = f"""Vous êtes un assistant administratif INALCO.
Répondez uniquement à partir du contexte suivant.

Contexte:
{context}

Question:
{query}

Réponse:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------
# 5. Test
# -----------------------

question = "Comment obtenir une attestation de scolarité ?"
print(generate_answer(question))
