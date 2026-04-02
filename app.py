# import os
# import pickle
# import faiss
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_core.prompts import ChatPromptTemplate
# import google.generativeai as genai

# # ----------------------------
# # Load .env and API key
# # ----------------------------
# load_dotenv()  # Load environment variables from .env
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# if not GENAI_API_KEY:
#     st.error("API key not found! Please set GENAI_API_KEY in your .env file.")
#     st.stop()

# genai.configure(api_key=GENAI_API_KEY)

# # ----------------------------
# # CONFIG
# # ----------------------------
# model = genai.GenerativeModel("gemini-2.5-flash")
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# RESUME_FOLDER = "resumes"

# JOB_SKILLS = {
#     "Python Developer": ["python", "django", "sql", "html", "css", "javascript", "react", "fullstack"],
#     "Data Science Developer": ["python", "pandas", "numpy", "matplotlib", "scikit-learn", "ml", "statistics"],
#     "ML Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "numpy", "pandas"],
#     "AI Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "nlp", "cv"],
#     "Fullstack Developer": ["html", "css", "javascript", "react", "python", "django", "sql"]
# }

# # ----------------------------
# # PDF Loader & Chunking
# # ----------------------------
# def load_pdf(file_path):
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"
#     return text

# def chunk_text(text, chunk_size=200, overlap=40):
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_size
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks

# # ----------------------------
# # Build / Load FAISS Index
# # ----------------------------
# def build_index(folder):
#     all_chunks = []
#     metadata = []

#     for file in os.listdir(folder):
#         if not file.endswith(".pdf"):
#             continue
#         text = load_pdf(os.path.join(folder, file))
#         chunks = chunk_text(text)
#         for chunk in chunks:
#             all_chunks.append(chunk)
#             metadata.append({"text": chunk, "source": os.path.basename(file)})

#     embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)

#     faiss.write_index(index, "resume.index")
#     with open("resume_docs.pkl", "wb") as f:
#         pickle.dump(metadata, f)

# def load_index():
#     # Build index if not exists
#     if not os.path.exists("resume.index") or not os.path.exists("resume_docs.pkl"):
#         build_index(RESUME_FOLDER)
#     index = faiss.read_index("resume.index")
#     with open("resume_docs.pkl", "rb") as f:
#         docs = pickle.load(f)
#     return index, docs

# # ----------------------------
# # Retrieval & ATS scoring
# # ----------------------------
# def retrieve(query, index, docs, top_k=5):
#     query_vec = embedder.encode([query], convert_to_numpy=True)
#     distances, indices = index.search(query_vec, top_k)
#     return [docs[i] for i in indices[0]]

# def calculate_ats_score(text, job_role):
#     skills = JOB_SKILLS.get(job_role, [])
#     if not skills:
#         return 0
#     matches = sum(1 for skill in skills if skill.lower() in text.lower())
#     return int((matches / len(skills)) * 100)

# def resume_assistant(query, index, docs, top_k=5, min_score=60):
#     retrieved_docs = retrieve(query, index, docs, top_k)
#     eligible_docs = []

#     for doc in retrieved_docs:
#         score = calculate_ats_score(doc["text"], query)
#         if score >= min_score:
#             eligible_docs.append({**doc, "ats_score": score})

#     if not eligible_docs:
#         return "No eligible resumes found with ATS score >= 60%", []

#     context_text = "\n\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(eligible_docs)])

#     prompt = ChatPromptTemplate.from_template("""
# You are an AI recruitment assistant.

# Analyze the resume content and return structured output.

# Provide:
# 1. Candidate Skills
# 2. Extracted Projects
# 3. ATS Score
# 4. Skill Match with Job Role
# 5. Interview Questions

# Rules:
# - Extract projects only from the resume context.

# Context:
# {context}

# Job Role:
# {job_role}
# """)

#     formatted_prompt = prompt.format(context=context_text, job_role=query)
#     response = model.generate_content(formatted_prompt)

#     ats_scores = {doc["source"]: doc["ats_score"] for doc in eligible_docs}
#     return response.text, ats_scores

# # ----------------------------
# # Streamlit UI
# # ----------------------------
# st.title("📄 Resume Assistant (From Folder)")

# job_role = st.selectbox("Select Job Role", list(JOB_SKILLS.keys()))
# min_score = st.slider("Minimum ATS Score (%)", 0, 100, 60)

# if st.button("Analyze Resumes"):
#     index, docs = load_index()
#     output, ats_scores = resume_assistant(job_role, index, docs, min_score=min_score)

#     st.subheader("🤖 AI Analysis")
#     st.text_area("Resume Assistant Output", output, height=300)

#     st.subheader("📊 ATS Scores")
#     st.table(ats_scores)
import os
import pickle
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ----------------------------
# Load .env and API key
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("API key not found! Please set GROQ_API_KEY in your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# CONFIG
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
RESUME_FOLDER = "resumes"

JOB_SKILLS = {
    "Python Developer": ["python", "django", "sql", "html", "css", "javascript", "react", "fullstack"],
    "Data Science Developer": ["python", "pandas", "numpy", "matplotlib", "scikit-learn", "ml", "statistics"],
    "ML Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "numpy", "pandas"],
    "AI Developer": ["python", "tensorflow", "pytorch", "ml", "ai", "deep learning", "nlp", "cv"],
    "Fullstack Developer": ["html", "css", "javascript", "react", "python", "django", "sql"]
}

# ----------------------------
# PDF Loader & Chunking
# ----------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ----------------------------
# Build / Load FAISS Index
# ----------------------------
def build_index(folder):
    all_chunks = []
    metadata = []

    for file in os.listdir(folder):
        if not file.endswith(".pdf"):
            continue
        text = load_pdf(os.path.join(folder, file))
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"text": chunk, "source": os.path.basename(file)})

    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "resume.index")
    with open("resume_docs.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    if not os.path.exists("resume.index") or not os.path.exists("resume_docs.pkl"):
        build_index(RESUME_FOLDER)

    index = faiss.read_index("resume.index")
    with open("resume_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    return index, docs

# ----------------------------
# Retrieval & ATS scoring
# ----------------------------
def retrieve(query, index, docs, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [docs[i] for i in indices[0]]

def calculate_ats_score(text, job_role):
    skills = JOB_SKILLS.get(job_role, [])
    if not skills:
        return 0
    matches = sum(1 for skill in skills if skill.lower() in text.lower())
    return int((matches / len(skills)) * 100)

# ----------------------------
# Resume Assistant (Groq)
# ----------------------------
def resume_assistant(query, index, docs, top_k=5, min_score=60):
    retrieved_docs = retrieve(query, index, docs, top_k)
    eligible_docs = []

    for doc in retrieved_docs:
        score = calculate_ats_score(doc["text"], query)
        if score >= min_score:
            eligible_docs.append({**doc, "ats_score": score})

    if not eligible_docs:
        return "No eligible resumes found with ATS score >= 60%", {}

    context_text = "\n\n".join(
        [f"[{i+1}] {doc['text']}" for i, doc in enumerate(eligible_docs)]
    )

    prompt = f"""
You are an AI recruitment assistant.

Analyze the resume content and return structured output.

Provide:
1. Candidate Skills
2. Extracted Projects
3. ATS Score
4. Skill Match with Job Role
5. Interview Questions

Rules:
- Extract projects only from the resume context.

Context:
{context_text}

Job Role:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an AI recruitment assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    ats_scores = {doc["source"]: doc["ats_score"] for doc in eligible_docs}

    return response.choices[0].message.content, ats_scores

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Resume Assistant (Groq Powered)")

job_role = st.selectbox("Select Job Role", list(JOB_SKILLS.keys()))
min_score = st.slider("Minimum ATS Score (%)", 0, 100, 60)

if st.button("Analyze Resumes"):
    with st.spinner("Processing resumes..."):
        index, docs = load_index()
        output, ats_scores = resume_assistant(job_role, index, docs, min_score=min_score)

    st.subheader("🤖 AI Analysis")
    st.text_area("Resume Assistant Output", output, height=300)

    st.subheader("📊 ATS Scores")
    st.table(ats_scores)