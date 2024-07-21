import streamlit as st
import openai
import torch
from sentence_transformers import SentenceTransformer, util
import json
import fitz  # PyMuPDF for extracting text from PDFs

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load and preprocess the corpus
pdf_path = 'corpus.pdf'
corpus_text = extract_text_from_pdf(pdf_path)
corpus_chunks = corpus_text.split('\n\n')  # Split based on paragraphs or other delimiters

# Load SentenceTransformer model and precompute embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
corpus_embeddings = model.encode(corpus_chunks, convert_to_tensor=True)

# Load Q&A data from JSON file
with open('saq.json', 'r') as f:
    saq_data = json.load(f)

# Set OpenAI API key
openai.api_key = 'your-api-key'  # Replace with your actual OpenAI API key

# Initialize session state and cache
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'cache' not in st.session_state:
    st.session_state['cache'] = {}
if 'context' not in st.session_state:
    st.session_state['context'] = ""

# Function to retrieve the most relevant answer from the corpus
def retrieve_answer(question, threshold=0.5):
    question_embedding = model.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)
    
    if top_result[0].item() < threshold:
        return None  # Indicating no relevant answer found in the corpus
    
    return corpus_chunks[top_result[1].item()]

# Function to generate response
def generate_response(history, question):
    # Check cache first
    if question in st.session_state['cache']:
        return st.session_state['cache'][question]
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150
    )
    answer = response.choices[0].message['content'].strip()
    
    # Cache the response
    st.session_state['cache'][question] = answer
    return answer

# Streamlit UI
st.title("Wine Business Chatbot")
st.write("Ask me anything about our wines!")

user_input = st.text_input("You: ")

if user_input:
    st.session_state['history'].append({"role": "user", "content": user_input})
    st.session_state['context'] += f"User: {user_input}\n"
    
    # Check if the question matches any in the sample Q&A
    saq_answer = next((item['answer'] for item in saq_data if item['question'].lower() in user_input.lower()), None)
    
    if saq_answer:
        answer = saq_answer
    else:
        retrieved_answer = retrieve_answer(user_input)
        if retrieved_answer:
            st.session_state['context'] += f"Bot: {retrieved_answer}\n"
            answer = generate_response(st.session_state['history'], retrieved_answer)
        else:
            answer = "Please contact the business directly for this information."
    
    st.session_state['history'].append({"role": "bot", "content": answer})
    st.session_state['context'] += f"Bot: {answer}\n"
    
    for msg in st.session_state['history']:
        st.write(f"{msg['role']}: {msg['content']}")

# Maintaining the conversation context
if user_input and len(st.session_state['history']) > 1:
    last_user_message = st.session_state['history'][-2]['content']
    question_with_context = f"{st.session_state['context']}User: {user_input}"
    retrieved_answer_with_context = retrieve_answer(question_with_context)
    
    if retrieved_answer_with_context:
        answer_with_context = generate_response(st.session_state['history'], retrieved_answer_with_context)
        st.session_state['history'].append({"role": "bot", "content": answer_with_context})
        st.session_state['context'] += f"Bot: {answer_with_context}\n"
        st.write(f"bot: {answer_with_context}")
    else:
        st.session_state['history'].append({"role": "bot", "content": "Please contact the business directly for this information."})
        st.session_state['context'] += "Bot: Please contact the business directly for this information.\n"
        st.write(f"bot: Please contact the business directly for this information.")
