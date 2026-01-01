import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# removed unused imports: load_qa_chain, OpenAI
import os
import bcrypt
from pymongo import MongoClient
import json
import requests
from datetime import datetime, UTC
import re
import torch
import speech_recognition as sr
import tempfile
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    # Fallback: try alternative import
    try:
        from streamlit_audio_recorder import audio_recorder
    except ImportError:
        audio_recorder = None

# Load environment variables
load_dotenv()

# OpenRouter and model config (use env with sane defaults)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'mistralai/devstral-2512:free')  # <-- default from your image

# quick check
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env")
    st.stop()

# Configure CUDA settings
if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
db = client['pdf_chatbot']
users_collection = db['users']
subjects_collection = db['subjects']
chat_history_collection = db['chat_history']

# session init and auth helpers (unchanged)
def init_session_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def login_user(username, password):
    user = users_collection.find_one({'username': username})
    if user and verify_password(password, user['password']):
        st.session_state.user = {'username': username, 'is_guest': False}
        return True
    return False

def register_user(username, password, confirm_password):
    if password != confirm_password:
        return False, "Passwords do not match"
    if users_collection.find_one({'username': username}):
        return False, "Username already exists"
    hashed_password = hash_password(password)
    users_collection.insert_one({
        'username': username,
        'password': hashed_password,
        'created_at': datetime.now(UTC)
    })
    return True, "Registration successful"

def guest_login():
    st.session_state.user = {'username': f'guest_{datetime.now().timestamp()}', 'is_guest': True}

def get_user_subjects():
    return list(subjects_collection.find({'username': st.session_state.user['username']}))

def add_subject(subject_name):
    if not subject_name.strip():
        return False, "Subject name cannot be empty"
    subjects_collection.insert_one({
        'username': st.session_state.user['username'],
        'name': subject_name,
        'created_at': datetime.now(UTC)
    })
    return True, "Subject added successfully"

def delete_subject(subject_id):
    subjects_collection.delete_one({'_id': subject_id})

def save_chat_message(subject_id, message, response):
    chat_history_collection.insert_one({
        'username': st.session_state.user['username'],
        'subject_id': subject_id,
        'message': message,
        'response': response,
        'timestamp': datetime.now(UTC)
    })

def get_chat_history(subject_id):
    return list(chat_history_collection.find({
        'username': st.session_state.user['username'],
        'subject_id': subject_id
    }).sort('timestamp', 1))

# Speech to text conversion
def speech_to_text(audio_bytes):
    """Convert audio bytes to text using speech recognition"""
    try:
        # Create a temporary file to store audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Read audio file
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
        
        # Recognize speech using Google's free API
        try:
            text = r.recognize_google(audio)
            os.unlink(tmp_path)  # Clean up temp file
            return text
        except sr.UnknownValueError:
            os.unlink(tmp_path)
            return None
        except sr.RequestError as e:
            os.unlink(tmp_path)
            st.error(f"Could not request results from speech recognition service: {e}")
            return None
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# OpenRouter call with conversation context and strict PDF-only answers
def get_openrouter_response(prompt, subject_name, chat_history_context="", has_pdf_context=False):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": "http://localhost:8501",
        "X-Title": "PDF Chatbot"
    }
    
    # Build system message with strict rules
    system_content = f"""You are a PDF-based AI assistant specialized in {subject_name}.

CRITICAL RULES:
1. Answer questions STRICTLY using ONLY the content from the uploaded PDF documents.
2. Do NOT use any external knowledge or information not found in the PDF.
3. If the answer is not found in the PDF context provided, you MUST reply with exactly: "This information is not available in the uploaded document."
4. Keep answers simple, clear, and easy to understand.
5. Maintain conversation context for follow-up questions."""
    
    # Build messages with conversation history
    messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history if available
    if chat_history_context:
        messages.append({"role": "user", "content": f"Previous conversation context:\n{chat_history_context}"})
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": MODEL_NAME,
        "messages": messages
    }
    
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        if 'choices' in result and result['choices']:
            response = result['choices'][0]['message'].get('content', '').strip()
            
            # If no PDF context was provided but user asked a question, return the specific message
            if not has_pdf_context and response and not response.startswith("This information is not available"):
                # Check if response seems to be using external knowledge
                # For now, we'll trust the model with the strict prompt, but we can add validation
                pass
            
            return response
        else:
            err = f"OpenRouter API error: {json.dumps(result)}"
            st.error(err)
            return err
    except requests.exceptions.RequestException as e:
        err = f"Failed to connect to OpenRouter API: {str(e)}"
        st.error(err)
        return err
    except Exception as e:
        err = f"Failed to parse OpenRouter response: {str(e)}"
        st.error(err)
        return err

def display_chat_message(message, is_user=True):
    if is_user:
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; margin: 8px 0;'>
            <div style='background: #3a86ff; color: #fff; padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 70%; box-shadow: 0 2px 8px rgba(58,134,255,0.08); font-size: 1.05rem;'>
                <span style='font-weight: 500;'>You</span><br>{message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-start; margin: 8px 0;'>
            <div style='background: #23272f; color: #e0e0e0; padding: 12px 18px; border-radius: 18px 18px 18px 4px; max-width: 70%; box-shadow: 0 2px 8px rgba(35,39,47,0.08); font-size: 1.05rem;'>
                <span style='font-weight: 500;'>Assistant</span><br>{message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

def main():
    init_session_state()
    if not st.session_state.user:
        st.title("Welcome to MindVault")
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest Login"])
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    if login_user(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup = st.form_submit_button("Sign Up")
                if signup:
                    success, message = register_user(new_username, new_password, confirm_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        with tab3:
            if st.button("Continue as Guest"):
                guest_login()
                st.rerun()
    else:
        st.sidebar.title(f"Welcome, {st.session_state.user['username']}")
        st.sidebar.header("Subjects")
        with st.sidebar.form("add_subject"):
            new_subject = st.text_input("Add New Subject")
            if st.form_submit_button("Add"):
                success, message = add_subject(new_subject)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        subjects = get_user_subjects()
        for subject in subjects:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(subject['name'], key=f"subject_{subject['_id']}"):
                    st.session_state.current_subject = subject
                    st.session_state.chat_history = get_chat_history(subject['_id'])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{subject['_id']}"):
                    delete_subject(subject['_id'])
                    st.rerun()

        if st.session_state.current_subject:
            st.header(f"Chat about {st.session_state.current_subject['name']}")
            # display history
            for chat in st.session_state.chat_history:
                display_chat_message(chat['message'], True)
                display_chat_message(chat['response'], False)

            # PDF upload and processing (better handling & chunking)
            pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
            if pdfs:
                texts = []
                for pdf in pdfs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texts.append(page_text.strip())
                # split into smaller chunks for better RAG
                if texts:
                    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
                    chunks = []
                    for t in texts:
                        chunks.extend(splitter.split_text(t))
                    # create embeddings and store to FAISS (in-memory vector store)
                    try:
                        embeddings = get_embeddings()
                        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                        st.success("Documents processed and stored.")
                    except Exception as e:
                        st.error(f"Error storing embeddings: {e}")
                        st.session_state.vector_store = None

            # Chat interface with text and voice input
            st.subheader("Ask questions about your PDFs")
            
            # Create two columns for text input and microphone
            col1, col2 = st.columns([4, 1])
            
            with col1:
                query = st.text_input("Type your question here", key="text_input", label_visibility="collapsed")
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if audio_recorder:
                    audio_bytes = audio_recorder(text="🎤", icon_size="2x")
                else:
                    audio_bytes = None
                    st.markdown("🎤")
                    st.caption("Voice input")
            
            # Process voice input
            voice_query = None
            if audio_bytes:
                with st.spinner("Converting speech to text..."):
                    voice_text = speech_to_text(audio_bytes)
                    if voice_text:
                        st.success(f"Recognized: {voice_text}")
                        voice_query = voice_text
                    else:
                        st.warning("Could not recognize speech. Please try again or type your question.")
            
            # Use voice query if available, otherwise use text input
            final_query = voice_query if voice_query else query
            
            # Process query (from text or voice)
            if final_query and final_query.strip():
                display_chat_message(final_query, True)
                try:
                    # Build conversation context from chat history
                    chat_context = ""
                    if st.session_state.chat_history:
                        recent_chats = st.session_state.chat_history[-5:]  # Last 5 exchanges for context
                        chat_context = "\n".join([
                            f"User: {chat['message']}\nAssistant: {chat['response']}"
                            for chat in recent_chats
                        ])
                    
                    has_pdf_context = False
                    if st.session_state.vector_store:
                        results = st.session_state.vector_store.similarity_search(query=final_query, k=3)
                        if results:
                            context_chunks = [r.page_content for r in results if getattr(r, 'page_content', None)]
                            if context_chunks:
                                context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
                                prompt = (
                                    "Use ONLY the following context from the uploaded PDF documents to answer the question.\n\n"
                                    f"Context from PDF documents:\n{context}\n\n"
                                    f"Question: {final_query}\n\n"
                                    "If the answer cannot be found in the provided context, reply with exactly: 'This information is not available in the uploaded document.'"
                                )
                                has_pdf_context = True
                            else:
                                prompt = f"Question: {final_query}\n\nNote: No relevant context found in PDF. Reply with exactly: 'This information is not available in the uploaded document.'"
                        else:
                            prompt = f"Question: {final_query}\n\nNote: No relevant context found in PDF. Reply with exactly: 'This information is not available in the uploaded document.'"
                    else:
                        # No PDF uploaded
                        prompt = f"Question: {final_query}\n\nNote: No PDF documents have been uploaded. Reply with exactly: 'This information is not available in the uploaded document.'"
                    
                    # Get response with conversation context
                    response = get_openrouter_response(
                        prompt, 
                        st.session_state.current_subject['name'],
                        chat_history_context=chat_context,
                        has_pdf_context=has_pdf_context
                    )
                    
                    display_chat_message(response, False)
                    save_chat_message(st.session_state.current_subject['_id'], final_query, response)
                    st.session_state.chat_history = get_chat_history(st.session_state.current_subject['_id'])
                    st.session_state.current_query = ""  # Clear query after processing
                    st.rerun()  # Rerun to clear input and update UI
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.current_subject = None
            st.session_state.chat_history = []
            st.rerun()

if __name__ == '__main__':
    main()
