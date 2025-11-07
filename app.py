"""
Flask Web Application for Medical Chatbot
Provides a web interface for the RAG-based medical information chatbot.
"""

import os
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.helper import download_embeddings, format_docs
from src.prompt import get_medical_prompt
from src.store_index import PineconeIndexManager

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global variables for chatbot components
retriever = None
rag_chain = None
chatbot_initialized = False


def initialize_chatbot():
    """
    Initialize the chatbot components (embeddings, vector store, retriever, RAG chain).
    This is called once when the app starts.
    """
    global retriever, rag_chain, chatbot_initialized
    
    try:
        print("Initializing Medical Chatbot...")
        
        # Load environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Set Google API key for Gemini
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Initialize embeddings
        print("Loading embeddings model...")
        embeddings = download_embeddings()
        
        # Initialize Pinecone index manager and load existing index
        print("Connecting to Pinecone index...")
        index_manager = PineconeIndexManager(
            api_key=pinecone_api_key,
            index_name="medical-chatbot",
            dimension=384
        )
        
        # Load existing index
        docsearch = index_manager.load_existing_index(embeddings)
        
        # Create retriever
        print("Setting up retriever...")
        retriever = index_manager.get_retriever(
            search_type="similarity",
            k=3
        )
        
        # Initialize chat model
        print("Initializing Gemini chat model...")
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0,
            request_timeout=60,
            max_retries=2
        )
        
        # Get prompt template
        prompt = get_medical_prompt()
        
        # Create RAG chain using LCEL
        print("Building RAG chain...")
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | chat_model
            | StrOutputParser()
        )
        
        chatbot_initialized = True
        print("✓ Medical Chatbot initialized successfully!")
        
    except Exception as e:
        print(f"✗ Error initializing chatbot: {str(e)}")
        raise


def get_bot_response(user_message):
    """
    Get a response from the chatbot for a user message.
    
    Args:
        user_message: The user's question/message
        
    Returns:
        Bot's response as a string
    """
    global rag_chain, chatbot_initialized
    
    if not chatbot_initialized or rag_chain is None:
        return "Sorry, the chatbot is not initialized. Please refresh the page and try again."
    
    try:
        # Get response from RAG chain
        response = rag_chain.invoke(user_message)
        return response
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error getting bot response: {error_msg}")
        
        # Handle rate limit errors specifically
        if "429" in error_msg or "Resource exhausted" in error_msg:
            return "⚠️ API rate limit reached. Please wait a moment and try again. The free tier has usage limits."
        elif "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Please check your Google Cloud console or try again later."
        else:
            return "Sorry, I encountered an error processing your request. Please try again."


@app.route("/")
def index():
    """
    Render the main chat interface.
    """
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Handle chat messages from the user and return bot responses.
    Expected POST data: {"msg": "user message"}
    """
    try:
        user_message = request.form.get("msg", "").strip()
        
        if not user_message:
            return "Please enter a message."
        
        # Get bot response
        bot_response = get_bot_response(user_message)
        
        return bot_response
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return "Sorry, something went wrong. Please try again."


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint to verify the app is running.
    """
    status = {
        "status": "healthy" if chatbot_initialized else "initializing",
        "chatbot_ready": chatbot_initialized
    }
    return jsonify(status)


if __name__ == "__main__":
    # Initialize the chatbot before starting the server
    try:
        initialize_chatbot()
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")
        print("The app will start but the chatbot may not work properly.")
    
    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
