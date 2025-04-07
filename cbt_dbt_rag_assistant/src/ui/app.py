# src/ui/app.py
"""
Simple Streamlit UI for the RAG Assistant.
"""

import logging
import streamlit as st
from pathlib import Path
import sys
import time

# --- Add project root to sys.path ---
# Necessary for Streamlit to find the 'src' module when run from the project root
project_root = Path(__file__).resolve().parent.parent.parent # Go up 3 levels from src/ui/app.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End sys.path modification ---

# Import components after path modification
try:
    from src.core.config import settings
    from src.services.query_service import QueryService
    from src.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
    from src.infrastructure.vector_stores.postgres_vector_store import PostgresVectorStore
    from src.infrastructure.llm_providers.ollama_provider import OllamaProvider
except ImportError as e:
    st.error(f"Failed to import necessary modules. Ensure the project structure is correct and dependencies are installed. Error: {e}")
    st.stop() # Stop execution if imports fail

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Component Initialization (Cached) ---
# Use Streamlit's caching to avoid re-initializing components on every interaction.
@st.cache_resource
def get_query_service():
    """Initializes and returns the QueryService."""
    logger.info("Initializing components for Streamlit UI...")
    try:
        embedding_model = SentenceTransformerEmbedder(
            model_name=settings.embedding_model_name
        )
        vector_store = PostgresVectorStore(
             db_url=str(settings.database_url),
             embedding_dim=embedding_model.dimension
        )
        llm = OllamaProvider(
            host=str(settings.ollama_api_base),
            default_model=settings.default_llm_model
        )
        query_service = QueryService(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm=llm,
            top_k_retrieval=5 # Default top_k for UI
        )
        logger.info("QueryService initialized successfully for Streamlit UI.")
        return query_service
    except Exception as e:
        logger.exception("Failed to initialize QueryService for Streamlit UI.")
        st.error(f"Failed to initialize backend components: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="CBT/DBT RAG Assistant", layout="wide")
st.title("🧠 CBT/DBT RAG Assistant")
st.caption("Ask questions about the ingested CBT/DBT documents.")

# Initialize QueryService
query_service = get_query_service()

if query_service:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                start_time = time.time()
                try:
                    # Get response from QueryService
                    # Note: Currently not passing chat history to the backend LLM
                    #       This could be added by formatting st.session_state.messages
                    #       appropriately for the OllamaProvider.
                    response = query_service.query(user_query=prompt)
                    duration = time.time() - start_time
                    logger.info(f"Query processed in {duration:.2f} seconds.")
                    full_response = response
                except Exception as e:
                    logger.error(f"Error during query processing: {e}")
                    full_response = f"Sorry, an error occurred: {e}"
                    st.error(full_response)

            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Query Service could not be initialized. Please check logs and ensure backend services (DB, Ollama) are running.")
