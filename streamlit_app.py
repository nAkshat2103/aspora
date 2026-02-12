"""Streamlit UI for the AI Document Q&A (RAG) system."""

import os
import tempfile
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline.chunker import Chunker
from pipeline.knowledge_base import KnowledgeBase
from pipeline.llm_service import LLMService
from pipeline.rag_pipeline import RAGPipeline
from pipeline.registry import DocumentRegistry
from pipeline.vector_store import VectorStoreManager

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DB_PATH = str(DATA_DIR / "documents.db")


def get_pipeline():
    """Initialize and cache the RAG pipeline (loaded once per session)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not set. Add it to your .env file.")
        st.stop()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMService(api_key=api_key)
    vector_store = VectorStoreManager(persist_directory=str(VECTOR_DB_DIR))
    registry = DocumentRegistry(db_path=DB_PATH)
    chunker = Chunker()
    kb = KnowledgeBase(registry=registry, vector_store=vector_store, chunker=chunker)
    return RAGPipeline(knowledge_base=kb, llm_service=llm)


# def main():
#     st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
#     st.title("📄 AI Document Q&A")

#     pipeline = get_pipeline()
#     docs = pipeline.knowledge_base.list_documents()

#     # Sidebar: Upload & manage documents
#     with st.sidebar:
#         st.header("Documents")

#         # File upload
#         uploaded_files = st.file_uploader(
#             "Upload PDF, TXT, or MD",
#             type=["pdf", "txt", "md"],
#             accept_multiple_files=True,
#         )

#         # URL input (external knowledge base)
#         st.caption("Or add a URL as external knowledge")
#         url_input = st.text_input(
#             "Web page URL",
#             placeholder="https://example.com/article",
#             label_visibility="collapsed",
#         )
#         urls = [u.strip() for u in url_input.split() if u.strip().startswith(("http://", "https://"))]

#         if (uploaded_files or urls) and st.button("Ingest", type="primary"):
#             items_to_ingest: list[str] = []
#             temp_paths: list[str] = []
#             try:
#                 for f in uploaded_files or []:
#                     suffix = Path(f.name).suffix
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                         tmp.write(f.read())
#                         temp_paths.append(tmp.name)
#                         items_to_ingest.append(tmp.name)

                
#                 items_to_ingest.extend(urls)
#                 doc_ids = pipeline.ingest(items_to_ingest)
#                 for path in temp_paths:
#                     try:
#                         os.unlink(path)
#                     except OSError:
#                         pass
#                 st.success(f"Ingested {len(doc_ids)} document(s)")
#                 st.rerun()
#             except ValueError as e:
#                 st.error(str(e))
#             except Exception as e:
#                 st.error(f"Ingestion failed: {e}")
#                 with st.expander("Show full error details"):
#                     st.code(traceback.format_exc())

#         st.divider()
#         st.subheader("Your documents")
#         if not docs:
#             st.caption("No documents yet. Upload files above.")
#         else:
#             if st.button("🗑️ Clear all & re-ingest", type="secondary", help="Delete all documents. Re-upload to use improved chunking."):
#                 for d in docs:
#                     pipeline.knowledge_base.delete_document(d["id"])
#                 st.success("Cleared. Re-upload files to ingest.")
#                 st.rerun()
#             for d in docs:
#                 label = d["file_name"]
#                 path = d.get("file_path", "")
#                 with st.expander(label, expanded=False):
#                     if path.startswith(("http://", "https://")):
#                         st.caption(f"[🔗 Open]({path})")
#                     st.caption(f"ID: `{d['id']}`")
#                     if st.button("Delete", key=d["id"], type="secondary"):
#                         pipeline.knowledge_base.delete_document(d["id"])
#                         st.rerun()

#     # Main: Chat
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     if prompt := st.chat_input("Ask a question about your documents..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         if not docs:
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": "⚠️ Upload and ingest documents first.",
#             })
#             st.rerun()

#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             try:
#                 history = st.session_state.messages[:-1]
#                 full_response = st.write_stream(
#                     pipeline.ask_stream(prompt, chat_history=history)
#                 )
#             except Exception as e:
#                 full_response = f"Error: {e}"
#                 st.error(full_response)

#         st.session_state.messages.append({"role": "assistant", "content": full_response})


# if __name__ == "__main__":
#     main()


def main():
    st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
    st.title("📄 AI Document Q&A")

    pipeline = get_pipeline()
    docs = pipeline.knowledge_base.list_documents()

    # Sidebar: Upload & manage documents
    with st.sidebar:
        st.header("Documents")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        # URL input (external knowledge base)
        st.caption("Or add a URL as external knowledge")
        url_input = st.text_input(
            "Web page URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
        )
        urls = [u.strip() for u in url_input.split() if u.strip().startswith(("http://", "https://"))]

        if (uploaded_files or urls) and st.button("Ingest", type="primary"):
            items_to_ingest: list[str] = []
            try:
                # ✅ FIX: Save using original filename instead of temp name
                for f in uploaded_files or []:
                    save_path = DATA_DIR / f.name
                    with open(save_path, "wb") as out:
                        out.write(f.read())
                    items_to_ingest.append(str(save_path))

                items_to_ingest.extend(urls)

                doc_ids = pipeline.ingest(items_to_ingest)

                st.success(f"Ingested {len(doc_ids)} document(s)")
                st.rerun()

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                with st.expander("Show full error details"):
                    st.code(traceback.format_exc())

        st.divider()
        st.subheader("Your documents")
        if not docs:
            st.caption("No documents yet. Upload files above.")
        else:
            if st.button(
                "🗑️ Clear all & re-ingest",
                type="secondary",
                help="Delete all documents. Re-upload to use improved chunking.",
            ):
                for d in docs:
                    pipeline.knowledge_base.delete_document(d["id"])
                st.success("Cleared. Re-upload files to ingest.")
                st.rerun()

            for d in docs:
                label = d["file_name"]
                path = d.get("file_path", "")
                with st.expander(label, expanded=False):
                    if path.startswith(("http://", "https://")):
                        st.caption(f"[🔗 Open]({path})")
                    st.caption(f"ID: `{d['id']}`")
                    if st.button("Delete", key=d["id"], type="secondary"):
                        pipeline.knowledge_base.delete_document(d["id"])
                        st.rerun()

    # Main: Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not docs:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Upload and ingest documents first.",
            })
            st.rerun()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                history = st.session_state.messages[:-1]
                full_response = st.write_stream(
                    pipeline.ask_stream(prompt, chat_history=history)
                )
            except Exception as e:
                full_response = f"Error: {e}"
                st.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
