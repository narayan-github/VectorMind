from __future__ import annotations
from typing import Literal, TypedDict, List, Optional
import asyncio
import os
import httpx
import logging
from datetime import datetime
from dataclasses import dataclass

import streamlit as st
from supabase import Client
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration constants
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
EMBEDDING_MODEL = 'mxbai-embed-large'
OLLAMA_TIMEOUT = 120
STREAM_TIMEOUT = 300


@dataclass
class PydanticAIDeps:
    supabase: Client


class OllamaModel:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        self.client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9}
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class ChatAgent:
    def __init__(self):
        self.supabase = self.init_supabase()
        self.model = OllamaModel()
        self.system_prompt = """
        You are a Pydantic AI expert. Focus on providing accurate, concise answers about Pydantic AI.
        Use the documentation retrieval tools before answering questions.
        """

    def init_supabase(self) -> Optional[Client]:
        try:
            return Client(
                os.getenv("SUPABASE_URL", ""),
                os.getenv("SUPABASE_SERVICE_KEY", "")
            )
        except Exception as e:
            logger.error(f"Supabase init error: {e}")
            return None

    async def get_embedding(self, text: str) -> List[float]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text}
                )
                return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0] * 1024

    async def retrieve_docs(self, query: str) -> str:
        if not self.supabase:
            return "Database connection unavailable"

        try:
            embedding = await self.get_embedding(query)
            result = self.supabase.rpc(
                'match_site_pages',
                {'query_embedding': embedding, 'match_count': 5}
            ).execute()

            return "\n\n".join(
                f"# {doc['title']}\n{doc['content']}"
                for doc in result.data
            ) if result.data else "No relevant docs found"
        except Exception as e:
            logger.error(f"Doc retrieval error: {e}")
            return f"Error retrieving docs: {e}"

    async def generate_response(self, user_input: str) -> str:
        try:
            docs = await self.retrieve_docs(user_input)
            context = f"Documentation context:\n{docs}\n\nUser question: {user_input}"
            return await self.model.generate(context, self.system_prompt)
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "Error generating response"


# Streamlit UI Components
class ChatMessage(TypedDict):
    role: Literal['user', 'model']
    timestamp: str
    content: str


async def check_ollama_connection():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama connection error: {e}")
        return False


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = ChatAgent()


async def run_agent(user_input: str):
    if not user_input.strip():
        return

    if not await check_ollama_connection():
        st.error("Ollama connection failed")
        return

    with st.spinner("Generating response..."):
        try:
            response = await st.session_state.agent.generate_response(user_input)
            st.session_state.messages.append({
                "role": "model",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Agent error: {str(e)}")


def main():
    st.title("Pydantic AI Assistant")
    init_session()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Pydantic AI:"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            asyncio.run(run_agent(prompt))
            st.rerun()


if __name__ == "__main__":
    main()