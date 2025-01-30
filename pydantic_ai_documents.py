import os
import sys
import json
import asyncio
import httpx
import logging
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio import Semaphore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
EMBEDDING_MODEL = 'mxbai-embed-large'
OLLAMA_MAX_CONCURRENT = 4
CHUNK_BATCH_SIZE = 5

# Excluded URLs - Add your URLs to exclude here
EXCLUDED_URLS = {
    # Add more URLs to exclude
}

# Initialize clients
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)


class OllamaClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
        self.semaphore = Semaphore(OLLAMA_MAX_CONCURRENT)

    async def close(self):
        await self.client.aclose()


ollama_client = OllamaClient()


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def should_process_url(url: str, excluded_urls: Set[str]) -> bool:
    """
    Check if a URL should be processed based on exclusion rules.

    Args:
        url: The URL to check
        excluded_urls: Set of URLs to exclude

    Returns:
        bool: True if URL should be processed, False if it should be excluded
    """
    # Normalize URLs for comparison (remove trailing slashes, etc.)
    normalized_url = url.rstrip('/')

    # Direct URL match
    if normalized_url in excluded_urls:
        logging.info(f"Excluding URL (direct match): {url}")
        return False

    # Check if URL is a subpath of any excluded URL
    for excluded_url in excluded_urls:
        if normalized_url.startswith(excluded_url.rstrip('/')):
            logging.info(f"Excluding URL (subpath match): {url}")
            return False

    return True


def filter_urls(urls: List[str], excluded_urls: Set[str]) -> List[str]:
    """
    Filter out excluded URLs from the list of URLs to process.

    Args:
        urls: List of URLs to filter
        excluded_urls: Set of URLs to exclude

    Returns:
        List[str]: Filtered list of URLs
    """
    filtered = [url for url in urls if should_process_url(url, excluded_urls)]
    excluded_count = len(urls) - len(filtered)
    if excluded_count > 0:
        logging.info(f"Excluded {excluded_count} URLs based on exclusion rules")
    return filtered


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def ollama_generate(prompt: str, system_prompt: str = None) -> str:
    """Generate text using Ollama with retries and error handling"""
    try:
        async with ollama_client.semaphore:
            response = await ollama_client.client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "format": "json"
                }
            )

            if response.status_code != 200:
                logging.error(f"Ollama API Error ({response.status_code}): {response.text}")
                return ""

            return response.json()["response"]

    except json.JSONDecodeError as e:
        logging.error(f"JSON Parse Error: {e}\nResponse: {response.text}")
    except httpx.ReadTimeout:
        logging.error("Timeout waiting for Ollama response")
    except Exception as e:
        logging.error(f"Unexpected error: {type(e).__name__} - {str(e)}")

    return ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embedding(text: str) -> List[float]:
    """Get embedding using Ollama's mxbai-embed-large with retries"""
    try:
        async with ollama_client.semaphore:
            response = await ollama_client.client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            return response.json()["embedding"]
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return [0] * 1024  # Return zero vector on error


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using local LLM"""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    Keep both title and summary concise but informative."""

    try:
        cleaned_chunk = chunk[:1000].replace('\n', ' ').strip()
        prompt_content = f"URL: {url}\n\nContent: {cleaned_chunk}"

        response = await ollama_generate(
            system_prompt=system_prompt,
            prompt=prompt_content
        )
        return json.loads(response)
    except Exception as e:
        logging.error(f"Error getting title/summary: {e}")
        return {"title": "Untitled", "summary": "No summary available"}


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks with context-aware boundaries"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        # Prefer breaking at code blocks
        last_code_block = chunk.rfind('```')
        if last_code_block > chunk_size * 0.3:
            end = start + last_code_block
        # Then paragraph breaks
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        # Then sentence boundaries
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunks.append(text[start:end].strip())
        start = end

    return chunks


async def process_and_store_document(url: str, markdown: str):
    """Process and store document chunks with batch processing"""
    chunks = chunk_text(markdown)

    # Process chunks in smaller batches
    for i in range(0, len(chunks), CHUNK_BATCH_SIZE):
        batch = chunks[i:i + CHUNK_BATCH_SIZE]
        tasks = [process_chunk(chunk, i + idx, url) for idx, chunk in enumerate(batch)]
        processed_chunks = await asyncio.gather(*tasks)

        # Store processed chunks
        insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
        await asyncio.gather(*insert_tasks)


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process individual chunk with validation"""
    # Add content validation
    if not chunk.strip():
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title="Empty Content",
            summary="No content available",
            content=chunk,
            metadata={
                "source": "pydantic_ai_docs",
                "chunk_size": 0,
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "url_path": urlparse(url).path
            },
            embedding=[0] * 1024
        )

    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert chunk into Supabase with error handling"""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("site_pages").insert(data).execute()
        logging.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        logging.error(f"Error inserting chunk: {e}")
        return None


async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    """Crawl URLs with reduced concurrency and improved error handling"""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                try:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    if result.success:
                        logging.info(f"Successfully crawled: {url}")
                        await process_and_store_document(url, result.markdown_v2.raw_markdown)
                    else:
                        logging.error(f"Failed: {url} - Error: {result.error_message}")
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap with error handling"""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

        return urls
    except Exception as e:
        logging.error(f"Error fetching sitemap: {e}")
        return []


async def warmup_ollama():
    """Warm up Ollama model before processing"""
    try:
        logging.info("Warming up Ollama model...")
        await ollama_generate("warmup")
        logging.info("Warmup complete")
    except Exception as e:
        logging.error(f"Warmup failed: {e}")


async def main():
    try:
        # Warm up the model
        await warmup_ollama()

        # Get URLs from Pydantic AI docs
        urls = get_pydantic_ai_docs_urls()
        if not urls:
            logging.error("No URLs found to crawl")
            return

        # Filter out excluded URLs
        filtered_urls = filter_urls(urls, EXCLUDED_URLS)

        if not filtered_urls:
            logging.error("All URLs were excluded by filters")
            return

        logging.info(f"Found {len(filtered_urls)} URLs to crawl after filtering")
        await crawl_parallel(filtered_urls)
    finally:
        # Clean up resources
        await ollama_client.close()


if __name__ == "__main__":
    asyncio.run(main())