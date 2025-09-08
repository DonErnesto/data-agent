import hashlib
import json
import os
from pathlib import Path

from litellm import completion

USE_CACHE = os.getenv("USE_CACHE", "false").lower() == "true"

# Initialize client once

# Cache directory
CACHE_DIR = Path(".cache_responses")
CACHE_DIR.mkdir(exist_ok=True)


def _hash_messages(messages: list) -> str:
    """Create a stable hash from messages for cache lookup."""
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cached_completion(messages, tools=None, temperature=0.1, max_tokens=1024, use_cache=True):
    """
    Wrapper around OpenAI's completion API with optional caching.
    """
    if use_cache is None:
        use_cache = USE_CACHE  # fallback to env var
    cache_key = _hash_messages(messages)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    # Check cache
    if use_cache and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    # API request
    response = completion(
        model="gpt-4-turbo-2024-04-09",
        messages=messages,
        temperature=temperature,
        tools=tools,
        max_tokens=max_tokens,
    )

    # Save response to cache
    if use_cache:
        with cache_file.open("w", encoding="utf-8") as f:
            # Convert response to dict before saving
            json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)

    return response
