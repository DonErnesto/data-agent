import hashlib
import json
import os
import pickle
import types
from pathlib import Path

from litellm import completion

from ..utils.logger import CustomLogger

logger = CustomLogger(console_level="INFO", file_level="DEBUG")


# Cache directory
CACHE_DIR = Path(".cache_responses")
CACHE_DIR.mkdir(exist_ok=True)
USE_CACHE = os.getenv("USE_CACHE", "false").lower() == "true"


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(x) for x in d]
    return d


def _hash_messages(messages):
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_response(cache_file, response):
    with open(cache_file, "wb") as f:
        pickle.dump(response, f)


def save_to_json(file, data):
    with file.open("w", encoding="utf-8") as f:
        # Convert response to dict before saving
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_response(cache_file):
    with open(cache_file, "rb") as f:
        return pickle.load(f)
    # Check cache


def cached_completion(messages, tools=None, temperature=0.1, max_tokens=1024, use_cache=None):
    """
    Wrapper around OpenAI's completion API with optional caching.
    """
    if use_cache is None:
        use_cache = USE_CACHE  # fallback to env var (given in make run)

    cache_key = _hash_messages(messages)
    cache_file = CACHE_DIR / f"{cache_key}.pickle"
    cache_json = CACHE_DIR / f"{cache_key}.json"
    cache_message_json = CACHE_DIR / f"{cache_key}_message.json"

    if use_cache:
        if cache_file.exists():
            logger.debug(f"---loading cached file: {cache_file} ...")
            response = load_response(cache_file)
            return response
        else:
            print(f"---can't find         : {cache_file}")

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
        logger.debug(f"---saving             : {cache_file}")
        save_response(cache_file, response)
        save_to_json(cache_json, data=response.model_dump())
        save_to_json(cache_message_json, data=messages)
    return response
