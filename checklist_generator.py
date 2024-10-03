from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import (
    OPENAI_API_KEY, RATE_LIMIT, RATE_LIMIT_PERIOD,
    INITIAL_BACKOFF, MAX_BACKOFF, BACKOFF_FACTOR, logger,
    DISK_CACHE_DIR, DISK_CACHE_EXPIRATION, SQLITE_DB_PATH, API_USAGE_LOG_PATH
)
from functools import lru_cache
from ratelimit import limits, sleep_and_retry
import os
import time
import random
import json
import diskcache
import openai
import sqlite3
from rag_system import batch_nli_score, cached_nli_score, integrated_gradients, nli_tokenizer
from typing import List, Tuple

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize disk cache
disk_cache = diskcache.Cache(DISK_CACHE_DIR)

# Initialize SQLite database for persistent cache
def init_sqlite_cache():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, value TEXT, expiry INTEGER)''')
    conn.commit()
    conn.close()

init_sqlite_cache()

def exponential_backoff(attempt):
    backoff = min(INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt), MAX_BACKOFF)
    jitter = random.uniform(0, backoff * 0.1)
    return backoff + jitter

@sleep_and_retry
@limits(calls=RATE_LIMIT, period=RATE_LIMIT_PERIOD)
def rate_limited_api_call(func, *args, **kwargs):
    attempt = 0
    while True:
        try:
            result = func(*args, **kwargs)
            log_api_usage(func.__name__)
            return result
        except openai.RateLimitError as e:
            attempt += 1
            backoff = exponential_backoff(attempt)
            logger.warning(f"API rate limit exceeded. Retrying in {backoff:.2f} seconds. Error: {str(e)}")
            time.sleep(backoff)
        except Exception as e:
            logger.error(f"API call failed. Error: {str(e)}")
            raise

def log_api_usage(api_name):
    with open(API_USAGE_LOG_PATH, 'a') as f:
        f.write(f"{time.time()},{api_name}\n")

@lru_cache(maxsize=500)
def cached_api_call(func, *args, **kwargs):
    cache_key = f"{func.__name__}:{args}:{kwargs}"
    
    # Check SQLite cache first
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value, expiry FROM cache WHERE key=?", (cache_key,))
    row = c.fetchone()
    if row:
        value, expiry = row
        if expiry > time.time():
            conn.close()
            return json.loads(value)
    
    # If not in SQLite, check disk cache
    cached_result = disk_cache.get(cache_key)
    if cached_result is not None:
        return json.loads(cached_result)
    
    # If not in any cache, make the API call
    result = rate_limited_api_call(func, *args, **kwargs)
    
    # Store in both caches
    json_result = json.dumps(result)
    expiry = int(time.time() + DISK_CACHE_EXPIRATION)
    c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (cache_key, json_result, expiry))
    conn.commit()
    conn.close()
    disk_cache.set(cache_key, json_result, expire=DISK_CACHE_EXPIRATION)
    
    return result

def generate_checklist(processed_text: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    logger.info("Starting checklist generation")
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert compliance officer tasked with creating a detailed and accurate checklist based on the following processed compliance document summaries. "
        "Each checklist item should be clear, actionable, and directly related to compliance requirements. "
        "Avoid any vague or ambiguous statements. If there's uncertainty about a specific requirement, "
        "indicate that further clarification may be needed. "
        "Format each checklist item as a concise statement followed by any necessary context or explanation. "
        "Here are the processed text chunks with their relevance scores:\n\n"
        "{text_chunks}\n\n"
        "Generate a comprehensive checklist based on these summaries:"
    )
    
    # Prepare text chunks for the prompt
    text_chunks = "\n".join([f"Chunk {i+1} (Relevance: {sum([score for _, score in chunk])/len(chunk):.2f}):\n" + 
                             "\n".join([f"- {item} (Score: {score:.2f})" for item, score in chunk])
                             for i, chunk in enumerate(processed_text)])
    
    messages = prompt.format_messages(text_chunks=text_chunks)
    
    start_time = time.time()
    try:
        response = cached_api_call(llm, messages)
        end_time = time.time()
        
        logger.info(f"Generated checklist in {end_time - start_time:.2f} seconds")
        logger.debug(f"Generated checklist content: {response.content}")
        
        # Validate and refine the checklist using NLI and Integrated Gradients
        checklist_items = response.content.split('\n')
        refined_checklist = []
        
        full_text = "\n".join([item for chunk in processed_text for item, _ in chunk])
        premises = [full_text] * len(checklist_items)
        hypotheses = checklist_items
        
        try:
            scores = batch_nli_score(premises, hypotheses)
        except Exception as e:
            logger.error(f"Error in batch NLI scoring: {str(e)}")
            scores = [cached_nli_score(full_text, item) for item in checklist_items]
        
        for item, score in zip(checklist_items, scores):
            if score > 0.7:  # Threshold for acceptance
                refined_checklist.append((item, score))
            else:
                try:
                    attributions = integrated_gradients(full_text, item)
                    important_tokens = nli_tokenizer.convert_ids_to_tokens(nli_tokenizer.encode(item))
                    important_tokens = [token for token, attr in zip(important_tokens, attributions) if attr > 0]
                    refined_item = ' '.join(important_tokens)
                    refined_score = cached_nli_score(full_text, refined_item)
                    refined_checklist.append((refined_item, refined_score))
                except Exception as e:
                    logger.error(f"Error in integrated gradients for item '{item}': {str(e)}")
                    refined_checklist.append((item, score))  # Keep original item if refinement fails
        
        return refined_checklist
    except Exception as e:
        logger.error(f"Error generating checklist: {str(e)}")
        return None
