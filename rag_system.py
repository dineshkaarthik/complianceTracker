import os
import time
import json
import random
import sqlite3
from functools import lru_cache
from ratelimit import limits, sleep_and_retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from config import (
    OPENAI_API_KEY, VECTOR_STORE_PATH, RATE_LIMIT, RATE_LIMIT_PERIOD,
    INITIAL_BACKOFF, MAX_BACKOFF, BACKOFF_FACTOR, logger,
    DISK_CACHE_DIR, DISK_CACHE_EXPIRATION, SQLITE_DB_PATH, API_USAGE_LOG_PATH
)
import diskcache
import openai
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
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

# Load NLI model and tokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

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

@torch.no_grad()
def batch_nli_score(premises, hypotheses, batch_size=8):
    scores = []
    for i in range(0, len(premises), batch_size):
        batch_premises = premises[i:i+batch_size]
        batch_hypotheses = hypotheses[i:i+batch_size]
        inputs = nli_tokenizer(batch_premises, batch_hypotheses, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = nli_model(**inputs)
        batch_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 2].tolist()  # Entailment scores
        scores.extend(batch_scores)
    return scores

@lru_cache(maxsize=1000)
def cached_nli_score(premise, hypothesis):
    return batch_nli_score([premise], [hypothesis])[0]

def integrated_gradients(premise, hypothesis):
    ig = IntegratedGradients(nli_model)
    inputs = nli_tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    
    attributions, delta = ig.attribute(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       return_convergence_delta=True,
                                       internal_batch_size=1,
                                       n_steps=50)
    
    return attributions.sum(dim=-1).squeeze(0)

def validate_and_refine_checklist(original_text, checklist):
    refined_checklist = []
    premises = [original_text] * len(checklist)
    hypotheses = checklist

    try:
        scores = batch_nli_score(premises, hypotheses)
    except Exception as e:
        logger.error(f"Error in batch NLI scoring: {str(e)}")
        scores = [cached_nli_score(original_text, item) for item in checklist]

    for item, score in zip(checklist, scores):
        if score > 0.7:  # Threshold for acceptance
            refined_checklist.append((item, score))
        else:
            try:
                attributions = integrated_gradients(original_text, item)
                important_tokens = nli_tokenizer.convert_ids_to_tokens(nli_tokenizer.encode(item))
                important_tokens = [token for token, attr in zip(important_tokens, attributions) if attr > 0]
                refined_item = ' '.join(important_tokens)
                refined_score = cached_nli_score(original_text, refined_item)
                refined_checklist.append((refined_item, refined_score))
            except Exception as e:
                logger.error(f"Error in integrated gradients for item '{item}': {str(e)}")
                refined_checklist.append((item, score))  # Keep original item if refinement fails
    
    return refined_checklist

def process_texts_with_rag(texts: List[Tuple[str, int]]) -> List[List[Tuple[str, float]]]:
    logger.info("Starting RAG processing")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Create or load FAISS index
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    else:
        vector_store = FAISS.from_texts([chunk for chunk, _ in texts], embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)

    logger.info("Created or loaded vector store")

    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )

    results = []
    for i, (text, chunk_id) in enumerate(texts):
        query = f"Summarize the key compliance requirements and generate a detailed checklist based on the following document chunk (ID: {chunk_id}):\n\n{text}"
        logger.info(f"Processing text chunk {i+1}/{len(texts)}")
        start_time = time.time()
        result = cached_api_call(qa_chain.run, query)
        end_time = time.time()
        logger.info(f"Processed text chunk {i+1} in {end_time - start_time:.2f} seconds")
        
        # Validate and refine the checklist
        checklist_items = result.split('\n')
        refined_checklist = validate_and_refine_checklist(text, checklist_items)
        results.append(refined_checklist)

    logger.info("Completed RAG processing")
    return results
