import os
import shutil
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask configuration
DEBUG = True
SECRET_KEY = os.urandom(24)
UPLOAD_FOLDER = '/tmp'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Vector store configuration
VECTOR_STORE_PATH = 'faiss_index'

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT = 1  # requests per minute
RATE_LIMIT_PERIOD = 600  # seconds (10 minutes)

# Exponential backoff configuration
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 1800  # seconds (30 minutes)
BACKOFF_FACTOR = 2

# Queue configuration
MAX_QUEUE_SIZE = 10
DOCUMENT_PROCESSING_DELAY = 30  # seconds

# Clear cache on startup
if os.path.exists(VECTOR_STORE_PATH):
    shutil.rmtree(VECTOR_STORE_PATH)
    logger.info(f"Cleared cache: {VECTOR_STORE_PATH}")

# Disk-based cache configuration
DISK_CACHE_DIR = 'api_response_cache'
DISK_CACHE_EXPIRATION = 3600  # 1 hour

# SQLite database configuration for persistent cache
SQLITE_DB_PATH = 'persistent_cache.db'

# API usage tracking
API_USAGE_LOG_PATH = 'api_usage.log'
