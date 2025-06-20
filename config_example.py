# SIGHT Evaluation Configuration Example
# Rename to config.py and update with your values

# OpenAI API Configuration
OPENAI_API_KEY = "your-api-key-here"  # Replace with your OpenAI API key

# Model Configuration
MODEL_ID = "gpt-3.5-turbo-0125"  # Base model ID

# Data Configuration - Using SIGHT dataset
DATA_PATH = "data/sample_dataset.csv"  # Path to SIGHT dataset file

# Output Configuration
OUTPUT_DIR = "results"  # Directory for saving results

# Rate Limiting
SLEEP_TIME = 1  # Seconds to wait between API calls

# SIGHT Categories (9 categories from the paper)
SIGHT_CATEGORIES = [
    "general",
    "confusion", 
    "pedagogy",
    "setup",
    "personal_experience",
    "clarification",
    "gratitude",
    "non_english",
    "na"
]