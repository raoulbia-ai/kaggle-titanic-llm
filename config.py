import os

# File paths
BASE_DIR = 'data/'
TRAIN_FILE = os.path.join(BASE_DIR, 'train/train.csv')
TEST_FILE = os.path.join(BASE_DIR, 'test/test.csv')
TRAIN_OUTPUT = os.path.join(BASE_DIR, 'train/jsonl/claude_train_v2.jsonl')
TEST_OUTPUT = os.path.join(BASE_DIR, 'test/jsonl/claude_test_v2.jsonl')
SUBMISSION_FILENAME =  os.path.join(BASE_DIR,"submissions/submission_claude_sonnet_v2.csv")

# Data preparation parameters
AGE_BINS = [0, 12, 18, 65, float('inf')]
AGE_LABELS = ['Child', 'Teenager', 'Adult', 'Elderly']
FARE_BINS = 4
FARE_LABELS = ['Low', 'Medium-Low', 'Medium-High', 'High']

# Prompt generation
HISTORICAL_CONTEXT = """
Inconsistent Protocols: The "women and children first" protocol was applied inconsistently. First Officer William Murdoch 
interpreted it to mean women and children first, while Second Officer Charles Lightoller adhered strictly to that, 
leading to disparities in survival rates.
Panic and Delays: Panic erupted among passengers, especially in third class, where many were unaware of the severity of 
the situation and struggled to reach the lifeboats due to flooding and confusion.
Musicians' Role: none survived."""

# System message for AI
SYSTEM_MESSAGE = """
You are an AI assistant trained to analyze Titanic passenger data and predict survival outcomes. 
Provide clear and concise predictions based on the given information.
"""

# Feature importance (for analytical breakdown)
FEATURE_IMPORTANCE = {
    'Sex': 'Critical',
    'Pclass': 'Very High',
    'Age': 'High',
    'Fare': 'Moderate',
    'Embarked': 'Low',
    'FamilySize': 'Moderate',
    'Title': 'High',
    'Deck': 'Moderate'
}