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
The Titanic sank on its maiden voyage in 1912 after hitting an iceberg. The ship had a severe shortage of lifeboats, 
with capacity for only about half of those onboard. First-class passengers, especially women and children, had a higher 
survival rate due to their proximity to the boat deck and preferential treatment during evacuation. The 'women and children first' 
protocol was generally followed, particularly in first and second class areas.
"""

# System message for AI
SYSTEM_MESSAGE = """
You are an AI assistant trained to analyze Titanic passenger data and predict survival outcomes. 
Consider multiple factors, weigh their importance, and think step-by-step before making a prediction. 
Provide a clear prediction along with your confidence level based on the given information.
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