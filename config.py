import os

# File paths
BASE_DIR = 'data/'
TRAIN_FILE = os.path.join(BASE_DIR, 'train/train.csv')
TEST_FILE = os.path.join(BASE_DIR, 'test/test.csv')
TRAIN_OUTPUT = os.path.join(BASE_DIR, 'train/jsonl/claude_train_v2.jsonl')
TEST_OUTPUT = os.path.join(BASE_DIR, 'test/jsonl/claude_test_v2.jsonl')
SUBMISSION_FILENAME =  os.path.join(BASE_DIR,"submissions/submission_baseline.csv")
MODEL_RESPONSES = os.path.join(BASE_DIR, 'test/model_responses_baseline.json')

# Data preparation parameters
AGE_BINS = [0, 12, 18, 65, float('inf')]
AGE_LABELS = ['Child', 'Teenager', 'Adult', 'Elderly']
FARE_BINS = 4
FARE_LABELS = ['Low', 'Medium-Low', 'Medium-High', 'High']


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