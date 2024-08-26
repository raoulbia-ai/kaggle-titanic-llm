import pandas as pd
import json

# Load the test data
test_data = pd.read_csv('data/test.csv')

# Function to categorize fare
def categorize_fare(fare):
    if pd.isnull(fare):
        return "an unknown fare"
    elif fare < 10:
        return "a low fare"
    elif fare < 50:
        return "a moderate fare"
    else:
        return "a high fare"

# Function to map embarkation codes to full names
def map_embarkation(embarked):
    embarkation_mapping = {
        "C": "Cherbourg",
        "Q": "Queenstown",
        "S": "Southampton"
    }
    return embarkation_mapping.get(embarked, "unknown port")

# Generate user messages from test data and include PassengerId (separately)
def generate_test_prompts(data):
    jsonl_data = []
    for _, row in data.iterrows():
        family_size = row['SibSp'] + row['Parch']
        embarkation_port = map_embarkation(row['Embarked'])
        pronoun = "They" if pd.isnull(row['Sex']) else "He" if row['Sex'] == "male" else "She"
        fare_category = categorize_fare(row['Fare'])
        
        if family_size == 0:
            family_description = "alone, with no family aboard"
        else:
            family_description = f"with {family_size} family members aboard"
        
        user_message = (
            f"This is a {int(row['Age']) if not pd.isnull(row['Age']) else 'person of unknown age'} "
            f"{row['Sex']} traveling in {ordinal(row['Pclass'])} class {family_description}. "
            f"{pronoun} boarded in {embarkation_port} with {fare_category}."
        )
        
        # Store the PassengerId separately along with the message
        entry = {
            "PassengerId": row["PassengerId"],
            "messages": [
                {"role": "system", "content": "You are an assistant that predicts the likelihood of surviving the Titanic sinking based on passenger details."},
                {"role": "user", "content": user_message}
            ]
        }
        jsonl_data.append(entry)
    
    return jsonl_data

def ordinal(n):
    """Convert an integer into its ordinal representation."""
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

# Generate the JSONL data
jsonl_data = generate_test_prompts(test_data)

# Save to JSONL file
with open('data/test.jsonl', 'w') as jsonl_file:
    for entry in jsonl_data:
        jsonl_file.write(json.dumps(entry) + '\n')

print("Formatted JSONL file saved as data/test.jsonl")
