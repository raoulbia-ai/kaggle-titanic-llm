import pandas as pd
import json

# Load the CSV file locally
train_data = pd.read_csv('data/train.csv')

# Function to infer missing age based on class and gender
def infer_age(row, data):
    if pd.isnull(row['Age']):
        median_age = data[(data['Pclass'] == row['Pclass']) & (data['Sex'] == row['Sex'])]['Age'].median()
        return median_age if not pd.isnull(median_age) else "unknown age"
    return row['Age']

# Function to infer missing fare based on class and embarkation port
def infer_fare(row, data):
    if pd.isnull(row['Fare']):
        median_fare = data[(data['Pclass'] == row['Pclass']) & (data['Embarked'] == row['Embarked'])]['Fare'].median()
        return median_fare if not pd.isnull(median_fare) else "unknown fare"
    return row['Fare']

# Function to categorize fare
def categorize_fare(fare):
    if fare == "unknown fare":
        return "an unknown fare"
    elif fare < 10:
        return "a low fare"
    elif fare < 50:
        return "a moderate fare"
    else:
        return "a high fare"

# Function to extract a title or role based on age
def extract_title_or_role(row):
    if row['Age'] == "unknown age":
        if row['Sex'] == "male":
            return "a man of unknown age"
        else:
            return "a woman of unknown age"
    elif row['Age'] < 18:
        return "a child"
    elif row['Age'] > 60:
        return "an elderly passenger"
    elif row['Sex'] == "male":
        return "a man"
    else:
        return "a woman"

# Function to map embarkation codes to full names
def map_embarkation(embarked):
    embarkation_mapping = {
        "C": "Cherbourg",
        "Q": "Queenstown",
        "S": "Southampton"
    }
    return embarkation_mapping.get(embarked, "unknown port")

# Function to generate JSONL entries in chat format
def generate_jsonl(data):
    jsonl_data = []
    for _, row in data.iterrows():
        # Infer missing values
        row['Age'] = infer_age(row, data)
        row['Fare'] = infer_fare(row, data)
        
        # Extracting useful features
        role = extract_title_or_role(row)
        family_size = row['SibSp'] + row['Parch']
        fare_category = categorize_fare(row['Fare'])
        embarkation_port = map_embarkation(row['Embarked'])
        
        # Convert age to integer if available
        age = int(row['Age']) if row['Age'] != "unknown age" else "unknown age"
        
        # Determine the appropriate pronoun based on the role
        if "man" in role:
            pronoun = "He"
        elif "woman" in role:
            pronoun = "She"
        else:
            pronoun = "The child"

        # Constructing the refined prompt
        if family_size == 0:
            family_description = "alone, with no family aboard"
        else:
            family_description = f"with {family_size} family members aboard"

        user_message = (
            f"This is {role} aged {age} traveling in {ordinal(row['Pclass'])} class "
            f"{family_description}. {pronoun} boarded in {embarkation_port} with {fare_category}."
        )
        
        assistant_message = "survived" if row['Survived'] == 1 else "did not survive"
        
        chat_format = {
            "messages": [
                {"role": "system", "content": "You are an assistant that predicts the likelihood of surviving the Titanic sinking based on passenger details."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        jsonl_data.append(chat_format)
    
    return jsonl_data

def ordinal(n):
    """Convert an integer into its ordinal representation."""
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

# Generate the JSONL data
jsonl_data = generate_jsonl(train_data)

# Save to JSONL file
with open('data/train.jsonl', 'w') as jsonl_file:
    for entry in jsonl_data:
        jsonl_file.write(json.dumps(entry) + '\n')

print("Enhanced JSONL file saved to data/train.jsonl")
