import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import json

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Feature Engineering
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Extract titles from names
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer",
        "Mlle": "Miss", "Mme": "Mrs", "Don": "Royalty", "Lady": "Royalty",
        "Countess": "Royalty", "Jonkheer": "Royalty", "Sir": "Royalty",
        "Capt": "Officer", "Ms": "Mrs"
    }
    data['Title'] = data['Title'].map(title_mapping).fillna('Other')
    
    # Extract cabin letter and fill missing with 'U' for Unknown
    data['Cabin'] = data['Cabin'].fillna('U')
    data['CabinLetter'] = data['Cabin'].str[0]
    
    # Handle missing values in Age and Fare
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # Create age groups
    age_bins = [0, 12, 18, 35, 60, np.inf]
    age_labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, include_lowest=True).astype(str)
    
    # Create fare categories
    fare_bins = [0, 7.91, 14.454, 31, np.inf]
    fare_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    data['FareCategory'] = pd.cut(data['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True).astype(str)
    
    # Sophisticated imputation for numeric features
    numeric_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'FamilySize']
    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=5)
    
    scaled_data = scaler.fit_transform(data[numeric_features])
    imputed_data = imputer.fit_transform(scaled_data)
    
    data[numeric_features] = scaler.inverse_transform(imputed_data)
    
    return data

def generate_prompt(row):
    age = int(round(row['Age']))
    title = str(row['Title'])
    pclass = int(row['Pclass'])
    gender = "male" if row['Sex'] == "male" else "female"
    family_size = int(row['FamilySize'])
    is_alone = bool(row['IsAlone'])
    fare_category = str(row['FareCategory'])
    embarkation_port = map_embarkation(row['Embarked'])
    cabin_letter = str(row['CabinLetter'])
    age_group = str(row['AgeGroup'])

    prompt = f"Passenger details for Titanic voyage:\n"
    prompt += f"- A {age}-year-old {gender} {title} ({age_group})\n"
    prompt += f"- Traveling in {ordinal(pclass)} class\n"
    
    if is_alone:
        prompt += "- Traveling alone\n"
    else:
        prompt += f"- Accompanied by {family_size - 1} family members\n"
    
    prompt += f"- Boarded at {embarkation_port}\n"
    prompt += f"- Paid a {fare_category.lower()} fare\n"
    
    if cabin_letter != 'U':
        prompt += f"- Assigned to a cabin in section {cabin_letter}\n"
    
    prompt += "\nBased on these factors and your knowledge of the Titanic disaster, "
    prompt += "determine whether this passenger survived or did not survive. "
    prompt += "Provide your reasoning, then conclude with a clear statement of 'Survived' or 'Did not survive' on a new line."
    prompt += "\n\nReasoning:"

    return prompt

def map_embarkation(embarked):
    embarkation_mapping = {
        "C": "Cherbourg, France",
        "Q": "Queenstown, Ireland",
        "S": "Southampton, England"
    }
    return embarkation_mapping.get(str(embarked), "an unknown port")

def ordinal(n):
    try:
        n = int(n)
        return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    except (ValueError, TypeError):
        return str(n)  # Return the original value if conversion fails

def generate_jsonl(data):
    jsonl_data = []
    for _, row in data.iterrows():
        user_message = generate_prompt(row)
        
        system_message = (
            "You are an assistant that predicts Titanic passenger survival. "
            "Analyze the given information and provide your reasoning. "
            "Conclude with a clear 'Survived' or 'Did not survive' statement on a new line."
        )
        
        assistant_message = "Based on the provided information, I believe this passenger "
        assistant_message += "survived" if row['Survived'] == 1 else "did not survive"
        assistant_message += " the Titanic disaster. Here's my reasoning:\n\n"
        
        # Add reasoning based on known factors
        if row['Sex'] == 'female' or row['Age'] < 18:
            assistant_message += "- Women and children were given priority in lifeboats, increasing their chances of survival.\n"
        if row['Pclass'] == 1:
            assistant_message += "- First-class passengers had better access to lifeboats and information about the ship's condition.\n"
        if row['CabinLetter'] in ['B', 'C', 'D', 'E']:
            assistant_message += "- Passengers in upper deck cabins (B, C, D, E) were closer to lifeboats and had a higher chance of escaping.\n"
        if row['Embarked'] == 'C':
            assistant_message += "- Passengers who boarded at Cherbourg were often wealthier and had better chances of survival.\n"
        if row['FamilySize'] > 1 and row['FamilySize'] < 5:
            assistant_message += "- Small to medium-sized family groups often stayed together, potentially increasing their chances of getting into a lifeboat.\n"
        if row['FareCategory'] in ['Medium-High', 'High']:
            assistant_message += "- Passengers who paid higher fares typically had better accommodations and possibly better access to lifeboats.\n"
        
        assistant_message += f"\n{'Survived' if row['Survived'] == 1 else 'Did not survive'}"
        
        chat_format = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        jsonl_data.append(chat_format)
    
    return jsonl_data

# Main execution
if __name__ == "__main__":
    train_data = load_and_preprocess_data('data/train.csv')
    jsonl_data = generate_jsonl(train_data)

    with open('data/train_refined.jsonl', 'w') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + '\n')

    print("Enhanced JSONL file saved to data/train_refined.jsonl")