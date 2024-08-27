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

    # Handle missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Create Family Identifier first
    data['LastName'] = data['Name'].apply(lambda x: x.split(",")[0].strip())
    data['FamilyIdentifier'] = data['LastName'] + "_" + data['Ticket'] + "_" + data['Embarked']

    # Refined Economic Tier
    data['FarePerPerson'] = data['Fare'] / data['FamilySize']
    data['EconomicScore'] = (4 - data['Pclass']) * 2 + (data['FarePerPerson'] / data['FarePerPerson'].max()) * 10
    data['EconomicTier'] = pd.qcut(data['EconomicScore'], q=5, labels=['Lower', 'Lower-Middle', 'Middle', 'Upper-Middle', 'Upper'])

    # Enhanced Family Dynamics
    data['FamilyType'] = data.apply(lambda row: categorize_family(row['SibSp'], row['Parch']), axis=1)
    data['HasChildren'] = ((data['SibSp'] > 0) & (data['Age'] > 18)).astype(int)
    data['FamilyGenderBalance'] = data.apply(lambda row: family_gender_balance(row, data), axis=1)

    # Improved Cabin Location
    data['Cabin'] = data['Cabin'].fillna('U0')
    data['DeckNumber'] = data['Cabin'].apply(lambda x: ord(x[0]) - ord('A') + 1 if x[0].isalpha() else 0)
    data['CabinNumber'] = data['Cabin'].apply(lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    data['CabinPosition'] = data.apply(lambda row: cabin_position_score(row['DeckNumber'], row['CabinNumber']), axis=1)

    # Nuanced Cultural Background
    data['CulturalBackground'] = data.apply(lambda row: infer_cultural_background(row, data), axis=1)

    # Comprehensive Survival Advantage Score
    data['SurvivalAdvantageScore'] = data.apply(calculate_survival_advantage, axis=1)

    # Create age groups and fare categories
    age_bins = [0, 12, 18, 35, 60, np.inf]
    age_labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

    fare_bins = [0, 7.91, 14.454, 31, np.inf]
    fare_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    data['FareCategory'] = pd.cut(data['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)

    return data

def categorize_family(sibsp, parch):
    if sibsp == 0 and parch == 0:
        return 'Alone'
    elif sibsp > 0 and parch == 0:
        return 'Couple/Siblings'
    elif sibsp == 0 and parch > 0:
        return 'Single Parent'
    else:
        return 'Nuclear Family'

def family_gender_balance(row, data):
    family = data[data['FamilyIdentifier'] == row['FamilyIdentifier']]
    female_ratio = family['Sex'].value_counts(normalize=True).get('female', 0)
    return female_ratio

def cabin_position_score(deck, cabin_num):
    deck_score = (9 - deck) / 8  # A is highest, G is lowest
    cabin_score = 1 - (cabin_num / 200)  # Assuming max cabin number is around 200
    return (deck_score + cabin_score) / 2

def infer_cultural_background(row, data):
    if row['Embarked'] == 'S':
        return 'British' if row['Fare'] < 30 else 'American'
    elif row['Embarked'] == 'C':
        return 'French' if row['Fare'] < 60 else 'Continental European'
    elif row['Embarked'] == 'Q':
        return 'Irish'
    else:
        return 'Unknown'

def calculate_survival_advantage(row):
    score = 0
    if row['Sex'] == 'female':
        score += 3
    if row['Age'] <= 12:
        score += 2
    elif 12 < row['Age'] <= 50:
        score += 1
    score += (4 - row['Pclass']) * 2
    score += row['EconomicScore'] / 5
    score += row['CabinPosition'] * 3
    if row['FamilyType'] in ['Couple/Siblings', 'Nuclear Family']:
        score += 1
    return score

def generate_prompt(row):
    age = int(round(row['Age']))
    title = str(row['Title'])
    pclass = int(row['Pclass'])
    gender = "male" if row['Sex'] == "male" else "female"
    economic_tier = str(row['EconomicTier'])
    family_type = str(row['FamilyType'])
    has_children = "with children" if row['HasChildren'] == 1 else "without children"
    family_gender_balance = "mostly female" if row['FamilyGenderBalance'] > 0.5 else "mostly male" if row['FamilyGenderBalance'] < 0.5 else "balanced"
    cabin_position = "upper" if row['CabinPosition'] > 0.66 else "middle" if row['CabinPosition'] > 0.33 else "lower"
    cultural_background = str(row['CulturalBackground'])
    survival_advantage_score = int(row['SurvivalAdvantageScore'])

    prompt = f"Passenger details for Titanic voyage:\n"
    prompt += f"- A {age}-year-old {gender} {title}\n"
    prompt += f"- Traveling in {ordinal(pclass)} class, part of the {economic_tier} economic tier\n"
    prompt += f"- Part of a {family_type} group {has_children}, with a {family_gender_balance} gender balance\n"
    prompt += f"- Assigned to a {cabin_position} deck cabin\n"
    prompt += f"- Cultural background: {cultural_background}\n"
    prompt += f"- Overall survival advantage score: {survival_advantage_score}\n"

    prompt += "\nBased on these factors and your knowledge of the Titanic disaster, "
    prompt += "determine whether this passenger survived or did not survive. "
    prompt += "Provide your reasoning, then conclude with a clear statement of 'Survived' or 'Did not survive' on a new line."
    prompt += "\n\nReasoning:"

    return prompt

def ordinal(n):
    return "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def generate_jsonl(data):
    system_prompt = """You are an AI tasked with predicting the survival of Titanic passengers using all available details (e.g., age, gender, class, family size). Analyze the information and provide a concise reasoning process. Additionally, compute and include an 'Overall survival advantage score,' which quantifies the passenger's likelihood of survival based on historical and sociological factors. The score considers:

        1. Gender: Women receive a higher score due to the 'women and children first' policy.
        2. Age: Younger passengers receive a higher score.
        3. Passenger class: First-class passengers receive a higher score due to proximity to lifeboats and better access to information.
        4. Economic status: Passengers with higher economic scores (based on class and fare) receive a higher survival advantage.
        5. Cabin location: Upper deck cabins near lifeboats receive higher scores.
        6. Family structure: Small family groups receive a slight boost in score.

        The output should be in JSON format with the following structure:

        ```json
        {
        'reasoning': 'Brief reasoning here',
        'confidence': 0.75,
        'survival_advantage_score': 0.85,
        'prediction': 'Survived' or 'Did not survive'
        }
        ```

        Ensure that the 'prediction' key clearly states either 'Survived' or 'Did not survive,' and the 'survival_advantage_score' reflects the passenger's likelihood based on the above factors."

        ### Suggestions:
        1. **Refinement of Advantage Score**: If your model has specific formulas for calculating the survival advantage score, those could be embedded as part of the prompt logic for consistent scoring.
        2. **Optional Feature Weights**: Consider allowing custom weights for the factors listed above (e.g., prioritizing class over age) if you plan to test variations.
        3. **Score Interpretation**: If the 'survival_advantage_score' is a continuous value, specify any useful thresholds (e.g., scores above 0.7 are more likely to survive).

        ### Questions:
        1. Should the survival advantage score calculation follow a predefined formula, or do you prefer a more general qualitative assessment?
        2. Would you like additional reasoning details included, such as breaking down how each factor contributed to the score?
        3. Is the confidence score based solely on the model’s internal assessment, or should it also factor in the survival advantage score?
        """
    jsonl_data = []
    for _, row in data.iterrows():
        user_message = generate_prompt(row)

        system_message = (system_prompt)

        assistant_message = "Based on the provided information, I believe this passenger "
        assistant_message += "survived" if row['Survived'] == 1 else "did not survive"
        assistant_message += " the Titanic disaster. Here's my reasoning:\n\n"

        # Add reasoning based on known factors
        if row['Sex'] == 'female' or row['Age'] < 18:
            assistant_message += "- Women and children were given priority in lifeboats, increasing their chances of survival.\n"
        if row['Pclass'] == 1:
            assistant_message += "- First-class passengers had better access to lifeboats and information about the ship's condition.\n"
        if row['CabinPosition'] > 0.66:
            assistant_message += "- Passengers in upper deck cabins were closer to lifeboats and had a higher chance of escaping.\n"
        if row['Embarked'] == 'C':
            assistant_message += "- Passengers who boarded at Cherbourg were often wealthier and had better chances of survival.\n"
        if row['FamilySize'] > 1 and row['FamilySize'] < 5:
            assistant_message += "- Small to medium-sized family groups often stayed together, potentially increasing their chances of getting into a lifeboat.\n"
        if row['EconomicTier'] in ['Upper-Middle', 'Upper']:
            assistant_message += "- Passengers in higher economic tiers typically had better access to information and resources during the disaster.\n"
        if row['SurvivalAdvantageScore'] > 7:
            assistant_message += f"- This passenger's high survival advantage score of {row['SurvivalAdvantageScore']} suggests they had multiple factors in their favor.\n"

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
    train_data = load_and_preprocess_data('data/balanced_train.csv')
    jsonl_data = generate_jsonl(train_data)

    with open('data/train_refined_iter_3_balanced.jsonl', 'w') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + '\n')

    print("Enhanced JSONL file saved to data/train_refined_iter_3_balanced.jsonl")