import csv
import json
import os

# Configuration
FILENAME = "train_natural_lang_prompts_with_survival_reasoning_v1.jsonl"
base_dir = 'data/train'
input_file = f'{base_dir}/train.csv'
output_file = f'{base_dir}/jsonl/{FILENAME}'

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def create_user_message(passenger):
    age = f"{passenger['Age']} years old" if passenger['Age'] else "unknown age"
    fare = f"{float(passenger['Fare']):.2f}" if passenger['Fare'] else "unknown"
    cabin = f", cabin {passenger['Cabin']}" if passenger['Cabin'] else ""
    family = int(passenger['SibSp']) + int(passenger['Parch'])
    
    class_names = {"1": "1st", "2": "2nd", "3": "3rd"}
    pclass = class_names[passenger['Pclass']]
    
    embarkation_ports = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    embarked = embarkation_ports.get(passenger['Embarked'], "unknown port")

    return f"Predict the survival of a {age} {passenger['Sex']} passenger in {pclass} class, " \
           f"traveling with {family} family members, who boarded at {embarked} and paid {fare} for their ticket{cabin}."

def create_assistant_message(passenger):
    survived = "survived" if passenger['Survived'] == '1' else "did not survive"
    
    reasons = []
    if passenger['Sex'] == 'female':
        reasons.append("being female generally increased survival chances")
    if passenger['Pclass'] == '1':
        reasons.append("first-class passengers had better survival rates")
    elif passenger['Pclass'] == '3':
        reasons.append("third-class passengers had lower survival rates")
    if passenger['Age'] and float(passenger['Age']) < 18:
        reasons.append("children were often prioritized for rescue")
    if passenger['Fare'] and float(passenger['Fare']) > 100:
        reasons.append("passengers with expensive tickets may have had better access to lifeboats")
    
    reasoning = " and ".join(reasons)
    return f"Based on the information provided, this passenger {survived}. " \
           f"Reasoning: {reasoning.capitalize() if reasoning else 'Multiple factors influenced survival rates, including class, gender, age, and location on the ship.'}"

def main():
    data = read_csv(input_file)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for passenger in data:
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant trained to analyze Titanic passenger data and predict survival outcomes. Provide clear and concise predictions based on the given information."
                    },
                    {
                        "role": "user",
                        "content": create_user_message(passenger)
                    },
                    {
                        "role": "assistant",
                        "content": create_assistant_message(passenger)
                    }
                ]
            }
            json.dump(conversation, f)
            f.write('\n')

if __name__ == "__main__":
    main()