import csv
import json
import os

# Configuration
FILENAME = "test_data_for_prediction_v3.jsonl"
base_dir = 'data/test'
input_file = f'{base_dir}/test.csv'
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

def main():
    data = read_csv(input_file)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for passenger in data:
            entry = {
                "PassengerId": int(passenger['PassengerId']),
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
                        "content": "Prediction: "
                    }
                ]
            }
            json.dump(entry, f)
            f.write('\n')

    print(f"Created {len(data)} prediction prompts in {output_file}")

if __name__ == "__main__":
    main()