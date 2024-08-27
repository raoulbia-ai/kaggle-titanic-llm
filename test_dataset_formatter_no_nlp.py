import csv
import json

def convert_csv_to_json(input_file, output_file):
    with open(input_file, 'r') as csvfile, open(output_file, 'w') as jsonfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            passenger_id = row['PassengerId']
            content = "Passenger details for Titanic voyage:\n"
            for key, value in row.items():
                if key != 'PassengerId':
                    content += f"- {key}: {value}\n"
            
            content += "\nBased on these factors and your knowledge of the Titanic disaster, determine whether this passenger survived or did not survive. Provide your reasoning, then conclude with a clear statement of 'Survived' or 'Did not survive' on a new line.\n\nReasoning:"

            json_data = {
                "PassengerId": int(passenger_id),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an assistant that predicts Titanic passenger survival. Analyze the given information and provide your reasoning. Conclude with a clear 'Survived' or 'Did not survive' statement on a new line."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
            json.dump(json_data, jsonfile)
            jsonfile.write('\n')

# Usage
input_file = 'data/test.csv'
output_file = 'data/test_no_nlp.jsonl'
convert_csv_to_json(input_file, output_file)

print(f"Conversion complete. Output saved to {output_file}")