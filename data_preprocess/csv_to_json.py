import pandas as pd
import json

# Load the CSV data into a DataFrame
data = pd.read_csv('./datasets/cnndm/train/cnndm_train.csv')

# Create a list to hold the JSON-formatted data
json_data = []

# Iterate over each row in the DataFrame and convert to JSON format
for index, row in data.iterrows():
    # Create a dictionary for each record
    record = {
        "id": index,
        "src": row["article"],
        "original_summary": row["highlights"],
    }
    # Append the dictionary to the list
    json_data.append(record)

# Save the JSON data to a file
# Replace 'path/to/your/output.json' with the desired output file path
with open('./datasets/cnndm/train/cnndm_train_287113.json', 'w') as json_file:
    # Convert the list of dictionaries to JSON formatted string
    # Use indent=4 for pretty printing with 4 spaces indentation
    json.dump(json_data, json_file, indent=4)
