import pandas as pd

# Load the DataFrame
df = pd.read_csv('your_csv_file.csv')

# Define a list to store the answer IDs
answer_ids_list = []

for index, row in df.iterrows():
    answer_ids_str = json.loads(row['profileresponses'])[0]['answerIds']
    if answer_ids_str:
        answer_ids = [int(id) for id in answer_ids_str if id.isdigit()]
    else:
        answer_ids = []
    df.at[index, 'answerIds'] = answer_ids
    
    
# Add a new column to the DataFrame to store the answer IDs
df['answerIds'] = answer_ids_list
