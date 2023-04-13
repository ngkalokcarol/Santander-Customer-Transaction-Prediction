import pandas as pd

# Load the DataFrame
df = pd.read_csv('your_csv_file.csv')

# Define a list to store the answer IDs
answer_ids_list = []

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    # Get the value of 'answerIds' for the given row
    answer_ids_str = row['profileresponses'].get('answerIds')
    
    # Check if answer_ids_str is empty or None
    if answer_ids_str:
        # Split the answer IDs string into a list of individual IDs
        answer_ids = answer_ids_str.split(',')
        
        # Convert each answer ID to an integer and add it to the answer_ids_list
        for answer_id in answer_ids:
            try:
                answer_ids_list.append(int(answer_id))
            except ValueError:
                pass
    else:
        # If answerIds is empty, append None to the answer_ids_list
        answer_ids_list.append(None)

# Add a new column to the DataFrame to store the answer IDs
df['answerIds'] = answer_ids_list
