# Define a function to extract answer IDs from the profileresponses column
def extract_answerIds(profileresponses):
    answer_ids = []
    for response in profileresponses:
        if response['answerIds']:
            answer_ids.extend(response['answerIds'])
    return answer_ids

# Load the input data into a Pandas DataFrame
df = pd.read_csv('input.csv')

# Convert the profileresponses column from a string to a list of dictionaries
df['profileresponses'] = df['profileresponses'].apply(eval)

# Extract the answer IDs from the profileresponses column
df['answerIds'] = df['profileresponses'].apply(extract_answerIds)

# Convert the answerIds column to a string
df['answerIds'] = df['answerIds'].apply(str)
