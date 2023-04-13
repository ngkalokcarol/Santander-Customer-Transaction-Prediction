# Merge the JISJA23 and B2B_Questions_Answers datasets on the 'answer_id' column
merged_df = JISJA23.merge(B2B_Questions_Answers, left_on='related_answer', right_on='answer_id', how='left')

# Create an empty list to store the related answer texts
related_answer_text = []

# Iterate over the rows in the merged dataframe
for i, row in merged_df.iterrows():
    # Get the related answers for the current row
    related_answers = row['related_answer']
    
    # If there are no related answers, append an empty list to the related_answer_text list
    if pd.isna(related_answers):
        related_answer_text.append([])
    else:
        # Split the related answers into a list of integers
        related_answers = [int(x) for x in related_answers.split(',')]
        
        # Get the corresponding answer texts from B2B_Questions_Answers
        answer_texts = B2B_Questions_Answers.loc[B2B_Questions_Answers['answer_id'].isin(related_answers), 'answer_text'].tolist()
        
        # Append the answer texts to the related_answer_text list
        related_answer_text.append(answer_texts)
        
# Add the related_answer_text column to the merged dataframe
merged_df['related_answer_text'] = related_answer_text
