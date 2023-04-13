# convert answer_id column in B2B_Questions_Answers to integer
B2B_Questions_Answers['answer_id'] = B2B_Questions_Answers['answer_id'].astype(int)

# merge the datasets on answer_id
merged_df = JISJA23.merge(B2B_Questions_Answers, left_on='related_answer', right_on='answer_id', how='left')

# create a new column 'related_answer_text' by mapping answer_text to related_answer
merged_df['related_answer_text'] = merged_df['related_answer'].apply(lambda x: [B2B_Questions_Answers.loc[B2B_Questions_Answers['answer_id'] == i, 'answer_text'].values[0] for i in x.split(',') if i.isdigit()] if isinstance(x, str) else x)
