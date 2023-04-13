# Split the 'related_answer' column of JISJA23 into individual rows
# Drop rows with NaN values, and convert each value to integer
JISJA23 = JISJA23.explode('related_answer').dropna()
JISJA23['related_answer'] = JISJA23['related_answer'].astype(int)

# Merge JISJA23 with B2B_Questions_Answers on 'answer_id'
merged = pd.merge(JISJA23, B2B_Questions_Answers, how='left', left_on='related_answer', right_on='answer_id')

# Group by the index of the original JISJA23 dataframe, and aggregate the related_answer_text column as a list
grouped = merged.groupby(merged.index)['answer_text'].agg(list)

# Join the grouped data with the original JISJA23 dataframe
JISJA23 = JISJA23.join(grouped, how='left')

# Rename the related_answer_text column to related_answer_text_list
JISJA23 = JISJA23.rename(columns={'answer_text': 'related_answer_text_list'})

# Create a new column 'related_answer_text' by converting the list to a string
JISJA23['related_answer_text'] = JISJA23['related_answer_text_list'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else '')

# Drop the related_answer_text_list column
JISJA23 = JISJA23.drop(columns=['related_answer_text_list'])

# Save the result to a new csv file
JISJA23.to_csv('path/to/output.csv', index=False)
