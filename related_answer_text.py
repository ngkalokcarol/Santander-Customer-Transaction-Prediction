# explode the lists in JISJA23
JISJA23 = JISJA23.explode('related_answer')

# join the dataframes using 'answer_id' column in B2B_Questions_Answers and 'related_answer' in JISJA23
merged_df = JISJA23.merge(B2B_Questions_Answers, left_on='related_answer', right_on='answer_id', how='left')

# create new column 'related_answer_text' containing answer_text values
merged_df['related_answer_text'] = merged_df['answer_text']

# group the rows by the original index and aggregate the 'related_answer_text' values into lists
merged_df = merged_df.groupby(merged_df.index)['related_answer_text'].apply(list)

# merge the aggregated 'related_answer_text' column with the original dataframe
JISJA23 = pd.concat([JISJA23, merged_df], axis=1)

# print the resulting dataframe
print(JISJA23)
