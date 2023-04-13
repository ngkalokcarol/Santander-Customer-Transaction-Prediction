# apply the function to the answer_id column in JISJA23
JISJA23['answer_id'] = JISJA23['answer_id'].apply(convert_to_list)

# merge the JISJA23 and B2B_Questions_Answers datasets on the 'answer_id' column
merged_df = pd.merge(JISJA23, B2B_Questions_Answers, left_on='answer_id', right_on='answer_id', how='left')

# group the merged dataframe by 'answer_id' and 'question_id', and aggregate the 'answer_text' column
grouped_df = merged_df.groupby(['answer_id', 'question_id'])['answer_text'].apply(list).reset_index(name='related_answer_text')

# merge the grouped dataframe back into the original JISJA23 dataframe
final_df = pd.merge(JISJA23, grouped_df, on=['answer_id', 'question_id'], how='left')
