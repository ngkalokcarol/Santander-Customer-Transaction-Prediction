# Merge transform_df with question_answer_list on 'answer_id'
merged_df = transform_df.merge(question_answer_list, on='answer_id')

# Pivot the merged_df to create the transformed_question_list
transformed_question_list_df = merged_df.pivot(index='question_id', columns='badgenumber', values='answer_text').reset_index().rename(columns={'question_id': ''})

# Merge the transformed_question_list with the master table on 'badgenumber'
master_table_df = transformed_question_list_df.merge(transform_df[['badgenumber', 'answer_id']], on='badgenumber')

# Iterate through each row of the master_table_df and update the corresponding value in transformed_question_list_df
for index, row in master_table_df.iterrows():
    badgenumber = row['badgenumber']
    for col in transformed_question_list_df.columns[1:]:
        answer_id = row['answer_id']
        question_id = col
        if pd.notna(answer_id) and answer_id in question_answer_list[question_answer_list['question_id'] == question_id]['answer_id'].tolist():
            answer_text = merged_df.loc[(merged_df['answer_id'] == answer_id) & (merged_df['question_id'] == question_id), 'answer_text'].iloc[0]
            transformed_question_list_df.loc[index, col] = answer_text

# Set the index of the transformed_question_list_df to 'badgenumber'
transformed_question_list_df.set_index('badgenumber', inplace=True)
