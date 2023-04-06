# Create an empty master table
master_table_df = pd.DataFrame(columns=['badgenumber'] + transformed_question_list_df[''].tolist())

# Iterate through each row of the transform_df and merge the data into the master table
for index, row in transform_df.iterrows():
    badgenumber = row['badgenumber']
    answer_id = row['answer_id']
    answer_text = row['answer_text']
    
    # Find the corresponding question id for the answer id in question_answer_list
    question_id = question_answer_list.loc[question_answer_list['answer_id'] == answer_id, 'question_id'].iloc[0]
    
    # Update the corresponding value in the master table
    master_table_df.loc[master_table_df['badgenumber'] == badgenumber, str(question_id)] = answer_text
    
# Set the index of the master table as 'badgenumber'
master_table_df.set_index('badgenumber', inplace=True)

# Merge transformed_question_list_df into the master table
master_table_df = pd.merge(master_table_df, transformed_question_list_df, left_index=True, right_index=True)

# Reset the index and rename the index column to an empty string
master_table_df.reset_index(inplace=True)
master_table_df.rename(columns={'index': ''}, inplace=True)

# Display the master table
print(master_table_df)
