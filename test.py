import pandas as pd

# Define the column names for the master table
columns = ['badgenumber'] + transformed_question_list_df.columns[1:].tolist()

# Create the master table with 10 rows for the 10 distinct badgenumbers
master_table_df = pd.DataFrame(columns=columns)
master_table_df['badgenumber'] = transform_df['badgenumber'].unique()

# Iterate through each row of the master table and set the values for each question
for index, row in master_table_df.iterrows():
    badgenumber = row['badgenumber']
    for col in transformed_question_list_df.columns[1:]:
        question_id = int(col.split('_')[1])
        answer_id = transform_df.loc[transform_df['badgenumber'] == badgenumber, 'answer_id'].iloc[0]
        if answer_id in question_answer_list.loc[question_answer_list['question_id'] == question_id, 'answer_id'].tolist():
            answer_text = question_answer_list.loc[(question_answer_list['question_id'] == question_id) & (question_answer_list['answer_id'] == answer_id), 'answer_text'].iloc[0]
            master_table_df.loc[index, col] = answer_text
        else:
            master_table_df.loc[index, col] = ''
