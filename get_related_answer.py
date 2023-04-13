# create a set of unique answer_ids in df3
df3_answer_ids = set(df3['answer_id'])

# create a new column 'related_answer' in JISJA23_2
JISJA23_2['related_answer'] = JISJA23_2['answer_id'].apply(lambda x: [val for val in x if val in df3_answer_ids])
