JISJA23_2['related_answer'] = JISJA23_2['answer_id'].apply(lambda x: [aid for aid in x if aid in df3['answer_id'].tolist()])
