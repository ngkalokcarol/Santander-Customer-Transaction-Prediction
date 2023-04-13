def get_related_answer(row):
    if isinstance(row['answer_id'], list) and row['related_answer'] is None:
        related_answer = []
        for ans_id in row['answer_id']:
            if ans_id in df3['answer_id'].values:
                related_answer.append(ans_id)
        return related_answer if related_answer else None
    else:
        return row['related_answer']

# create the related_answer column in JISJA23_2
JISJA23['related_answer'] = JISJA23['answer_id'].apply(get_related_answer, args=(df3,))
