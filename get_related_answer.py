def get_related_answer(row):
    if isinstance(row['answer_id'], list) and pd.isna(row['related_answer']):
        related_answer = []
        for ans_id in row['answer_id']:
            if ans_id in df3['answer_id'].values:
                related_answer.append(ans_id)
        return related_answer if related_answer else None
    else:
        return row['related_answer']

JISJA23['related_answer'] = JISJA23.apply(get_related_answer, axis=1)
