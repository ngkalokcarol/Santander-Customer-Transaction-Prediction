def get_related_answer(row):
    if not pd.isna(row['answer_id']):
        answer_ids = [int(id) for id in row['answer_id']]
        related_answers = []
        for answer_id in answer_ids:
            if answer_id in df3['answer_id'].values:
                related_answers.append(answer_id)
        return related_answers
    else:
        return None

JISJA23_2['related_answer'] = JISJA23_2.apply(get_related_answer, axis=1)
