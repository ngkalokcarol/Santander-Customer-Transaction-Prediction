def get_related_answer(answer_ids):
    related_answer = []
    for answer_id in answer_ids:
        if answer_id in df3['question_id'].values:
            related_answer.append(answer_id)
    return related_answer

JISJA23_3['related_answer'] = JISJA23_3['answerIds'].apply(get_related_answer)
