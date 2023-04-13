def get_related_answer(answer_id, df):
    if pd.isna(answer_id):
        return None
    else:
        related_answer = []
        for ans in answer_id:
            if ans in df['answer_id'].tolist():
                related_answer.append(ans)
        return related_answer if related_answer else None

# create the related_answer column in JISJA23_2
JISJA23['related_answer'] = JISJA23['answer_id'].apply(get_related_answer, args=(df3,))
