def get_related_answer(row, df):
    """
    Function to match the answer IDs between two data frames and create a new column 'related_answer' in the first data
    frame with the matching IDs.
    """
    if isinstance(row['answer_id'], list) and 'related_answer' not in row.index:
        related_answer = []
        for ans_id in row['answer_id']:
            if ans_id in df['answer_id'].values:
                related_answer.append(ans_id)
        return related_answer if related_answer else np.nan
    else:
        return row['related_answer']

# create the related_answer column in JISJA23_2
JISJA23['related_answer'] = JISJA23.apply(get_related_answer, args=(df3,), axis=1)
