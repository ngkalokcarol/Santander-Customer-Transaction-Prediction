def convert_to_list(row):
    if isinstance(row, str):
        try:
            return [int(num) for num in row.split(",")]
        except ValueError:
            return np.nan
    else:
        return np.nan
    
JISJA23['answer_id'] = JISJA23['answer_id'].apply(convert_to_list)





# define a function to check if the answer id is in df3
def get_related_answer(row):
    if isinstance(row['answer_id'], list) and row['related_answer'] is None:
        related_answer = []
        for ans_id in row['answer_id']:
            if str(ans_id) in df3['answer_id'].values:
                related_answer.append(ans_id)
        return related_answer if related_answer else None
    else:
        return row['related_answer']
    
# create the related_answer column in JISJA23
JISJA23['related_answer'] = None
JISJA23['related_answer'] = JISJA23.apply(get_related_answer, axis=1)
