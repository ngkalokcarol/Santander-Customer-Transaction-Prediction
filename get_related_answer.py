def convert_to_list(row):
    if pd.isna(row):
        return np.nan
    try:
        return [int(num) for num in row.split(",")]
    except ValueError:
        return np.nan
    
JISJA23['answer_id'] = JISJA23['answer_id'].apply(convert_to_list)





# define a function to get the related answers
def get_related_answer(row, df):
    if isinstance(row['answer_id'], list) and pd.isnull(row['related_answer']):
        related_answer = []
        for ans_id in row['answer_id']:
            if ans_id in df['answer_id'].values:
                related_answer.append(ans_id)
        return related_answer if related_answer else np.nan
    else:
        return row['related_answer']

# apply the function to JISJA23
JISJA23['related_answer'] = JISJA23.apply(get_related_answer, args=(df3,), axis=1)
