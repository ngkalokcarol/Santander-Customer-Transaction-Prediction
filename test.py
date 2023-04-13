def extract_answerIds(row):
    answer_ids_str = row['profileresponses']
    answer_ids = re.findall(r'"answerIds":\s*\[([\d,\s]*)\]', answer_ids_str)
    if answer_ids:
        answer_ids = answer_ids[0].split(',')
        answer_ids = [int(id) for id in answer_ids if id.strip().isdigit()]
    else:
        answer_ids = []
    return answer_ids

df['answerIds'] = df.apply(extract_answerIds, axis=1)

print(df.head())
