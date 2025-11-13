"""
`partiju_skirotava.json` comes from https://www.lsm.lv/?rt=velesanas22/sorting&ac=init
(Network -> Fetch/XHR -> ?rt=velesanas... Copy response)
"""
import json
import pandas as pd
import os

def load_and_clean_data(config):
    # fetch the parties defined in config
    defined_parties = list(config['colours'].keys())

    # load the data file
    with open(config['data']['SOURCE_DATA_PATH'], 'r') as file:
        data = json.load(file)

    party_dict = data['parties']
    question_dict = data['questions']

    questions = []
    for row in question_dict:
        questions.append(row['question'])

    answer_dict = {}
    for row in party_dict:
        party_options = [p for p in defined_parties if row['id'] in p or p in row['id']]
        if len(party_options)>0:
            party = party_options[0]
            print(f"{row['id']} >>>> {party}")
        else:
            print(f"NOT IN CONFIG: {row['id']}")
            continue
        
        answers = row['answers']
        if answers is not None:
            answer_dict[party] = list(answers.values())
        else:
            print(f"{party} has not answered!")

    # save parties' answers
    final_df = pd.DataFrame(answer_dict).T-1
    final_df.columns = questions

    # save questions if the manually supplemented topic-mapping file does not exist
    if not os.path.exists(config['data']['TOPIC_MAPPING_PATH']):
        question_df = pd.DataFrame(questions, columns=['jautājums'])
        question_df['tēma'] = pd.NA
        question_df.to_excel(config['data']['TOPIC_MAPPING_PATH'], index=False)
        print(f"Go to {config['data']['TOPIC_MAPPING_PATH']} and and topics!")

    return final_df