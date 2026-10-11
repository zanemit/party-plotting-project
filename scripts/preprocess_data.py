"""
`partiju_skirotava.json` comes from https://www.lsm.lv/?rt=velesanas22/sorting&ac=init
(Network -> Fetch/XHR -> ?rt=velesanas... Copy response)
"""
import json
import pandas as pd
import os
from pathlib import Path

def load_and_clean_data(config):
    # fetch the parties defined in config
    defined_parties = list(config['colours'].keys())

    # load the data file
    final_df_list=[]
    for year in config['data']['YEARS']:
        source_data_path = Path(config['data']['SOURCE_DATA_PATH'])
        source_data_path = source_data_path.with_name(
            f"{source_data_path.stem}_{year}{source_data_path.suffix}"
            )
        with open(source_data_path, 'r') as file:
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
        output_data_path = Path(config['data']['TOPIC_MAPPING_PATH'])
        output_data_path = output_data_path.with_name(
            f"{output_data_path.stem}_{year}{output_data_path.suffix}"
        )
        if not os.path.exists(output_data_path):
            question_df = pd.DataFrame(questions, columns=['jautājums'])
            question_df['tēma'] = pd.NA
            question_df['virstēma'] = pd.NA
            question_df.to_excel(output_data_path, index=False)
            print(f"Go to {output_data_path} and add topics!")
        final_df_list.append(final_df)

    return final_df_list