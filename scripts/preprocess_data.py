import pandas as pd
import os

def load_and_clean_data(folder):
    dfs = []
    for file in os.listdir(folder):
        df = pd.read_excel(os.path.join(folder, file), header=None)
        df = df.dropna().reset_index(drop=True)
        new_df = pd.DataFrame(
            df.loc[1::2].values,
            index=df.loc[::2].values.ravel(),
            columns=[file.split('_')[-1].split('.')[0]]
        )
        dfs.append(new_df)
    combined_df = pd.concat(dfs, axis=1)
    return combined_df

def encode_answers(df):
    encoding_dict = {
        'Partijas atbilde:\xa0pilnībā nepiekrītu': 0,
        'Partijas atbilde:\xa0daļēji piekrītu': 3,
        'Partijas atbilde:\xa0daļēji nepiekrītu': 1,
        'Partijas atbilde:\xa0pilnībā piekrītu': 4,
        'Partijas atbilde:\xa0gan piekrītu, gan nepiekrītu': 2
    }
    return df.apply(lambda col: col.map(encoding_dict)).T