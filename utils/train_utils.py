import pandas as pd


def load_smiles(csv_file, smiles_columns="smiles"):
    '''
    :param csv_file: should be {dataset}_processed_ac.csv
    :return:
    '''
    df = pd.read_csv(csv_file)
    smiles = df[smiles_columns].values.flatten().tolist()
    return smiles

