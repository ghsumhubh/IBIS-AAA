import os
import pandas as pd


def save_partial_prediction(experiment_base, experiment_target, preds, custom_name, k, protein, reverse=False):
    folder_path = f"output/{experiment_target}/{experiment_base}/partial_predictions/{protein}/{reverse}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    name = f'{custom_name}_{k}'

    file_path = name + '.csv'

    output_path = os.path.join(folder_path, file_path)

    df = pd.DataFrame(preds, columns=[experiment_target])

    df.to_csv(output_path, index=False)

def check_if_partial_predictions_exist(experiment_base, experiment_target, custom_name, k, protein, reverse=False):
    folder_path = f"output/{experiment_target}/{experiment_base}/partial_predictions/{protein}/{reverse}/"
    if not os.path.exists(folder_path):
        return False

    name = f'{custom_name}_{k}'

    file_path = name + '.csv'

    output_path = os.path.join(folder_path, file_path)

    return os.path.exists(output_path)


def load_partial_predictions(experiment_base, experiment_target, custom_name, k, protein, reverse=False):
    folder_path = f"output/{experiment_target}/{experiment_base}/partial_predictions/{protein}/{reverse}/"
    name = f'{custom_name}_{k}'

    file_path = name + '.csv'

    output_path = os.path.join(folder_path, file_path)

    df = pd.read_csv(output_path)
    return df[experiment_target].values




def create_output(experiment_to_preds, base_method, target_method, custom_name = None, test=False):
    folder_path = f"output/{target_method}/{base_method}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if test:
        df = pd.read_csv("data/Test/test_sequences.csv")
    else:
        df = pd.read_csv("data/test_sequences.csv")
    df = df[df['experiment'] == target_method]
    test_identifiers = df['Identifier'].values


    experiment_to_column_name = {'PBM': 'probes',
                                 'CHS': 'peaks',
                                 'GHTS': 'peaks',
                                  'HTS': 'reads',
                                    'SMS': 'reads'}
    

    
    experiment_to_column_name = {'PBM': 'probes',
                                 'CHS': 'peaks',
                                 'GHTS': 'peaks',
                                  'HTS': 'reads',
                                    'SMS': 'probes'}
    
    
    name = f'{base_method}_to_{target_method}'
    file_path = name + '.tsv.gz'
    if custom_name:
        file_path = 'output_' + custom_name + '.tsv.gz'
    else:
        file_path = 'output.tsv.gz'
    output_path = os.path.join(folder_path, file_path)
    df = pd.DataFrame(test_identifiers, columns=[experiment_to_column_name[target_method]])

    for experiment, preds in experiment_to_preds.items():
        df[experiment] = preds
        # normaloze to range of 0-1
        df[experiment] = (df[experiment] - df[experiment].min()) / (df[experiment].max() - df[experiment].min())

        # up to 5 decimal points
        df[experiment] = df[experiment].apply(lambda x: round(x, 5))

    df.to_csv(output_path, sep='\t', index=False, compression='gzip')