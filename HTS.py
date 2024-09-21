from scripts.file_formats import *
from scripts.models import *
from scripts.test_output import *
from scripts.sequence_processing import *
import numpy as np
import pandas as pd
import os
import gc
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # some TF spam


DUMMY_MODE = False # Set to true to check if the code runs without errors,

    


hts_path = 'data/HTS/'
PROTEINS = [f for f in os.listdir(hts_path) if os.path.isdir(os.path.join(hts_path, f))]
OG_PROTEINS = ['NFKB1','LEF1','NACC2','RORB','TIGD3']
FINAL_PROTEINS = list(set(PROTEINS) - set(OG_PROTEINS))
if DUMMY_MODE:
    FINAL_PROTEINS = FINAL_PROTEINS[:1]
FINAL_PROTEINS_WITHOUT_HTS_TEST = ['ZFTA', 'ZNF500']
FINAL_PROTEINS_WITH_HTS_TEST  = list(set(FINAL_PROTEINS) - set(FINAL_PROTEINS_WITHOUT_HTS_TEST))

MODEL_VERSION = 'v35'
DO_TRAINING = True
TEST_SELF = True
TEST_OTHERS = True
AGG_ID = '1_2'


if DO_TRAINING:

    for protein in FINAL_PROTEINS:\
        # In case of reruns -> don't train if model already exists
        if check_if_ensemble_model_exists('HTS', protein, MODEL_VERSION, 5):
            print(f'Model for {protein} already exists')
            continue

        print("Training model for {}".format(protein))
        data = pd.read_pickle(f'data/HTS/{protein}/sequences_data_with_aliens_no_duplicates.pkl') # TODO: was good except NAAC 2?

        sequences_one_hot, labels = data['Sequence'].values, data['intensity'].values
        sequences_one_hot = np.stack(sequences_one_hot)

        # Was used to mess with different scaling for labels
        label_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4
        } 
        labels = [label_dict[label] for label in labels]
        sequences_one_hot = sequences_one_hot[np.array(labels) != -1]
        labels = np.array(labels)[np.array(labels) != -1]

        labels = np.stack(labels)
        sequences_one_hot = np.stack(sequences_one_hot)
        del data

        # We aren't sure why it's here but we keep it as a good luck charm
        sequences_one_hot = np.stack(sequences_one_hot)
        
        # make into boolean array to save memory
        sequences_one_hot = sequences_one_hot.astype(np.bool_)

        #  Train and save the model
        params = {'experiment': 'HTS',
                'steps_per_epoch': 400, 
                'patience': 40,} 
        params['model_type'] = 'cnn'
        params['metrics'] = None
        params['epochs'] = 300 
        params['validation_split'] = 0.04
        params['dense_list'] = params.get('dense_list', [(128, 0.2), (64,0),(32,0), (32, 0)])
        params['patience'] = 10
        ensemble = get_ensemble(sequences_one_hot, labels, params, n_splits=5)
        save_ensemble_model(ensemble, 'HTS', protein, MODEL_VERSION)

        # free memory
        del ensemble
        del sequences_one_hot
        del labels
        gc.collect()





# HTS to HTS
if TEST_SELF:
    experiment_to_preds = {'HTS': {}}
    seqs_one_hot = load_test_sequences_onehot('HTS', reverse_compliment=False, final_test=True)
    for protein in FINAL_PROTEINS_WITH_HTS_TEST:
        intensities = []
        ensemble = load_ensemble_model('HTS', protein, MODEL_VERSION, num_models=5)
        experiment_to_preds['HTS'][protein] = ensemble_predict(ensemble, seqs_one_hot)
    create_output(experiment_to_preds['HTS'], base_method='HTS', target_method='HTS', custom_name=MODEL_VERSION, test=True)



TARGET_EXPERIMENTS = ['CHS', 'GHTS']
K_SIZES = {'GHTS': [301, 101, 51], # Agg 1
            'CHS': [301, 151, 51]} # Agg 2

# HTS to others
# Here we make predictions for each window sized 'k'
if TEST_OTHERS:
    for protein in FINAL_PROTEINS:
        print(f'Testing {protein}')
        ensemble = load_ensemble_model('HTS', protein, MODEL_VERSION, num_models=5)

        for experiment in TARGET_EXPERIMENTS:
            if all([check_if_partial_predictions_exist('HTS',experiment, MODEL_VERSION, k, protein, False) for k in K_SIZES[experiment]]):
                print(f'\tAlready calculated {experiment} {protein}, reverse_compliment: {False}')
                continue

            seqs_one_hot = load_test_sequences_onehot(experiment, reverse_compliment=False, final_test=True)
            print(f'\tLoaded {experiment} sequences, reverse_compliment: {False}')
            for k in K_SIZES[experiment]:
                print(f'\t\tk: {k}')
                if not check_if_partial_predictions_exist("HTS", experiment, MODEL_VERSION, k, protein,False):
                    seqs_one_hot = trim_sequences(seqs_one_hot, k)
                    params = {'model_type': 'cnn'}
                    params['validation_split'] = 0.04
                    params['patience'] = 10
                    longer_input_ensemble = [get_converted_model(original_model=model, n_nucleotides=k, params=params) for model in ensemble]
                    print('\t\t',end='')
                    intensities = ensemble_predict(longer_input_ensemble, seqs_one_hot)
                    save_partial_prediction("HTS", experiment, intensities, MODEL_VERSION, k, protein,False)
        print('')


# HTS to others
# Here we average the predictions for each window size 'k' to get the final predictions
if TEST_OTHERS:
    for experiment in TARGET_EXPERIMENTS:
        experiment_to_preds = {}
        for protein in FINAL_PROTEINS:
            partial_preds = []
            for k in K_SIZES[experiment]:
                partial_preds.append(load_partial_predictions("HTS", experiment, MODEL_VERSION, k, protein,False))

            # average the predictions
            intensities = np.mean(partial_preds, axis=0)
            experiment_to_preds[protein] = intensities
        
        create_output(experiment_to_preds=experiment_to_preds, base_method='HTS', target_method=experiment, custom_name=MODEL_VERSION + '_' + AGG_ID, test=True)