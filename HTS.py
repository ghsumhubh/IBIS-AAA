from scripts.file_formats import *
from scripts.models import *
#from scripts.torch_models import *
from scripts.test_output import *
# from scripts.knn_models import *
from scripts.sequence_processing import *
import numpy as np
import pandas as pd
import os
import gc

import warnings
warnings.filterwarnings('ignore', category=UserWarning) # some TF spam

hts_path = 'data/HTS/'
PROTEINS = [f for f in os.listdir(hts_path) if os.path.isdir(os.path.join(hts_path, f))]
OG_PROTEINS = ['NFKB1','LEF1','NACC2','RORB','TIGD3']
FIRST_PROTEIN = ['NFKB1']
TEST_PROTEINS = list(set(PROTEINS) - set(OG_PROTEINS))
TEST_PROTEINS_WITHOUT_TEST = ['ZFTA', 'ZNF500']
TEST_PROTEINS_WITH_TEST  = list(set(TEST_PROTEINS) - set(TEST_PROTEINS_WITHOUT_TEST))

MODEL_VERSION = 'v35'
DO_TRAINING = False
TEST_SELF = False
TEST_OTHERS = True
AGG_ID = '1_2'


if DO_TRAINING:

    for protein in TEST_PROTEINS:
        #if check_if_protein_model_exists('HTS', protein, custom_name=MODEL_VERSION):
        if check_if_ensemble_model_exists('HTS', protein, MODEL_VERSION, 5):
            print(f'Model for {protein} already exists')
            continue

        print("Training model for {}".format(protein))

        #data = pd.read_pickle(f'data/HTS/{protein}/sequences_data_with_aliens_unique.pkl')
        #data = pd.read_pickle(f'data/HTS/{protein}/sequences_data_with_aliens.pkl')
        data = pd.read_pickle(f'data/HTS/{protein}/sequences_data_with_aliens_no_duplicates.pkl') # TODO: was good except NAAC 2?

        sequences_one_hot, labels = data['Sequence'].values, data['intensity'].values
        sequences_one_hot = np.stack(sequences_one_hot)


        label_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4
        } 
        #print(f'Before: {len(labels)}')
        labels = [label_dict[label] for label in labels]
        #print(f'After: {len(labels)}')
        sequences_one_hot = sequences_one_hot[np.array(labels) != -1]
        labels = np.array(labels)[np.array(labels) != -1]
        labels = np.stack(labels)



        sequences_one_hot = np.stack(sequences_one_hot)


        del data

        # stack
        sequences_one_hot = np.stack(sequences_one_hot)
        
        # make into boolean array
        sequences_one_hot = sequences_one_hot.astype(np.bool_)
        params = {'experiment': 'HTS',
                'steps_per_epoch': 400, 
                'patience': 40,} 
        params['model_type'] = 'cnn'
        params['metrics'] = None
        params['epochs'] = 300 # was 200
        params['validation_split'] = 0.04
        # params['n_filters'] = params.get('n_filters', 64)
        # params['sizes_to_use'] = params.get('sizes_to_use', [4,8,16,32])

        # CHANGE LATER TO TEST
        params['dense_list'] = params.get('dense_list', [(128, 0.2), (64,0),(32,0), (32, 0)])


        params['patience'] = 10

        #model = get_trained_model(sequences_one_hot, labels, params)
        ensemble = get_ensemble(sequences_one_hot, labels, params, n_splits=5)
        #save_protein_model(model=model, experiment='HTS', protein=protein, custom_name=MODEL_VERSION)
        save_ensemble_model(ensemble, 'HTS', protein, MODEL_VERSION)

        # free memory
        #del model
        del ensemble
        del sequences_one_hot
        del labels
        gc.collect()






if TEST_SELF:

    #seqs_one_hot_reverse_compliment = load_test_sequences_onehot('HTS', reverse_compliment=True)
    experiment_to_preds = {'HTS': {}}
    seqs_one_hot = load_test_sequences_onehot('HTS', reverse_compliment=False, final_test=True)

    for protein in TEST_PROTEINS_WITH_TEST:
        intensities = []
        
        #model = load_protein_model(experiment='HTS', protein=protein, custom_name=MODEL_VERSION)
        ensemble = load_ensemble_model('HTS', protein, MODEL_VERSION, num_models=5)

       # experiment_to_preds['HTS'][protein] = model.predict(seqs_one_hot,batch_size=256, workers=8, use_multiprocessing=True, verbose=1)
        experiment_to_preds['HTS'][protein] = ensemble_predict(ensemble, seqs_one_hot)

    create_output(experiment_to_preds['HTS'], base_method='HTS', target_method='HTS', custom_name=MODEL_VERSION, test=True)



TARGET_EXPERIMENTS = ['CHS', 'GHTS']


K_SIZES = {'GHTS': [301, 101, 51], # Agg 1
            'CHS': [301, 151, 51]} # Agg 2


if TEST_OTHERS:
    for protein in TEST_PROTEINS:
        print(f'Testing {protein}')
        #model = load_protein_model(experiment='HTS', protein=protein, custom_name=MODEL_VERSION)
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

                    #longer_input_model = get_converted_model(original_model=model, n_nucleotides=k, params=params)
                    longer_input_ensemble = [get_converted_model(original_model=model, n_nucleotides=k, params=params) for model in ensemble]
                    print('\t\t',end='')
                    #intensities = longer_input_model.predict(seqs_one_hot,batch_size=256, workers=8, use_multiprocessing=True, verbose=1)
                    intensities = ensemble_predict(longer_input_ensemble, seqs_one_hot)
                    save_partial_prediction("HTS", experiment, intensities, MODEL_VERSION, k, protein,False)
        print('')



if TEST_OTHERS:
    for experiment in TARGET_EXPERIMENTS:
        experiment_to_preds = {}
        for protein in TEST_PROTEINS:
            partial_preds = []
            for k in K_SIZES[experiment]:
                partial_preds.append(load_partial_predictions("HTS", experiment, MODEL_VERSION, k, protein,False))

            # average the predictions
            intensities = np.mean(partial_preds, axis=0)
            experiment_to_preds[protein] = intensities
        
        create_output(experiment_to_preds=experiment_to_preds, base_method='HTS', target_method=experiment, custom_name=MODEL_VERSION + '_' + AGG_ID, test=True)