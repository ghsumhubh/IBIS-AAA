from scripts.file_formats import *
from scripts.models import *
from scripts.test_output import *
from scripts.sequence_processing import *
import numpy as np
import pandas as pd
import os
import gc
import warnings
warnings.filterwarnings('ignore', category=UserWarning) 



hts_path = 'data/HTS/'
PROTEINS = [f for f in os.listdir(hts_path) if os.path.isdir(os.path.join(hts_path, f))]
OG_PROTEINS = ['NFKB1','LEF1','NACC2','RORB','TIGD3']
FIRST_PROTEIN = ['NFKB1']
TEST_PROTEINS = list(set(PROTEINS) - set(OG_PROTEINS))

# converts the original .gz files to csv files
def preprocess_train_part_a():
    path = f"data/{'HTS'}/"
    # get all subdirectories
    subdirs = [f.path for f in os.scandir(path) if f.is_dir()]
    for subdir in subdirs:

        # get all the .gz files in the subdirectory
        files = [f.path for f in os.scandir(subdir) if f.is_file() and f.name.endswith('.gz')]

        r_to_len = {}
        index = 0
        while True:
            r_string = f'_R{index}_'
            # get number of files with this r_string
            r_files = [f for f in files if r_string in f]
            if len(r_files) == 0:
                break
            r_to_len[r_string] = len(r_files)
            index += 1
        
        # get files corresponding to the biggest r_to_len
        print(subdir.split("/")[-1])
        biggest_r = max(r_to_len, key=r_to_len.get)
        files = [f for f in files if biggest_r in f]

        print(f'R: {biggest_r}, files: {len(files)}')
            


        # for each file, extract the sequences and save them to a csv
        for index , file in enumerate(files):
            hts_fastq_to_csv(file, path + "/" + subdir.split("/")[-1] + f"/sequences_{index}.csv")

# Takes the csv files and concact them into one file, while adding labels
def preprocess_train_part_b():
    for protein in PROTEINS:
        print(f'Preprocessing {protein}')
        folder = f'data/HTS/{protein}'
        sequence_files = [f for f in os.listdir(folder) if f.startswith('sequences')]
        # renove all files that are not csv
        sequence_files = [f for f in sequence_files if f.endswith('.csv')]

        sequences_data = pd.DataFrame()
        # intensity is what file it came from + 1
        for i, file in enumerate(sequence_files):
            data = pd.read_csv(f'{folder}/{file}')
            data['intensity'] = i + 1
            sequences_data = pd.concat([sequences_data, data])

        # 50% of the data should be 0 to indicate random sequences
        for i in range(len(sequence_files)):
            data = pd.read_csv(f'{folder}/{sequence_files[i]}')
            data['Sequence'] = data['Sequence'].apply(shuffle_sequence)
            data['intensity'] = 0
            sequences_data = pd.concat([sequences_data, data])

        sequences_data = sequences_data.reset_index(drop=True)


        # make into one hot
        one_hot = [one_hot_encode_dna(seq) for seq in sequences_data['Sequence']]
        sequences_data['Sequence'] = one_hot
        
        
        # save to pickle
        sequences_data.to_pickle(f'{folder}/sequences_data.pkl')

# Add aliens and shuffled sequences
def preprocess_train_part_c():
    for protein in PROTEINS:
            print(f'Preprocessing {protein}')

            self_data = pd.read_pickle(f'data/HTS/{protein}/sequences_data.pkl')
                
            others_data = pd.concat([pd.read_pickle(f'data/HTS/{p}/sequences_data.pkl') for p in PROTEINS if p != protein])

            # remove the negative class in others_data
            others_data = others_data[others_data['intensity'] > 0]

            size = len(self_data)//2
            others_data = others_data.sample(n=size, random_state=42)

            # change the intensity of the othes to 0
            others_data['intensity'] = 0


            # combine the data
            data = pd.concat([self_data, others_data])

            # save to pickle
            data.to_pickle(f'data/HTS/{protein}/sequences_data_with_aliens.pkl')

# Remove duplicates within the same cycle
def preprocess_train_part_d():
    for protein in PROTEINS:
        print(f'Preprocessing {protein}')

        alien_data = pd.read_pickle(f'data/HTS/{protein}/sequences_data_with_aliens.pkl')

        sequences_one_hot, labels = alien_data['Sequence'].values, alien_data['intensity'].values

        labels = np.stack(labels)
        sequences_one_hot = np.stack(sequences_one_hot)
        # maximum cycle is the highest label
        maximum_cycle = max(labels)


        class_i_unique_seqs = []
        for i in range(maximum_cycle+1):
            indices_where_i = np.where(labels == i)
            unique_seqs = np.unique(sequences_one_hot[indices_where_i], axis=0)
            class_i_unique_seqs.append(unique_seqs)

        sequences_one_hot = np.concatenate(class_i_unique_seqs)
        labels = np.concatenate([np.full(class_i_unique_seqs[i].shape[0], i) for i in range(maximum_cycle+1)])

        # make sequences a normal array of 2d arrays
        sequences_one_hot = [np.array(seq) for seq in sequences_one_hot]

        alien_data = pd.DataFrame({'Sequence': sequences_one_hot, 'intensity': labels})

        alien_data.to_pickle(f'data/HTS/{protein}/sequences_data_with_aliens_no_duplicates.pkl')

# Converts the original .fasta files to csv files
def preprocess_test_part_a():
    # create folders
    os.makedirs('data/Test/GHTS', exist_ok=True)
    os.makedirs('data/Test/CHS', exist_ok=True)
    os.makedirs('data/Test/HTS', exist_ok=True)

    geneomic_test_to_csv('./data/Test/GHTS_participants.fasta', './data/Test/GHTS/sequences.csv')
    geneomic_test_to_csv('./data/Test/CHS_participants.fasta', './data/Test/CHS/sequences.csv')
    fasta_to_csv("data/Test/HTS_participants.fasta", "data/Test/HTS/sequences.csv")

# Creates the test_sequences.csv file
def preprocess_test_part_b():
    EXPERIMENTS = ['CHS', 'GHTS', 'HTS']
    dfs = []
    for experiment in EXPERIMENTS:
        df = pd.read_csv(f'data/Test/{experiment}/sequences.csv')
        df['experiment'] = experiment

        dfs.append(df)

    df = pd.concat(dfs)

    # rearrange columns so it is 
    df = df[['experiment', 'Identifier', 'Sequence']]

    # save to csv
    df.to_csv('data/Test/test_sequences.csv', index=False)

    #print(df.head())





def main():
    print('Preprocessing train data')
    print('Converting .gz files to csv files')
    preprocess_train_part_a()
    print('Combining csv files and adding labels')
    preprocess_train_part_b()
    print('Adding aliens and shuffled sequences')
    preprocess_train_part_c()
    print('Removing duplicates within the same cycle')
    preprocess_train_part_d()

    print('Preprocessing test data')
    print('Converting .fasta files to csv files')
    preprocess_test_part_a()
    print('Creating test_sequences.csv')
    preprocess_test_part_b()
    print('Preprocessing done')

    

if __name__ == '__main__':
    main()


