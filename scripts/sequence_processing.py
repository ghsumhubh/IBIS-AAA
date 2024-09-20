import numpy as np
import pickle
import pandas as pd
import math
import gc



def one_hot_encode_dna(dna_sequence, pad_to_length=None):
    # Mapping of DNA bases to their respective indices
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': None}
    
    # Create an empty matrix with dimensions (length of sequence) x (number of bases)
    one_hot_encoded = np.zeros((len(dna_sequence), 4), dtype=float)
    
    # Fill the matrix: set 1 at the respective base index for each base in the sequence
    for i, base in enumerate(dna_sequence):
        if base in base_to_index:
            if base == 'N':  # Special case for 'N'
                one_hot_encoded[i] = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                one_hot_encoded[i, base_to_index[base]] = 1
        else:
            raise ValueError(f"Invalid base '{base}' found in DNA sequence.")
        
    if pad_to_length:
        # If the sequence is shorter than the padding length, add zeros at the end
        if len(dna_sequence) < pad_to_length:
            padding_length = pad_to_length - len(dna_sequence)
            padding_array = np.zeros((padding_length, 4))
            one_hot_encoded = np.vstack([one_hot_encoded, padding_array])

    # make into boolean TODO: make sure works
    one_hot_encoded = one_hot_encoded.astype(np.bool_)
    
    return one_hot_encoded


def generate_random_dna_sequence(length):
    # Generate a random DNA sequence of the specified length
    dna_bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(dna_bases, length))

def shuffle_sequence(seq):
    return ''.join(np.random.permutation(list(seq)))


# def reverse_one_hot_encode_dna(one_hot_encoded_sequence):
#     # Mapping of indices to DNA bases
#     index_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
#     # Create an empty list to store the decoded sequence
#     decoded_sequence = []
    
#     # Decode the one-hot encoded sequence
#     for base in one_hot_encoded_sequence:
#         # Find the index of the base with value 1
#         index = np.argmax(base)
#         decoded_sequence.append(index_to_base[index])
    
    return ''.join(decoded_sequence)

def save_pickle_seq_only(sequences, labels, filename):
    one_hot_encoded = [one_hot_encode_dna(seq) for seq in sequences]

    with open(filename, 'wb') as f:
        pickle.dump((one_hot_encoded, labels), f)

def check_if_pickle_exists(filename):
    try:
        with open(filename, 'rb') as f:
            pickle.load(f)
        return True
    except FileNotFoundError:
        return False

def load_pickle_seq_only(filename):
    with open(filename, 'rb') as f:
        one_hot_encoded, labels = pickle.load(f)
    return one_hot_encoded, labels


def get_kmer_count_dict(path):
    df = pd.read_csv(path)

    # make it into a dictionary where the key is 'Sequence' and 'Count' is the value
    kmer_count_dict = dict(zip(df['Sequence'], df['Normalized_Count']))

    return kmer_count_dict

def get_scores(sequence, k_numbers, kmer_dicts):
    scores = []
    for k in k_numbers:
        score = 0
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer in kmer_dicts[k]:
                score = max(score, kmer_dicts[k][kmer])
        scores.append(score)
    return scores

def get_all_scores(sequences, k_numbers, kmer_dicts):
    all_scores = []
    for seq in sequences:
        all_scores.append(get_scores(seq, k_numbers, kmer_dicts))
    return np.array(all_scores)

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join([complement[base] for base in seq[::-1]])
    

def load_test_sequences_onehot(experiment, reverse_compliment=False, final_test=False):
    if final_test:
        df = pd.read_csv("data/Test/test_sequences.csv")
    else:
        df = pd.read_csv("data/test_sequences.csv")
    df = df[df['experiment'] == experiment]
    seqs = df['Sequence'].values
    if reverse_compliment:
        seqs = [reverse_complement(seq) for seq in seqs]


    seqs_one_hot = [one_hot_encode_dna(seq) for seq in seqs]
    seqs_one_hot = np.array(seqs_one_hot)
    seqs_one_hot = np.array(seqs_one_hot).astype(np.bool_)

    del df
    del seqs
    gc.collect()

    return seqs_one_hot

def trim_sequences(seqs_one_hot, k):
    """Trim the sequences to a width of k centered around the middle."""
    seq_len = seqs_one_hot.shape[1]
    start_index = (seq_len - k) // 2
    if k != seq_len:
        seqs_one_hot = seqs_one_hot[:, start_index:start_index + k, :]
    return seqs_one_hot



