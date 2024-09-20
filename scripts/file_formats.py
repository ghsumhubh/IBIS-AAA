import pandas as pd
from scripts.sequence_processing import one_hot_encode_dna
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
import gzip



def geneomic_test_to_csv(filepath, save_filepath):
    data = []
    identifier = ''
    sequence = ''

    file = open(filepath, 'r')
    
    for line in file:
        line = line.strip()  # Remove any trailing whitespace characters
        if line.startswith('>'):
            # Save the previous sequence before starting a new one (if not the first entry)
            if identifier:
                data.append((identifier, sequence))
                sequence = ''  # Reset the sequence
            
            # Extract the identifier from the line
            identifier = line.split()[0][1:]  # Remove the '>' and take the first part as identifier
        else:
            # This is a sequence line; append it to the current sequence string
            sequence += line
    
    # Don't forget to save the last sequence
    if identifier and sequence:
        data.append((identifier, sequence))

    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(data, columns=['Identifier', 'Sequence'])

    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv(save_filepath, index=False)

def hts_fastq_to_csv(fastq_gz_path, csv_path):
    # Initialize a list to store the sequences
    sequences = []

    # Open the gzipped FASTQ file
    with gzip.open(fastq_gz_path, 'rt') as file:  # 'rt' mode for reading as text
        while True:
            # Read lines in blocks of four (FASTQ format)
            header = file.readline().strip()  # Read the header
            if not header:
                break  # Stop if we have reached the end of the file
            sequence = file.readline().strip()  # Sequence line
            file.readline()  # Plus line (ignore)
            file.readline()  # Quality line (ignore)

            # Append the sequence to our list
            sequences.append(sequence)

    # Convert the list to a DataFrame
    df = pd.DataFrame(sequences, columns=['Sequence'])

    # Save to CSV
    df.to_csv(csv_path, index=False)



def fasta_to_csv(fasta_file, csv_file):
    """Convert a FASTA file to a CSV file with columns 'Identifier' and 'Sequence'."""
    with open(fasta_file, 'r') as file:
        sequences = []
        identifiers = []
        current_seq = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:  # Save the previous sequence
                    sequences.append(current_seq)
                identifiers.append(line[1:])  # Remove the '>' and save the identifier
                current_seq = ''  # Reset the sequence
            else:
                current_seq += line  # Continue building the sequence
                
        if current_seq:  # Save the last sequence
            sequences.append(current_seq)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Identifier': identifiers,
        'Sequence': sequences
    })
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)



def pbm_fasta_to_csv(filepath, save_filepath):
    # Initialize lists to store the data
    identifiers = []
    spot_ids = []
    rows = []
    cols = []
    linkers = []
    sequences = []
    
    # Read the FASTA file
    with open(filepath, 'r') as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                # Split the header to extract metadata
                header, metadata = line[1:].split(' ', 1)
                identifiers.append(header)
                metadata_parts = metadata.split(';')
                spot_id = metadata_parts[0].split('=')[1]
                row = metadata_parts[1].split('=')[1]
                col = metadata_parts[2].split('=')[1]
                linker = metadata_parts[3].split('=')[1]
                
                spot_ids.append(spot_id)
                rows.append(row)
                cols.append(col)
                linkers.append(linker)
            else:
                sequence += line
        # Add the last sequence
        if sequence:
            sequences.append(sequence)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Identifier': identifiers,
        'Spot_ID': spot_ids,
        'Row': rows,
        'Column': cols,
        'Linker': linkers,
        'Sequence': sequences
    })

    # remove the first 25 bases from the sequence
    df['Sequence'] = df['Sequence'].str[25:]

    # only keep Identifiers, and Sequences
    df = df[['Identifier', 'Sequence']]

    # Save the DataFrame to a CSV file
    df.to_csv(save_filepath, index=False)


def pbm_to_csv(filepath, save_filepath, already_normalized =False):
    df = pd.read_csv(filepath, sep='\t')
    # keep only mean_signal_intensity	mean_background_intensity and pbm_sequence
    df = df[['mean_signal_intensity', 'mean_background_intensity', 'pbm_sequence']]
    # rename pbm_sequence to Sequence
    df = df.rename(columns={'pbm_sequence': 'Sequence'})

    if already_normalized:
        df['Intensity'] = df['mean_signal_intensity']
    else:
        #df['Intensity'] = df['mean_signal_intensity'] / df['mean_background_intensity']
        df['Intensity'] =  df['mean_background_intensity']


    # normalize using power transformer
    #pt = PowerTransformer()
    #df['Intensity'] = pt.fit_transform(df['Intensity'].values.reshape(-1, 1))

    # normalize using min max scaler to -1 to 1
    mm = MinMaxScaler()
    df['Intensity'] = mm.fit_transform(df['Intensity'].values.reshape(-1, 1))



    # keep only Sequence and Intensity
    df = df[['Sequence', 'Intensity']]

    # Save the DataFrame to a CSV file
    df.to_csv(save_filepath, index=False)


def pbm_train_to_pickle(filepath, save_filepath):
    df = pd.read_csv(filepath)
    sequences = df['Sequence'].values
    intensities = df['Intensity'].values


    sequences_one_hot = [one_hot_encode_dna(seq) for seq in sequences]

    # make to numpy array
    sequences_one_hot = np.array(sequences_one_hot)
    intensities = np.array(intensities)



    with open(save_filepath, 'wb') as f:
        pickle.dump((sequences_one_hot, intensities), f)

def pbm_train_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        sequences_one_hot, intensities = pickle.load(f)
    return sequences_one_hot, intensities


def pbm_test_to_pickle(filepath, save_filepath):
    df = pd.read_csv(filepath)
    sequences = df['Sequence'].values

    sequences_one_hot = [one_hot_encode_dna(seq) for seq in sequences]

    identifiers = df['Identifier'].values

    # make to numpy array
    sequences_one_hot = np.array(sequences_one_hot)

    with open(save_filepath, 'wb') as f:
        pickle.dump((sequences_one_hot, identifiers), f)

def pbm_test_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        sequences_one_hot, identifiers = pickle.load(f)
    return sequences_one_hot, identifiers



    


def hts_csv_to_kmer_count(path, k):
    df = pd.read_csv(path)
    # Get the sequence column
    sequences = df['Sequence'].values

    # remove all sequeces that contain the letter N
    sequences = [seq for seq in sequences if 'N' not in seq]
    
    counts = {}
    
    # Process each sequence
    for seq in sequences:
        for i in range(len(seq) - k + 1):  
            mer = seq[i:i+k]
            if mer in counts:
                counts[mer] += 1
            else:
                counts[mer] = 1
    
    return counts


def hts_csv_to_kmer_count_csv(paths, csv_path, k=35):
    # Initialize a dictionary to store the counts of each 35mer
    total_counts = {}
    
    for path in paths:
        counts = hts_csv_to_kmer_count(path,k)
        # Combine the counts
        for mer, count in counts.items():
            if mer in total_counts:
                total_counts[mer] += count
            else:
                total_counts[mer] = count

        
    # Create a DataFrame from the counts
    df = pd.DataFrame(list(total_counts.items()), columns=['Sequence', 'Count'])

    # Sort the DataFrame by count in descending order
    df = df.sort_values('Count', ascending=False)


    # add a normalized count column to the range of 1 to 100
    df['Normalized_Count'] = df['Count'] / df['Count'].max() * 100

    # save to csv
    df.to_csv(csv_path, index=False)



