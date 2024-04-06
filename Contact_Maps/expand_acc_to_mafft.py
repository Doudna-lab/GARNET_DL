import os
from Bio import SeqIO
import pandas as pd
import numpy as np

# Get the full path of the current script
script_path = os.path.realpath(__file__)

# Extract the directory part of the path
script_dir = os.path.dirname(script_path)

# Set csvDir to the script's directory
csv_dir = script_dir

# Construct the FASTA file path
fasta_file = os.path.join(csv_dir, 'pdb_sequences_structure_projected.fa')

# List all CSV files in the specified directory
csv_files = [f for f in os.listdir(csv_dir) if f.startswith('hs_raw_map_') and f.endswith('.csv')]

# Read the FASTA file
fasta_data = list(SeqIO.parse(fasta_file, 'fasta'))

# Iterate through the CSV files
for csv_file_name in csv_files:
    # Extract the identifier (XXXX) from the CSV file name
    identifier = csv_file_name.split('hs_raw_map_')[1].split('.csv')[0].lower()

    # Find the matching sequence in the FASTA data
    matching_seq = ''
    for record in fasta_data:
        fasta_header = record.id.lower()
        if fasta_header == identifier:
            matching_seq = str(record.seq)
            break

    # Check if a matching sequence was found
    if not matching_seq:
        print(f'No matching sequence found for {identifier} in FASTA file.')
        continue  # Skip to the next CSV file

    # Read the CSV file as a DataFrame (assuming no header)
    curmap = pd.read_csv(os.path.join(csv_dir, csv_file_name), header=None)

    # Initialize the modified matrix
    B = curmap.values

    # Iterate through the sequence and add zeros for gaps
    for ii, sym in enumerate(matching_seq, start=1):
        if sym == '-' or sym == '.':
            # Add a row and a column of zeros
            B = np.insert(B, ii - 1, 0, axis=0)
            B = np.insert(B, ii - 1, 0, axis=1)

    # Create the output CSV file name
    output_file_name = os.path.join(csv_dir, f'{identifier}_mod_iter1.csv')

    # Write the modified matrix to the output CSV file
    pd.DataFrame(B).to_csv(output_file_name, header=None, index=False)