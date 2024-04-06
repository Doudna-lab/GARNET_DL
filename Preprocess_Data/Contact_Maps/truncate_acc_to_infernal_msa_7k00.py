import os
from Bio import SeqIO
import pandas as pd
import numpy as np
import re

# Set the directory to the current working directory
csv_dir = os.getcwd()
print(f"Working in directory: {csv_dir}")

# Construct the FASTA file path
fasta_file = os.path.join(csv_dir, 'pdb_sequences_infernal_msa.afa')
print(f"Reading FASTA file from: {fasta_file}")

# Read the FASTA file
fasta_data = list(SeqIO.parse(fasta_file, 'fasta'))
print(f"Read {len(fasta_data)} sequences from the FASTA file.")

# Identify positions to contract in "7K00" sequence
sequence_7K00 = ''
for record in fasta_data:
    if record.id.upper().startswith('7K00'):  # Case-insensitive comparison
        sequence_7K00 = str(record.seq)
        break

if sequence_7K00:
    print("Found sequence for '7K00'.")
else:
    print("Did not find sequence for '7K00'.")

to_truncate_7K00 = [i for i, char in enumerate(sequence_7K00) if char == '-']

# List all CSV files
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_mod_iter2.csv')]
print(f"Found {len(csv_files)} CSV files to process.")

# Iterate through CSV files
for csv_file_name in csv_files:
    print(f"\nProcessing {csv_file_name}...")
    fileNamePrefix = csv_file_name[:4].lower()  # Convert to lowercase for comparison

    # Read the CSV file as a DataFrame (excluding the header)
    curmap = pd.read_csv(os.path.join(csv_dir, csv_file_name), header=None)

    # Find corresponding sequence
    curseq = ''
    for record in fasta_data:
        if record.id.lower().startswith(fileNamePrefix):  # Case-insensitive comparison
            curseq = str(record.seq)
            break

    # Check if sequence is found
    if not curseq:
        print(f"No matching sequence found for {fileNamePrefix} in FASTA file.")
        continue  # Skip to the next file if no sequence is found
    else:
        print(f"Found matching sequence for {fileNamePrefix}.")

    # Truncate for lowercase letters or dot gaps in curseq
    to_truncate = [i for i, char in enumerate(curseq) if char.islower() or char == '.']

    # Combine positions to truncate from curseq and 7K00
    all_truncate_positions = sorted(set(to_truncate + to_truncate_7K00), reverse=True)

    # Truncate the matrix
    C = curmap.values
    for pos in all_truncate_positions:
        if pos < C.shape[0]:
            C = np.delete(C, pos, axis=0)  # Delete row
            C = np.delete(C, pos, axis=1)  # Delete column

    # Create the output CSV file name
    output_file_name = os.path.join(csv_dir, f'{fileNamePrefix}_aligned.csv')

    # Write the modified matrix to the output CSV file
    pd.DataFrame(C).to_csv(output_file_name, header=None, index=False)
    print(f"Truncated matrix written to {output_file_name}")

    # Remove residues from fasta sequence and update the record
    modified_seq = ''.join([char for i, char in enumerate(curseq) if i not in all_truncate_positions])
    for record in fasta_data:
        if record.id.lower().startswith(fileNamePrefix):
            record.seq = modified_seq