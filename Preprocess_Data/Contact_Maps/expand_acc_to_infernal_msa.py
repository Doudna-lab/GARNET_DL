import os
from Bio import SeqIO
import pandas as pd
import numpy as np
import re

# Set the directory to the current working directory
csv_dir = os.getcwd()
print(f"Current working directory: {csv_dir}")

# Construct the FASTA file path
fasta_file = os.path.join(csv_dir, 'pdb_sequences_infernal_msa.afa')
print(f"FASTA file path: {fasta_file}")

# Check if the FASTA file exists
if not os.path.exists(fasta_file):
    print(f"FASTA file not found at {fasta_file}")
else:
    print("FASTA file found, proceeding...")

    # List all CSV files in the specified directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_mod_iter1.csv')]
    print(f"Found {len(csv_files)} CSV files to process.")

    # Read the FASTA file
    fasta_data = list(SeqIO.parse(fasta_file, 'fasta'))

    # Iterate through the CSV files
    for csv_file_name in csv_files:
        print(f"Processing {csv_file_name}...")

        # Extract the identifier (XXXX) from the CSV file name
        match = re.search(r'(\w+)_mod_iter1\.csv', csv_file_name)
        if match:
            identifier = match.group(1).lower()
            print(f"Identifier extracted: {identifier}")

            # Find the matching sequence in the FASTA data
            matching_seq = ''
            for record in fasta_data:
                fasta_header = record.id.lower()
                if fasta_header == identifier:
                    matching_seq = str(record.seq)
                    break

            if not matching_seq:
                print(f"No matching sequence found for {identifier} in FASTA file.")
            else:
                print(f"Matching sequence found for {identifier}. Modifying matrix...")

                # Read the CSV file as a DataFrame (assuming no header)
                curmap = pd.read_csv(os.path.join(csv_dir, csv_file_name), header=None)

                # Initialize the modified matrix
                B = curmap.values

                # Iterate through the sequence and add zeros for gaps
                for ii, sym in enumerate(matching_seq, start=1):
                    if sym == '-' or sym == '.':
                        B = np.insert(B, ii - 1, 0, axis=0)
                        B = np.insert(B, ii - 1, 0, axis=1)

                # Create the output CSV file name
                output_file_name = os.path.join(csv_dir, f'{identifier}_mod_iter2.csv')

                # Write the modified matrix to the output CSV file
                pd.DataFrame(B).to_csv(output_file_name, header=None, index=False)
                print(f"Modified matrix written to {output_file_name}")
        else:
            print(f"Could not extract identifier from {csv_file_name}")