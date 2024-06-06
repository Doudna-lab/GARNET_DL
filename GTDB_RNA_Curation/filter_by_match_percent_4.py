## This script takes in an Stockholm alignment file and outputs which sequences
# align to fewer than p_match % match states in the model.
# YS 2023.11.08, updated 2024.02.13

import string
import argparse

def argument_parsing():
    parser = argparse.ArgumentParser(description="This script takes in an Stockholm alignment file and outputs a list of sequences that align to fewer than p_match % match states in the consensus model.")
    parser.add_argument('alignment_file', help='input alignment file in Stockholm format')
    parser.add_argument('p_match', type=float, help='minimum % match states')
    parser.add_argument('output_file', help='name of output text file')
    return parser.parse_args()

args = argument_parsing()
alignment_file = args.alignment_file
p_match = args.p_match
output_file = args.output_file

match_counts = dict()

with open(alignment_file , 'r') as af:
    #
    line = af.readline()
    while line:
        # skip all lines that start with #
        if line[0] == '#' or len(line) < 20:
            line = af.readline()
            continue
        #
        genome_id = line.split()[0]
        sequence = line.split()[1].rstrip()
        #
        # remove gaps due to insertions in other sequences (represented by .)
        seq1 = sequence.replace('.', '')
        #
        # remove gaps due to insertions in this sequence (represented by lowercase)
        seq2 = seq1.translate(str.maketrans('', '', string.ascii_lowercase))
        #
        # only keep match states
        seq3 = seq2.replace('-', '')
        #
        # now count everything
        model_length = len(seq2)
        n_match = len(seq3)
        n_delete = model_length - n_match
        n_insert = len(seq1) - len(seq2)
        #
        # add to dict
        try:
            counts = match_counts[genome_id]
            c_model_length, c_n_match, c_n_delete, c_n_insert = counts
            match_counts[genome_id] = [c_model_length + model_length, c_n_match + n_match, c_n_delete + n_delete, c_n_insert + n_insert]
        except KeyError:
            match_counts[genome_id] = [model_length, n_match, n_delete, n_insert]
        line = af.readline()

with open(output_file, 'w') as cf:
    for genome_id in match_counts.keys():
        counts = match_counts[genome_id]
        model_length, n_match, n_delete, n_insert = counts
        fraction_match = n_match / model_length
        #
        # if too low % match states, write names to file
        if fraction_match < p_match:
            cf.write(f"{genome_id}\n")
