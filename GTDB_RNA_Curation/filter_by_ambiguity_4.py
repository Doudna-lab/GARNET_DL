## This script takes in an Stockholm alignment file and outputs which sequences
# contain more than p_amb % of ambiguity characters
# YS 2023.11.08, updated 2024.02.13

import string
import argparse

def argument_parsing():
    parser = argparse.ArgumentParser(description="This script takes in an Stockholm alignment file and outputs a list of sequences with >p_amb % of ambiguity characters")
    parser.add_argument('alignment_file', help='input alignment file in Stockholm format')
    parser.add_argument('p_amb', type=float, help='maximum % ambiguity characters')
    parser.add_argument('output_file', help='name of output text file')
    return parser.parse_args()

args = argument_parsing()
alignment_file = args.alignment_file
p_amb = args.p_amb
output_file = args.output_file

amb_counts = dict()

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
        # remove deletions
        seq2 = seq1.replace('-', '')
        #
        # count non-ambiguity characters
        n_acug = seq2.count('A') + seq2.count('a') + seq2.count('G') + seq2.count('g') + seq2.count('C') + seq2.count('c') + seq2.count('U') + seq2.count('u')
        #
        # now count ambiguity characters
        n_ns = len(seq2) - n_acug
        #
        # add to dict
        try:
            counts = amb_counts[genome_id]
            c_n_ns, c_len = counts
            amb_counts[genome_id] = [c_n_ns + n_ns, c_len + len(seq2)]
        except KeyError:
            amb_counts[genome_id] = [n_ns, len(seq2)]
        #
        line = af.readline()

with open(output_file, 'w') as cf:
    for genome_id in amb_counts.keys():
        counts = amb_counts[genome_id]
        n_ns, seq_len = counts
        fraction_amb = n_ns / seq_len
        #
        # if too low % match states, write names to file
        if fraction_amb > p_amb:
            cf.write(f"{genome_id}\n")
