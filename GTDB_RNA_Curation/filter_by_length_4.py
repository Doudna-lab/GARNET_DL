## This script takes in an Stockholm alignment file and outputs which sequences
# have a raw length that is >n standard deviations above the mean for the alignment.
# YS 2024.02.14

import string
import argparse
import numpy as np
import math

def argument_parsing():
    parser = argparse.ArgumentParser(description="This script takes in an Stockholm alignment file and outputs a list of sequences have a raw length that is >n standard deviations above the mean for the alignment.")
    parser.add_argument('alignment_file', help='input alignment file in Stockholm format')
    parser.add_argument('std_dev', help='how many standard deviation above mean')
    parser.add_argument('output_file', help='name of output text file')
    return parser.parse_args()

args = argument_parsing()
alignment_file = args.alignment_file
output_file = args.output_file
std_dev = args.std_dev

length_dict = dict()

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
        # add to dict
        try:
            s_len = length_dict[genome_id]
            length_dict[genome_id] = s_len + len(seq2)
        except KeyError:
            length_dict[genome_id] = len(seq2)
        #
        line = af.readline()

length_list = list(length_dict.values())
mean_list = math.floor(np.mean(length_list))
stddev_list = math.floor(np.std(length_list))

with open(output_file, 'w') as cf:
    for genome_id in length_dict.keys():
        s_len = length_dict[genome_id]
        #
        # if too long, write names to file
        if s_len > (mean_list + std_dev*stddev_list):
            cf.write(f"{genome_id}\n")
