## This script takes in an Stockholm alignment file and compares each sequences against
# the consensus secondary structure in the SS_cons line. For each sequences, calculates 
# the amount of secondary structure that is non-AU/GC/GU pairs. Outputs a list of 
# sequences that fall >stddev outside the distribution
# YS 2023.02.14

import string
import argparse
import numpy as np

def argument_parsing():
    parser = argparse.ArgumentParser(description="This script takes in an Stockholm alignment file and compares each sequences against the consensus secondary structure in the SS_cons line. For each sequences, calculates the amount of secondary structure that is non-AU/GC/GU pairs. Outputs a list of sequences that fall >stddev outside the distribution.")
    parser.add_argument('alignment_file', help='input alignment file in Stockholm format')
    parser.add_argument('std_dev', help='how many standard deviation above mean')
    parser.add_argument('output_file', help='name of output text file')
    return parser.parse_args()

args = argument_parsing()
alignment_file = args.alignment_file
output_file = args.output_file
std_dev = args.std_dev

alignments = dict()
ss_line = []

with open(alignment_file, 'r') as af:
    for line in af.readlines():
        if len(line)<5:
            continue
        if line[0] != '#':
            info = line.rstrip().split()
            name = info[0]
            sequence = list(info[1].upper())
            try:
                prev_lines = alignments[name]
                alignments[name] = prev_lines + sequence
            except KeyError:
                alignments[name] = sequence
        if '#=GC SS_cons' in line:
            ss_line += line.rstrip().split()[2]

ss = list(ss_line)
names = np.array(list(alignments.keys()))
alignment = np.array(list(alignments.values()))

base_pairs = list()
ss_stack = list()
for i, ss_char in enumerate(ss):
    if ss_char == '{' or ss_char == '<' or ss_char == '[' or ss_char == '(':
        ss_stack.append(alignment[:,i])
    if ss_char == '}' or ss_char == '>' or ss_char == ']' or ss_char == ')':
        left_half = ss_stack.pop()
        right_half = alignment[:,i]
        base_pairs.append(np.array([left_half, right_half]))


wc_bp = np.zeros(len(names))
nonwc_bp = np.zeros(len(names))
for m, basepair in enumerate(base_pairs):
    for n in range(len(basepair[0])):
        bp = set(basepair[:,n])
        if bp == {'G', 'U'} or bp == {'G', 'C'} or bp == {'A', 'U'}:
            wc_bp[n] += 1
        elif bp == {'G', 'A'} or bp == {'C', 'A'} or bp == {'C', 'U'} or bp == {'G'} or bp == {'A'} or bp == {'C'} or bp == {'U'}:
            nonwc_bp[n] += 1

frac_nc_list = list()
for i, name in enumerate(names):
    if (wc_bp[i]+nonwc_bp[i]) > 0:
        frac_nc_list.append(nonwc_bp[i]/(wc_bp[i]+nonwc_bp[i]))
    else:
        frac_nc_list.append(0)

mean_list = np.mean(frac_nc_list)
stddev_list = np.std(frac_nc_list)

with open(output_file, 'w') as cf:
    for i, name in enumerate(names):
        if frac_nc_list[i] > (mean_list + std_dev*stddev_list):
            dum = cf.write(f"{name}\n")


