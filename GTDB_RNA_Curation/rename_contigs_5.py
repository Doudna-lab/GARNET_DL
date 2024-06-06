## This script takes in a FASTA sequence file and renames the sequences to include the GTDB 
# genome ID and Rfam ID as such: <genome_ID>__<contig location>__<Rfam ID>
# YS 2024.02.14

import string
import argparse

def argument_parsing():
    parser = argparse.ArgumentParser(description="This script takes in a FASTA file and renames sequences to include GTDB genome ID.")
    parser.add_argument('fasta_file', help='input FASTA file')
    parser.add_argument('output_file', help='output FASTA file')
    parser.add_argument('rfam_id', help='ID of Rfam family')
    return parser.parse_args()

args = argument_parsing()
fasta_file = args.fasta_file
output_file = args.output_file
rfam_id = args.rfam_id
contig_to_GTDB_file = "gtdb_ids_to_contigs.txt"

# load in dictionary of associations
contig_to_GTDB = dict()
with open(contig_to_GTDB_file, 'r') as cf:
	for line in cf.readlines():
		contig, genome_id = line.rstrip().split(',')
		# some contigs accidentally have [''] surrounding, need to remove
		if contig[0] == '[':
			contig = contig.replace('[', '')
			contig = contig.replace(']', '')
			contig = contig.replace('\'', '')
		contig_to_GTDB[contig] = genome_id

# now go through and process these files
with open(fasta_file, 'r') as ff, open(output_file, 'w') as of:
	for line in ff.readlines():
		if line[0] == '>':
			# process name
			original_name = line.rstrip().split()[0][1:]
			name_contig = original_name.split('/')[0]
			genome_id = contig_to_GTDB[name_contig]
			dum = of.write('>%s__%s__%s\n' % (genome_id, original_name, rfam_id))
		else:
			dum = of.write(line)
