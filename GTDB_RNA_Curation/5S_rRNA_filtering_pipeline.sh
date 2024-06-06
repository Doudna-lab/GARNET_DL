#!/bin/bash
#SBATCH -p [partition name]
#SBATCH --job-name filtering_5S
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --exclusive
#SBATCH --nodes=8

## ACTIVATE CONDA
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bioinfo_tools_Py_310

infernal_dir="/home/ubuntu/anaconda3/bin/"
rfam_file="/home/ubuntu/datasets/rfam/Rfam.cm"

## MAKING COMBINED ALIGNMENT
# combining archaeal and bacterial sequences into a single FASTA file. The info files are 
# outputs from the cmsearch Python script
cat archaea_RF00001.cm_info.csv bacteria_RF00001.cm_info.csv > combined_5S_info.csv
cat archaea_RF00001.cm.fa bacteria_RF00001.cm.fa > combined_5S.fa

# fetching the CM from Rfam
${infernal_dir}cmfetch $rfam_file RF00001 > RF00001.cm

# aligning both bacterial and archaeal sequences to the bacterial Rfam model
${infernal_dir}cmalign -o combined_5S.sto RF00001.cm combined_5S.fa


## FILTERING OUT LOW QUALITY SEQUENCES
# filtering sequences based on % aligned match states
python filter_by_match_percent_4.py combined_5S.sto 0.85 truncation_filter_accessions_to_remove_5S.txt

# filtering sequences based on % ambiguity characters
python filter_by_ambiguity_4.py combined_5S.sto 0.05 ambiguity_filter_accessions_to_remove_5S.txt

# filtering out sequences that are >2 stddev longer than the mean. 
python filter_by_length_4.py combined_5S.sto 2 length_filter_accessions_to_remove_5S.txt

# filtering out sequences that have >2 stddev more % non-WC basepairs than the mean (this is 
# mean to filter out pseudogenes). 
python filter_by_secondary_structure.py combined_5S.sto 2 mispairs_filter_accessions_to_remove_5S.txt


# combine all accessions that should be removed
cat ambiguity_filter_accessions_to_remove_5S.txt truncation_filter_accessions_to_remove_5S.txt \
     length_filter_accessions_to_remove_5S.txt mispairs_filter_accessions_to_remove_5S.txt | sort | uniq > \
     shared_accessions_to_remove_5S.txt

# removing from alignment
${infernal_dir}esl-alimanip --seq-r shared_accessions_to_remove_5S.txt combined_5S.sto > \
     combined_5S_filtered.sto


## FINAL LM ALIGNMENTS
# making new fasta file
${infernal_dir}esl-reformat fasta combined_5S_filtered.sto > final_5S_LM.fa

# redoing alignment to remove all gap columns
${infernal_dir}cmalign -o final_5S_LM.sto RF00001.cm final_5S_LM.fa
${infernal_dir}esl-reformat afa final_5S_LM.sto > final_5S_LM.afa

