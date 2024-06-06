#!/bin/bash
#SBATCH -p [partition name]
#SBATCH --job-name filtering_23S
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
cat archaea_RF02540.cm_info.csv bacteria_RF02541.cm_info.csv > combined_23S_info.csv
cat archaea_RF02540.cm.fa bacteria_RF02541.cm.fa > combined_23S.fa

# fetching the CM from Rfam
${infernal_dir}cmfetch $rfam_file RF02541 > RF02541.cm

# aligning both bacterial and archaeal sequences to the bacterial Rfam model
${infernal_dir}cmalign -o combined_23S.sto RF02541.cm combined_23S.fa


## FILTERING OUT LOW QUALITY SEQUENCES
# filtering sequences based on % aligned match states
python filter_by_match_percent_4.py combined_23S.sto 0.85 truncation_filter_accessions_to_remove_23S.txt

# filtering sequences based on % ambiguity characters
python filter_by_ambiguity_4.py combined_23S.sto 0.05 ambiguity_filter_accessions_to_remove_23S.txt

# filtering out sequences that are >1 stddev longer than the mean. 
python filter_by_length_4.py combined_23S.sto 1 length_filter_accessions_to_remove_23S.txt

# filtering out sequences that have >2 stddev more % non-WC basepairs than the mean (this is 
# mean to filter out pseudogenes). 
python filter_by_secondary_structure.py combined_23S.sto 2 mispairs_filter_accessions_to_remove_23S.txt

# combine all accessions that should be removed
cat ambiguity_filter_accessions_to_remove_23S.txt truncation_filter_accessions_to_remove_23S.txt \
     length_filter_accessions_to_remove_23S.txt mispairs_filter_accessions_to_remove_23S.txt | \
     sort | uniq > shared_accessions_to_remove_23S.txt

# removing from alignment
${infernal_dir}esl-alimanip --seq-r shared_accessions_to_remove_23S.txt combined_23S.sto > \
     combined_23S_filtered.sto


## FINAL LM ALIGNMENTS
# making new fasta file
${infernal_dir}esl-reformat fasta combined_23S_filtered.sto > final_23S_LM.fa

# redoing alignment to remove all gap columns
${infernal_dir}cmalign -o final_23S_LM.sto RF02541.cm final_23S_LM.fa
${infernal_dir}esl-reformat afa final_23S_LM.sto > final_23S_LM.afa
cp final_23S_LM.* ../final_alignments/.


## TRIMMING INSERTIONS FOR GNN ALIGNMENTS
# trimming columns that are insertions relative to Rfam model
${infernal_dir}esl-alimask --rf-is-mask final_23S_LM.sto > \
     combined_23S_filtered_match_only.sto

# creating a mask file which indicates columns that are gaps in E coli PDB 7K00 sequence 
# relative to Rfam
${infernal_dir}cmalign RF02541.cm 7K00_23S.fa > 7K00_23S.sto
${infernal_dir}esl-reformat fasta 7K00_23S.sto > 7K00_23S_insertions_lowercase.fa
${infernal_dir}esl-alimask --rf-is-mask 7K00_23S.sto > 7K00_23S_match_only.sto
grep -v '^#' 7K00_23S_match_only.sto  | awk '{print $2}' | tr -d '\n' | sed 's|-|0|g' | \
     sed 's|[A-Z]|1|g' > 7K00_mask.txt


## FINAL GNN ALIGNMENTS
# trimming columns that are insertions relative to reference structure
${infernal_dir}esl-alimask combined_23S_filtered_match_only.sto 7K00_mask.txt > final_23S_GNN.sto

# save final files for GNN approach
${infernal_dir}esl-reformat fasta final_23S_GNN.sto > final_23S_GNN.fa
${infernal_dir}esl-reformat afa final_23S_GNN.sto > final_23S_GNN.afa
