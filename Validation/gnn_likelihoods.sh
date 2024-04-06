#!/bin/bash
#SBATCH -p cpu-c6i-16xlarge
#SBATCH --job-name gnn_likelihoods
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --exclusive
#SBATCH --nodes=1

## ACTIVATE CONDA
eval "$(conda shell.bash hook)"
conda deactivate
conda activate gnn_rna

infernal_dir="/home/ubuntu/anaconda3/bin/"
ecoli_mask="/home/ubuntu/datasets/alignments/23S_rRNA/filtering/7K00_mask.txt"
rfam_model="/home/ubuntu/datasets/alignments/23S_rRNA/filtering/RF02541.cm"

file_prefix=$1
#file_prefix="Ecoli_23S_7K00_mutations_Fig6"

# aligning to the bacterial Rfam model
${infernal_dir}cmalign -o ${file_prefix}.sto $rfam_model ${file_prefix}.fasta

# trimming columns that are insertions relative to Rfam model
${infernal_dir}esl-alimask --rf-is-mask ${file_prefix}.sto > ${file_prefix}_match_only.sto

# trimming columns that are insertions relative to reference structure (7k00)
${infernal_dir}esl-alimask ${file_prefix}_match_only.sto $ecoli_mask > ${file_prefix}_trimmed_to_GNN.sto

# re-format to aligned fasta
${infernal_dir}esl-reformat afa ${file_prefix}_trimmed_to_GNN.sto > ${file_prefix}_trimmed_to_GNN.afa

python /home/ubuntu/software/gnn_log_liks/structure-based-rna-model/bin/score_likelihoods.py \
        -i ${file_prefix}_trimmed_to_GNN.afa \
        -o ${file_prefix}_log_liks_ft.csv \
        --device cpu \
        --model_path /home/ubuntu/software/gnn_log_liks/inputs/finetuned.pt \
        --distance_map /home/ubuntu/software/gnn_log_liks/inputs/distance_map.npy

python /home/ubuntu/software/gnn_log_liks/structure-based-rna-model/bin/score_likelihoods_without_mutation.py
        -i ${file_prefix}_trimmed_to_GNN.afa \
        -o ${file_prefix}_log_liks_ft_mutation_masked.csv \
        --mut_idx $mutation_index
	--device cpu \
        --model_path /home/ubuntu/software/gnn_log_liks/inputs/finetuned.pt \
        --distance_map /home/ubuntu/software/gnn_log_liks/inputs/distance_map.npy
