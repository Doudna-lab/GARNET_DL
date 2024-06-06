#!/bin/bash
#SBATCH -p [partition name]
#SBATCH --job-name realigning_miniRfam
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --exclusive
#SBATCH --nodes=6

## ACTIVATE CONDA
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bioinfo_tools_Py_310

infernal_dir="/home/ubuntu/anaconda3/bin/"

# iterate through each Rfam ID
while read rfam_id; do
	echo "$rfam_id"
	#
	# convert to fasta
	${infernal_dir}esl-reformat fasta ${rfam_id}/${rfam_id}_4.sto > ${rfam_id}/${rfam_id}_4.fa
	#
	# rename sequences to <GTDB genome ID>__<former info>__<Rfam ID>
	python rename_contigs_5.py ${rfam_id}/${rfam_id}_4.fa ${rfam_id}/${rfam_id}_5.fa ${rfam_id}
	#
	# realign using original CM
	${infernal_dir}cmalign -o $rfam_id/${rfam_id}_5.sto $rfam_id/$rfam_id.cm ${rfam_id}/${rfam_id}_5.fa
done < rfam_ids.txt


