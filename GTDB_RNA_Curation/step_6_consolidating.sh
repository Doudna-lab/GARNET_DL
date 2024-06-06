#!/bin/bash
#SBATCH -p [partition name]
#SBATCH --job-name consolidating_miniRfam
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --exclusive
#SBATCH --nodes=4

## ACTIVATE CONDA
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bioinfo_tools_Py_310

infernal_dir="/home/ubuntu/anaconda3/bin/"

touch 'mini_rfam.fa'
touch 'mini_rfam.counts'

# iterate through each Rfam ID
while read rfam_id; do
	echo "$rfam_id"
	#
	# check number of sequences
	N_SEQ=$(${infernal_dir}esl-seqstat ${rfam_id}/${rfam_id}_5.fa | grep 'Number of sequences' | cut -c22-)
	#
	# if number of sequences if above 10, add to conglomerate alignment
	if [ $N_SEQ -ge 10 ]
	then
		cat mini_rfam.fa ${rfam_id}/${rfam_id}_5.fa > mini_rfam.fa.temp
		mv mini_rfam.fa.temp mini_rfam.fa
		cp ${rfam_id}/${rfam_id}_5.fa final_split_by_model//${rfam_id}_final.fa
		echo "$rfam_id $N_SEQ" >> 'mini_rfam.counts'
	fi
	#
done < rfam_ids.txt


