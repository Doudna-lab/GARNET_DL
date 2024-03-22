#!/bin/bash
#SBATCH -p cpu-c6i-16xlarge
#SBATCH --job-name cmalign_miniRfam
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --exclusive
#SBATCH --nodes=6

## ACTIVATE CONDA
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bioinfo_tools_Py_310

infernal_dir="/home/ubuntu/anaconda3/bin/"
rfam_file="/home/ubuntu/datasets/rfam/Rfam.cm"
gtdb_file="/home/ubuntu/datasets/gtdb/gtdb_genomes_reps_r214/database/gtdb_genomes_reps_r214.fna"

# iterate through each Rfam ID
while read rfam_id; do
	echo "$rfam_id"
	mkdir $rfam_id
	#
	# first, get Rfam CM
	echo "fetching CM file from Rfam..."
	${infernal_dir}cmfetch $rfam_file $rfam_id > $rfam_id/$rfam_id.cm
	#
	# next, get all sequences for this Rfam and align them
	echo "aligning sequences from just $rfam_id"
	awk -v var=$rfam_id '$7 ~ var {print}' RF00001-4192_stats_overlaps_removed_2.list > $rfam_id/${rfam_id}_info_3.txt
	${infernal_dir}esl-sfetch -C -f $gtdb_file $rfam_id/${rfam_id}_info_3.txt | ${infernal_dir}cmalign -o $rfam_id/${rfam_id}_3.sto $rfam_id/$rfam_id.cm -
	#
done < rfam_ids.txt


