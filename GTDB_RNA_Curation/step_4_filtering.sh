#!/bin/bash
#SBATCH -p [partition name]
#SBATCH --job-name filtering_miniRfam
#SBATCH -o %j.out
#SBATCH -e %j.out
#SBATCH --exclusive
#SBATCH --nodes=2

## ACTIVATE CONDA
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bioinfo_tools_Py_310

infernal_dir="/home/ubuntu/anaconda3/bin/"

# iterate through each Rfam ID
while read rfam_id; do
	echo "$rfam_id"
	# filter based on match state percentage
	echo "filtering based on match states..."
	python filter_by_match_percent_4.py ${rfam_id}/${rfam_id}_3.sto 0.90 ${rfam_id}/${rfam_id}_match_state_remove_list_4.txt
	#
	# filter based on ambiguity character percentage
	echo "filtering based on ambiguity characters..."
	python filter_by_ambiguity_4.py ${rfam_id}/${rfam_id}_3.sto 0.05 ${rfam_id}/${rfam_id}_ambiguity_remove_list_4.txt
	#
	# filter based on total sequence length
	echo "filtering based on sequence length..."
	python filter_by_length_4.py ${rfam_id}/${rfam_id}_3.sto 2 ${rfam_id}/${rfam_id}_length_remove_list_4.txt
	#
	# filter based on secondary structure preservation
	echo "filtering based on secondary structure..."
	python filter_by_secondary_structure.py ${rfam_id}/${rfam_id}_3.sto 2 ${rfam_id}/${rfam_id}_secondary_structure_remove_list_4.txt
	#
	# combine these lists
	echo "combining lists..."
	cat ${rfam_id}/${rfam_id}_match_state_remove_list_4.txt ${rfam_id}/${rfam_id}_ambiguity_remove_list_4.txt ${rfam_id}/${rfam_id}_length_remove_list_4.txt \
		${rfam_id}/${rfam_id}_secondary_structure_remove_list_4.txt | sort | uniq > ${rfam_id}/${rfam_id}_remove_list_4.txt
	#
	# filter remove-list sequences out
	echo "removing sequences from alignment..."
	if [ ! -s "${rfam_id}/${rfam_id}_remove_list_4.txt" ]; then
    		echo "removal list is empty; copying over old alignment..."
		cp ${rfam_id}/${rfam_id}_3.sto ${rfam_id}/${rfam_id}_4.sto
	else
		${infernal_dir}esl-alimanip --seq-r ${rfam_id}/${rfam_id}_remove_list_4.txt ${rfam_id}/${rfam_id}_3.sto > ${rfam_id}/${rfam_id}_4.sto
	fi
	#
done < rfam_ids.txt


