## Building GARNET RNA alignments

To build diverse and minimally redundant alignments of RNA sequences, we used Infernal to search for hits to Rfam covariance models in GTDB representative genomes. Methodology for searches for rRNA genes vs other RNA families were slightly different, mainly 1) when searching for rRNA sequences, we also looked at non-representative genomes if no hit could be found in the representative genome for a species cluster and 2) for rRNA, we only considered the most significant hit, while for other types of RNA we kept multiple hits per genome. The methodology used for both types of searches and subsequent quality control is described in detail below. 

### GTDB and Rfam files

#### GTDB files

For our analyses, we used GTDB R214. This version of GTDB contains 402,709 genomes, which are organized into 85,205 species clusters. Each species cluster has one designated representative genome, and the remaning genomes are non-representative. Throughout GARNET, we use the accession for the represenative genome as the unique identifier for the species cluster.

Upon downloading, the representative genomes are located in the directory `gtdb_genomes_reps_r214`, which has a nested structure where genomes go into subdirectories based on their accession IDs. For example, the genome `GCA_900766195.1` is located in `gtdb_genomes_reps_r214/database/GCA/900/766/195/GCA_900766195.1_genomic.fna.gz`

The main GTDB metadata files have information about all genomes, representative and non-representative. For ease, we subsetted these files to only representatives with

	awk -F "\t" '$16=="t" {print}' ar53_metadata.tsv > ar53_rep_only_metadata.tsv
	awk -F "\t" '$16=="t" {print}' bac120_metadata.tsv > bac120_rep_only_metadata.tsv

The first column of the metadata file contains the genome accession, looking something like `RS_GCF_001425175.1`. Notice that the first three characters aren’t used in the path to where the genome is located.


#### Rfam models

For our analyses, we used Rfam 14.9.

We fetched individual covariance models (CMs) from the main Rfam.cm files with this Infernal command (example for 23S rRNA CMs)

	cmfetch <path to Rfam.cm> RF02540 > RF02540.cm
	cmfetch <path to Rfam.cm> RF02541 > RF02541.cm

See Supplementary Table 1 for the list of all Rfam models used in our searches. 

### Searching for rRNA sequences in GTDB genomes

For rRNA hits, we first searched the representative genome of each species cluster using Infernal 1.1.4 for an aligned hit with e-value < 1e-5 and >85% of model length, keeping the most significant hit per genome. If no such hit could be found, then any available non-representative genomes for that species cluster were searched, in order of increasing CheckM contamination (provided in the GTDB metadata). 


#### Python script

To automate this process, we wrote a Python script.

This script searches a species cluster for hits to Rfam model (searching first representative then non-representative genomes), and then writes the sequence and info about the hit to summary files.

Note: This script non-functional as-is because it is missing a value for `download_cmd`, a string specifying a command for downloading a genome from NCBI given a genome accession to the path specified by the variable `genome_file`.  



	# Script to use Infernal cmsearch to look for a particular CM in all genomes for a 
	# particular GTDB species. If a hit that covers >85% of model match positions cannot 
	# be found, then look in non-representative genomes sorted by lowest contamination 
	# until such a hit can be found. Infernal outputs are saved as tblout format in the
	# same directory as the original representative genome. Final hits and info about 
	# them are saved in summary files that all parallel jobs write to (writing to same 
	# summary files in parallel doesn't work as intended). 
	# Arguments are 1. GTDB species cluster representative genome accession (ex/ 
	# RS_GCF_001425175.1), 2. path to CM file, 3. path to metadata file to find 
	# non-representative genomes, and 4. prefix for the summary files (something like 
	# "bacteria")
	
	from subprocess import call, Popen, PIPE
	import sys
	import os
	import fcntl
	
	rep_accession = sys.argv[1]
	rfam_model = sys.argv[2]
	metadata_file = sys.argv[3]
	output_prefix = sys.argv[4]
	
	# get model length (in order to know if the top hit is ~>85%)
	p = Popen('grep CLEN %s' % rfam_model, shell=True, stdout=PIPE)
	output = p.communicate()[0].decode().rstrip().split()
	model_length = int(output[-1])
	
	# just make sure the summary files exist
	summary_file = '%s_%s_info.csv' % (output_prefix, rfam_model)
	fasta_file = '%s_%s.fa' % (output_prefix, rfam_model)
	p = Popen('touch %s' % summary_file, shell=True)
	dum = p.wait()
	p = Popen('touch %s' % fasta_file, shell=True)
	dum = p.wait()
	
	# reconstruct path to representative genome
	r_acc = rep_accession[3:]
	f1 = r_acc[:3]
	f2 = r_acc[4:7]
	f3 = r_acc[7:10]
	f4 = r_acc[10:13]
	genome_dir = 'gtdb_genomes_reps_r214/database/%s/%s/%s/%s' % (f1, f2, f3, f4)
	rep_genome_file = '%s/%s_genomic.fna' % (genome_dir, r_acc)
	
	# get info about non-representative genomes from GTDB metadata file: ID, percent 
	# completeness, percent contamination
	awk_command = "awk -F '\t' '$15 == \"%s\" && $16 == \"f\" {print $1, $3, $4}' %s" \
		% (rep_accession, metadata_file)
	p = Popen(awk_command, shell=True, stdout=PIPE)
	output = p.communicate()[0].decode().rstrip().split('\n')
	if len(output[0]) > 0:
	    nonrep_info = [[i.split()[0], float(i.split()[1]), float(i.split()[2])] \
	    	for i in output]
	else:
	    nonrep_info = []
	    
	# sort first by completeness, then by contamination. Thus low contamination is 
	# prioritized over completeness
	nonrep_info.sort(key=lambda x: -x[1])
	nonrep_info.sort(key=lambda x: x[2])
	genomes = [rep_accession] + [i[0] for i in nonrep_info]
	hit_found = False
	
	# iterate through each genome and look for a suitable hit 
	for accession in genomes:
	    acc = accession[3:]
	    
	   # download these genomes (this is mainly for non-representatives, which are 
	   # not provided by GTDB
	   genome_file = '%s/%s_genomic.fna' % (genome_dir, acc)
	   if not os.path.isfile(genome_file):
	   # download_cmd should be defined-- I used a personal script that downloads 
	   # genomes from NCBI based on accession
	        p = Popen(download_cmd, shell=True)
	        p.wait()
	    # if genome download failed, skip. Sometimes genomes are repressed by NCBI.
	    if not os.path.isfile(genome_file):
	        continue
	    
	    # cmsearch
	    tblout_file = '%s.%s.tblout' % (genome_file, rfam_model)
	    if not os.path.isfile(tblout_file) or os.path.getsize(tblout_file) < 900:
	        p = Popen("cmsearch --tblout %s %s %s" % (tblout_file, rfam_model, \
	        genome_file), shell=True)
	        p.wait()
	    
	    # extract information about cmsearch hits
	    with open('%s.%s.tblout' % (genome_file, rfam_model), 'r') as ar:
	        lines = [line.rstrip('\n') for line in ar]
	       
	    hits = list()
	    for line in lines:
	        if line[0] != '#':
	            hits.append(line)
	    
	    # if no hits at all, move onto the next genome
	    if len(hits) == 0:
	        continue
	    
	    # see if top hit is approximately 85% complete or better by looking at 
	    # ([model end] - [model start]) / [model length]
	    top_hit = hits[0]
	    top_hit_info = top_hit.split()
	    hit_model_length = int(top_hit_info[6]) - int(top_hit_info[5])
	    approx_overlap = hit_model_length / model_length
	    if approx_overlap < 0.85:
	        continue
	    
	    # save this hit
	    contig = top_hit_info[0]
	    model_start = int(top_hit_info[5])
	    model_end = int(top_hit_info[6])
	    seq_start = int(top_hit_info[7])
	    seq_end = int(top_hit_info[8])
	    strand = top_hit_info[9]
	    evalue = top_hit_info[15]
	    
	    # make esl-sfetch index
	    if not os.path.isfile('%s.ssi' % genome_file):
	        p = Popen("esl-sfetch --index %s" % genome_file, shell=True)
	        p.wait()
	    
	    # retrieve actual sequence using esl-sfetch
	    p = Popen("esl-sfetch -c %i-%i %s %s" % (seq_start, seq_end, genome_file, \
	    	contig), shell=True, stdout=PIPE)
	    esl_output = p.communicate()[0].decode()
	    rna_seq = ''.join(esl_output.rstrip().split('\n')[1:])
	    hit_len = len(rna_seq)
	    
	    # write to summary file
	    # columns are: species representative, actual accession that hit is from, 
	    # y/n whether actual accession is rep, y/n hit,  hit length, e-value, contig, 
	    # strand, seq start, seq end, model start, model end
	    yn_rep = 'n'
	    if accession == rep_accession:
	        yn_rep = 'y'
	    with open(summary_file, 'a') as sf:
	        fcntl.flock(sf, fcntl.LOCK_EX)
	        dum = sf.write('%s,%s,%s,y,%i,%s,%s,%s,%i,%i,%i,%i\n' % (rep_accession, \
	        	accession, yn_rep, hit_len, evalue, contig, strand, seq_start, seq_end, \
	        	model_start, model_end))
	        fcntl.flock(sf, fcntl.LOCK_UN)
	    
	    # write sequence to fasta files
	    with open('%s.%s.%s_hit.fa' % (rep_genome_file, accession, rfam_model), 'w') as ff:
	        dum = ff.write('>%s\n%s\n' % (rep_accession, rna_seq))
	    with open(fasta_file, 'a') as sff:
	        fcntl.flock(sff, fcntl.LOCK_EX)
	        dum = sff.write('>%s\n%s\n' % (rep_accession, rna_seq))
	        fcntl.flock(sff, fcntl.LOCK_UN)
	    
	    hit_found = True
	    break
	
	# write to info summary file if no sequence was found
	if hit_found == False:
	    with open(summary_file, 'a') as sf:
	        fcntl.flock(sf, fcntl.LOCK_EX)
	        dum = sf.write('%s,-1,-1,n,-1,-1,-1,-1,-1,-1,-1,-1\n' % (rep_accession))
	        fcntl.flock(sf, fcntl.LOCK_UN)


This script can be sent in parallel for all species clustering using SLURM job arrays.
Here is an example SLURM script to send these jobs in parallel.

	#!/bin/bash
	#SBATCH -p [partition names]
	#SBATCH --time=30:00
	#SBATCH -c 1
	#SBATCH --mem=4000M
	#SBATCH -o /dev/null
	#SBATCH --array=1-4416
	
	[activate conda environment]
	
	awk 'NR=="'"${SLURM_ARRAY_TASK_ID}"'"' rep_ar53_assemblies | xargs -I {} python \
		cmsearch_script.py {} RF02540.cm ar53_metadata_r214.tsv archaea

#### Alignment and quality control of rRNA sequences
The resulting rRNA sequences from both bacteria and archaea were aligned back to a single Rfam model for each type of RNA (typically the bacterial Rfam model).

* 23S rRNA: RF02541
* 16S rRNA: RF00177
* 5S rRNA: RF00001

These alignments were subsequently processed for quality control, removing sequences that hit <85% of Rfam model, sequences that contain >5% ambiguity characters (ie Ns), sequences that are outliers based on length (too long), and sequences that contain non-Watson-Crick-Franklin base pairs for a large fraction of the Rfam consensus secondary structure. See the files `*_rRNA_filtering_pipeline.sh` files for actual commands used.

These scripts also contain the commands used to trim the final alignments for the GNN models.

### Searching for other RNA sequences

We searched for additional RNA families resulting in the 228 RNA families dataset. We first selected 256 Rfam models were based on 1) consensus model length >100 nt, 2) seed alignment containing >10 sequences, and 3) probable distribution in bacteria and/or archaea. This set does not include rRNA. The list of the 228 final Rfam families can be found in Supplementary Table 1. The selected CMs were concatenated into a file called `mini_Rfam.cm`.

We searched all GTDB representative genomes for hits to these Rfam models by first concatenating all of the GTDB representative genomes

	cat **/**/**/**/*.gz > gtdb_genomes_reps_r214.fna

And then searching these genomes using Infernal for this set of 256 Rfam models

	cmsearch --cpu 64 --tblout RF00001-4192.tbl mini_Rfam.cm gtdb_genomes_reps_r214.fna

The results were processed using this command:
	
	awk '{print $1"_"$8"-"$9" "$8" "$9" "$1" "$11" "$16" "$4}' RF00001-4192.tbl > RF00001-4192_stats.list

#### Alignment and quality control

The resulting hits were filtered for quality control in a six step process:

1. Only keeping hits with e-value < 1e-5
	
		grep -v '^#' RF00001-4192_stats.list | awk '$6<1e-5 {print}' > RF00001-4192_stats_evalue_filtered_1.list

2. For any two hits (to any model) that overlap by any number of nucleotides, only keep the hit with the more significant e-value. Use the script `step_2_overlap_removal.py`

3. Align remaining hits for each Rfam model back to the Rfam CM. Use the script `step_3_cmalign.sh`

4. For each Rfam model separately, filter remaining hits based on percent aligned match states > 90%, percent ambiguity characters < 5%, sequence length < mean + 2 standard deviation of the distribution for hits to the Rfam model, proportion non-Watson-Crick-Franklin base pairs < mean + 2 2 standard deviation of the distribution for hits to the Rfam model. Use the script `step_4_filtering.sh`

5. Re-align the remaining hits back to the Rfam model for each family. Use the script `step_5_realigning.sh`

6. Compile remaining sequences into a single fasta file, only keeping Rfam families with >10 hits found in this search. Use the script `step_6_consolidating.sh`


The resulting RNA sequences can be found in the Zenodo accompanying this manuscript.