#!/bin/bash

python score_likelihoods.py \
       -i example_input_log_liks.fa \
       -o example_output.csv \
       --model_path /home/hunter/projects/structure/structure-based-rna-model/log/feb4_rna_hidden_dim_128_k_nbrs_50/models/checkpoint_32.pt \
       --distance_map /home/hunter/projects/structure/structure-based-rna-model/data/rna/processed_for_ml/distance_map.npy
