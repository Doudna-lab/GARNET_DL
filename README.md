# DESCRIPTION
This repository contains all code described in our recent manuscript: 
- Yekaterina Shulgina\*, Marena I. Trinidad\*, Conner J. Langeberg\*, Hunter Nisonoff\*, Seyone Chithrananda\*, Petr Skopintsev\*, Amos J. Nissley\*, Jaymin Patel, Ron S. Boger, Honglue Shi, Peter H. Yoon, Erin E Doherty, Tara Pande, Aditya M. Iyer, Jennifer A. Doudna, Jamie H. D. Cate\*. *RNA language models predict mutations that improve RNA function*. bioRxiv 2024.04.05.588317; doi: https://doi.org/10.1101/2024.04.05.588317

    \* Contributed equally 

It includes the curation of diverse RNA datasets from the Genome Taxonomy Database (GTDB), prediction of optimal growth temperature (OGT) phenotypes, and the application of LM and GNN models for the development of thermostable ribosomes. These resources include examples for dataset preprocessing, the generation of training and test sets utilizing hierarchical clustering with CD-HIT, model validation, and sequence generation. This project is a community effort coordinated by The Innovative Genomics Institute and Department of Electrical Engineering and Computer Sciences at University of California Berkeley.

The GARNET database (GTDB-Acquired RNA with Environmental Temperatures) is freely available here:  [https://doi.org/10.5281/zenodo.12541208](https://doi.org/10.5281/zenodo.14003346)

## Software Requirements and Installation with Anaconda

  All software requirements are specified in the following yml files. Dependencies may be configured with [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) as detailed below.
- Dependencies for Rfam curation, data preprocessing, figures and OGT prediction:
    - Conda_Environments/data_processing_env.yml
- Dependencies for GNN Model:
    - Conda_Environments/gnn_environment.yml
- Dependencies for LM Model:
    - Conda_Environments/lm_environment.yml

<b>*Example Installation*:</b>

```bash
conda env create --file data_processing_env.yml --name data_processing_env
```
## Dataset Curation: Extraction of Rfams from GTDB
  
  Extended directions are available under GTDB_RNA_Curation/rna_alignment_methodology.md
- <b>Directory:</b>
    
  - GTDB_RNA_Curation/*
## Phenotype Annotation: OGT Prediction and Figure 2

  The workflow for predicting optimal growth temperatures with TOME and the generation of Figure 2 and S2 are provided in the notebook below. 
- <b>Directory:</b>
    - GTDB_RNA_Curation/Figure_2_OGT.ipynb

## Train/Test Split: Hierarchical Clustering with CD-HIT

  Train-test splits were performed using hierarchical clustering with CD-HIT-EST, as described in the manuscript. Step-by-step commands for splitting the rRNA datasets, 228 Rfam sequences, aggregation of the 231 Rfam superset and generation of the thermophile finetuning sets are described in the following notebook: Train_Test_Split/Train_Test_Splits.ipynb.
- <b>Directory:</b>
    -  train_Test_Split/
- <b>Dependency Files (Zenodo):</b>
    - RNA Sequences in fasta format (*fa.gz)
    - hyperthermophiles_60dC_gtdb_ids.txt
        
## Dataset Preprocessing: Create Contact Maps for GNN Model
- <b>Directory</b>:
    - Contact_Maps/*
    
## Language Model:

  An RNA language model built on a GPT framework, starting from the nanoGPT code at https://github.com/karpathy/nanoGPT. Modifications include rotary positional embedding and rms normalization. The optimal tokenization scheme involves overlapping nucleotide triplets, i.e. triplets with 1-nt steps for each token. Models were pretrained (PT) using all RNA sequences or finetuned (FT) on hyperthermophile sequences. Models include those trained on 23S sequences only, or on the 231 RFAM RNA set. Config files for training and logs provided in respective subdirectories.
- <b>Directory:</b>
    - LM_Model/
- <b>Dependency Files</b>:
    - configurator.py
    - model_RNA_rot.py
- <b>*Tokenization Example:*</b>
  - Generate Training Tokens:    
  ```bash
  python prepare_RNA_train.py -i 23S_train.fa.gz -o 23S_triples --type triples
  ```
  - Generate Validation Tokens:
  ```bash
  python prepare_RNA_val.py -i 23S_test.fa.gz -o 23S_triples --type triples
  ```
  - Outputs:
      - 23S_triples_train.bin
      - 23S_triples_val.bin, 23S_triples_meta.pkl
- <b>*Training Example:*</b>
  ```bash
  python -u train_RNA_rot_flash.py ./Config/train_23S_triples_0_18_6_300_rot_flash.py &> ./Logs/train_23S_triples_0_18_6_300_rot_flash.log
  python -u resume_RNA_rot_flash.py ./Config/resume_23S_triples_0_18_6_300_rot_flash.py &> ./Logs/resume_23S_triples_0_18_6_300_rot_flash.log
  ```
  - Output: out/23S_triples_resume_0_18_6_300_rot_flash.pt
- <b>*Finetuning Example:*</b>
  ```bash
  python -u finetune_RNA_rot_flash.py ./Config/finetune_23S_Hyperthermophiles_triples_0_18_6_300_rot_flash.py &> ./Logs/finetune_23S_Hyperthermophiles_triples_0_18_6_300_rot_flash.log
  ```
  - Output:
      - out/23S_Hyperthermophiles_triples_finetune_0_18_6_300_rot_flash.pt
## GNN Model: Graph-Based RNA Sequence Modeling
- <b>Directory:</b>
    - GNN_Model/*

## Generated Sequences: Sequences Generated by LM and GM Models
  
  23S rRNA sequences (or 16S sequences) generated using a seed from the 5’ end of the corresponding E. coli 23S rRNA or 16S rRNA.
- <b>Directories:</b>
    - Generated_Sequences/*
    - LM_Model/*
- <b>Dependency Files:</b>
    - model *.pt files
    - model_RNA_rot.py
    - configurator.py
    - Ecoli_23S_7K00.fasta
    - Ecoli_16S_rRNA.fasta
- <b>*Example*:</b>

  For the RNA LM’s, sequences generated using either the pretrained (PT) or hyperthermophile finetuned (FT) models. Models include those trained on 23S sequences only, or on the 231 RFAM RNA set (231 or Superset in file names).
  ```bash
  python -u sample_RNA.py &> sample_23S_1.log
  ```
## Validation: Likelihoods of Generated Sequence
  
  Calculation of the probability of generating a particular sequence from a model. Input sequences should all be of the same length, in multifasta format.
- <b>Directory:</b>
    - Validation/*  
- <b>Dependency Files:</b>
    - Pretrained (PT) model .pt file or hyperthermophile finetuned (FT) model .pt file
    - configurator.py
    - model_RNA_rot_v2.py
- <b>*Example:*</b>
  ```bash
  python -u seqprob_23S_PT_v2_sweep.py &> seqprob_23S_EcoliWT_sweep_231RNAsmodel_PT.log
  python -u seqprob_23S_FT_v2_sweep.py &> seqprob_23S_EcoliWT_sweep_231RNAsmodel_FT.log
  ```

