# DESCRIPTION
This repository contains all code described in our recent manuscript, including the curation of diverse RNA datasets from the Genome Taxonomy Database (GTDB), prediction of optimal growth temperature (OGT) phenotypes, and utilization of GPT and GNN models for the development of thermostable ribosomes. These resources include examples for dataset preprocessing, the generation of training and test sets utilizing hierarchical clustering with CD-HIT and model validation. This project is a community effort coordinated by The Innovative Genomics Institute and Department of Electrical Engineering and Computer Science at University of California Berkeley.

The GARNET database (GTDB-Acquired RNA with Environmental Temperatures) is freely available here: https://tinyurl.com/5abszup9

## CONTENTS
- Dataset curation: Extraction of Rfams from GTDB
    - GTDB_RNA_Curation/*sh
    - GTDB_RNA_Curation/*py
- Phenotype Annotation: OGT Prediction and Figure 2
    - GTDB_RNA_Curation/Figure_2_OGT.ipynb
- Train/Test Split
    - Train_Test_Split/Train_Test_Splits.ipynb
- Dataset Preprocessing
    - Contact_Maps/*
- Language Model:
    - LM_Model/*
- GNN Model: Graph-Based RNA Sequence Modeling
    - GNN_Model/*
- Validation: Generate Likelihoods
    - Validation/*
