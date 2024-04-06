
import os
import sys
import time
import pandas as pd
import numpy as np
from common.matvec import *
from pdblib2.num import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import itertools as it
import operator as op
import pandas as pd
from matplotlib import colors


def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))

rna = 'rRNA'
#rna = 'TPP'

pdbdir = '%s/pdb'%rna
input_dir = '%s/alignment'%rna
input_f = 'pdb_sequences_corr.fa'
#df0 = pd.read_csv('rRNA/table/pdb_23S.csv')
df0 = pd.read_csv('%s/table/pdb_%s.csv'%(rna,rna))

align_seq_dic = {}

with open(os.path.join(input_dir,input_f)) as fp:
    for n, seq in read_fasta(fp):
        align_seq_dic[n[1:]] = seq

output = [['pdbid','chain_name','chain_index','pdbseq_len','pdbseq']]

for i in range(len(df0)):

    if i >= 2:
        continue

    pdbid = df0['pdbid'].ix[i]
    pdb_f = df0['pdb_f'].ix[i]+'.pdb'
    chain_index_23 = df0['chid'].ix[i] 
    align_seq = align_seq_dic[pdbid]
    no_gap_seq = align_seq.replace('-','').replace('.','')
    gap_index = np.array([j for j in range(len(align_seq)) if align_seq[j] in ['-','.']])
    indel_index = np.array([j for j in range(len(align_seq)) if align_seq[j] in ['-','.','a','u','g','c']])
    lower_index = np.array([j for j in range(len(no_gap_seq)) if no_gap_seq[j] not in ['A','U','G','C']])
    upper_index = np.array([j for j in range(len(no_gap_seq)) if no_gap_seq[j] in ['A','U','G','C']])


    mol = Mol(os.path.join(pdbdir,pdb_f))
    seg = mol.segs[chain_index_23]

    MM_list = []
    for res in seg.reses:

        M = getmat(res)
        MM_list.append(M)

    MM = np.array(MM_list)
    n_b = len(MM) + len(gap_index)
    not_index = np.array([k for k in range(n_b) if k not in indel_index])

    time1 = time.time()
    pairwise_idx = it.combinations(range(len(MM)),2)
    pairwise_values = it.combinations(MM,2)
    matrix_size = len(MM)
    
    contact_matrix = np.zeros((matrix_size,matrix_size),dtype=float)
    
    for idx, values in zip(pairwise_idx,pairwise_values):
    
        mindist = cdist(values[0],values[1]).min()
        if mindist <= 4.5:
            contact_matrix[idx[0],idx[1]] = 1.
            contact_matrix[idx[1],idx[0]] = 1.
        else:
            contact_matrix[idx[0],idx[1]] = 0.
            contact_matrix[idx[1],idx[0]] = 0.
    
    time2 = time.time()
    print("Time: %f sec"%((time2-time1)))
    print(np.shape(contact_matrix))
    print(contact_matrix)
    

    n_b = contact_matrix.shape[0] + len(gap_index)
    not_index = np.array([k for k in range(n_b) if k not in indel_index])
    #b = np.zeros((n_b, n_b), dtype=dist_matrix.dtype)
    b = np.full((n_b,n_b), 0., dtype=contact_matrix.dtype)
    b[not_index.reshape(-1,1), not_index] = contact_matrix[upper_index.reshape(-1,1),upper_index]
    b[not_index,not_index] = 1.
    print(np.shape(b))
    np.savetxt('%s/map/hs_raw_mask_%s.txt'%(rna,pdbid),b,fmt='%i')

    #orig_map=plt.cm.get_cmap('hot') 
    #reversed_map = orig_map.reversed()
    #fig = plt.figure(1,dpi=400)
    #plt.imshow(b,interpolation=None,cmap=reversed_map,vmin=0.,vmax=2.,resample=False)
    #plt.hist(b,cmap='jet')
    #plt.colorbar()
    #plt.savefig('rRNA/plot/task_2_mask_plot_single_%s.pdf'%(pdbid))
    #plt.clf()

    

