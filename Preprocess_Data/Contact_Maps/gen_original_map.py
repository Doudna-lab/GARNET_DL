
import os
import sys
import time
import pandas as pd
import numpy as np
from common.bio import *
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
#df0 = pd.read_csv('rRNA/table/pdb_23S.csv')
df0 = pd.read_csv('%s/table/pdb_%s.csv'%(rna,rna))
df_fasta = pd.read_csv('%s/table/%s_pdb_fasta.csv'%(rna,rna))

target_list = [
'6v39',
'7s9u',
'8fmw',
'5dm6',
'7nhk',
'6hrm',
'7jil',
'3cc2',
'8a57',
'7sfr',
'7s0s',
'7ood',
'6spb',
'5ngm',
'8hku',
'6skf',
'4w2e',
'4ybb',
'7k00',
]

output = [['pdbid','pdbseq_len','pdbseq']]

for i in range(len(df0)):

    pdbid = df0['pdbid'].ix[i]
    if pdbid not in target_list:
        continue

    pdbid = df0['pdbid'].ix[i]
    pdb_f = df0['pdb_f'].ix[i]+'.pdb'
    chain_index = df0['chain_index'].ix[i]
    chid = df0['chid'].ix[i] 
    pdbseq_len = df_fasta['pdbseq_len'][(df_fasta.pdbid == pdbid) & (df_fasta.chain_index == chain_index)].values[0]
    pdbseq = df_fasta['pdbseq'][(df_fasta.pdbid == pdbid) & (df_fasta.chain_index == chain_index)].values[0]
    output.append([pdbid,pdbseq_len,pdbseq])

    mol = Mol(os.path.join(pdbdir,pdb_f))
    seg = mol.segs[chid]

    MM_list = []
    for res in seg.reses:

        if res.name in solvent_list:
            continue
        else:
            M = getmat(res)
            MM_list.append(M)

    if len(MM_list) != pdbseq_len:
        print("ERROR(): inconsistent length of rRNA: %s"%pdbid)
        sys.exit()


    MM = np.array(MM_list)

    time1 = time.time()
    pairwise_idx = it.combinations(range(len(MM)),2)
    pairwise_values = it.combinations(MM,2)
    matrix_size = len(MM)
    
    contact_matrix = np.zeros((matrix_size,matrix_size),dtype=float)
    
    for idx, values in zip(pairwise_idx,pairwise_values):
    
        mindist = cdist(values[0],values[1]).min()
        contact_matrix[idx[0],idx[1]] = mindist
        contact_matrix[idx[1],idx[0]] = mindist
    
    time2 = time.time()
    print("Time: %f sec"%((time2-time1)))
    print(np.shape(contact_matrix))
    print(contact_matrix)

    np.savetxt('%s/map/hs_raw_map_%s.csv'%(rna,pdbid),contact_matrix,delimiter=',',fmt='%.8f')

output_df = pd.DataFrame(output[1:],columns=output[0])
print(output_df)
output_df.to_csv('%s/table/%s_pdb_23S_sequence.csv'%(rna,rna),index=False)

