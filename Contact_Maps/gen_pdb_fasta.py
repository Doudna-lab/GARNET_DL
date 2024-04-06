
import os
import sys
import pandas as pd
from common.bio import *
from pdblib2.num import *

rna = 'TPP'
rna = 'Cobalamin'
rna = 'rRNA'

pdbdir = '%s/pdb'%rna

df0 = pd.read_csv('%s/table/pdb_info.csv'%rna)

output = [['pdbid','bundle_id','chain','chain_index','chid','chain_name','pdbseq_len','pdbseq']]

for i in range(len(df0)):
    
    pdbid = df0['pdbid'].ix[i]
    bundle_id = df0['bundle_id'].ix[i]
    if bundle_id == 1:
        chain_offset = 0
    else:
        chain_offset = chain_index + 1
    pdb_f = df0['pdb_f'].ix[i]+'.pdb'

    mol = Mol(os.path.join(pdbdir,pdb_f))
    print(pdbid)
    for j, seg in enumerate(mol.segs):
        chain_index = chain_offset + j
        chid = j
        chain = pdbid.upper() + '_' + str(chain_index+1) 
        chain_name = seg.chid
        reses = seg.reses
        if reses[0].name not in modabbr.keys():
            continue
        reses_names = [modabbr[res.name].upper() for res in reses if res.name not in solvent_list]
        pdbseq_len = len(reses_names)
        pdbseq = ''.join(reses_names)
        output.append([pdbid,bundle_id,chain,chain_index,chid,chain_name,pdbseq_len,pdbseq])
    

output_df = pd.DataFrame(output[1:],columns=output[0])
print(output_df)
output_df.to_csv('%s/table/%s_pdb_fasta.csv'%(rna,rna),index=False)

