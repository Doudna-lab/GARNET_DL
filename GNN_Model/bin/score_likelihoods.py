import argparse
import numpy as np
from pathlib import Path
from Bio import SeqIO
import torch
import math
import torch.nn.functional as F
from struct2seq.rna_struct2seq import RNAStruct2Seq
from tqdm import tqdm
import pandas as pd


def load_gnn(model_path, dist_map, device='cuda'):
    vocab_size = 6
    num_node_feats = 64
    num_edge_feats = 64
    hidden_dim = 128
    num_encoder_layers = 1
    num_decoder_layers = 3

    k_nbrs = 50

    model = RNAStruct2Seq(vocab_size, num_node_feats, num_edge_feats, dist_map, hidden_dim, num_encoder_layers, num_decoder_layers, k_nbrs)
    model.to(device)

    sd = torch.load( model_path)['model_state_dict']
    for k, v in list(sd.items()):
        sd[k.split('module.')[1]] = sd.pop(k)
    model.load_state_dict(sd)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', '-i', type=Path, required=True)
    parser.add_argument('--out_csv', '-o', type=Path, required=True)
    parser.add_argument('--batch_size', '-bs', type=int, default=100)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--distance_map', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dist_map = torch.tensor(np.load(args.distance_map), device='cpu')

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: User requested GPU but GPU is not available... using CPU")

    model = load_gnn(args.model_path, dist_map, device=device)
    ids, seqs = zip(*[(s.id, str(s.seq)) for s in SeqIO.parse(args.input_fasta, 'fasta')])

    i_to_nt = ['A', 'U', 'C', 'G', '-', 'X']
    nt_to_i = {nt:i for i, nt in enumerate(i_to_nt)}

    X = torch.tensor([[nt_to_i[nt] for nt in s] for s in seqs])    

    bs = args.batch_size
    num_batches = math.ceil(len(X) / bs)
    X_b = [X[i*bs:(i+1)*bs] for i in range(num_batches)]
    with torch.no_grad():
        log_liks = []
        for x in tqdm(X_b, desc="Computing Log Likelihoods"):
            log_probs = model(x.to(device)).cpu()
            log_lik = (F.one_hot(x, num_classes=6) * log_probs).sum(dim=-1).sum(dim=-1)
            log_liks.append(log_lik) 
        log_liks = torch.cat(log_liks).cpu().numpy()
    df = pd.DataFrame(zip(ids, seqs, log_liks), columns=["ID", "Seq", "LogLikelihood"])
    df.to_csv(args.out_csv, index=False)
    print(df)
