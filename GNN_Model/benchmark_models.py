import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from struct2seq.rna_features import RNAFeatures, PositionalEncodings
from struct2seq.rna_struct2seq import RNAStruct2Seq
from struct2seq import noam_opt
import torch.nn as nn
import torch.nn as nn

import wandb
import argparse
#TODO: get argparse working, and wandb/tensorboard for experiment tracking

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1, vocab_size=6):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, num_classes=vocab_size).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RNA Structure to Sequence Model Training')
    parser.add_argument('--vocab_size', type=int, default=6, help='Vocabulary size')
    parser.add_argument('--num_node_feats', type=int, default=64, help='Number of node features')
    parser.add_argument('--num_edge_feats', type=int, default=64, help='Number of edge features')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers') ## same as Ingraham
    parser.add_argument('--smoothing_weight', type=float, default=0.1, help='Smoothing weight') ## same as Ingraham
    parser.add_argument('--k_nbrs', type=int, default=10, help='Number of nearest neighbors')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test_data', type=str, default='thermophile_data_feb_24/test.pt', help='Test data file')
    parser.add_argument('--finetune_path', type=str, default=None, help='Directory to finetune from')
    ## dropout = 0.1 (fixed as per paper)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    basedir = Path('.').resolve()
    processed_dir = basedir / 'data/rna/processed_for_ml'

    dist_map = torch.tensor(np.load(processed_dir / 'distance_map.npy'), device='cpu')
    
    X_test = torch.load(processed_dir / args.test_data) #'#jan_2023_23s_data/new_23s_test.pt')
    #X_train = torch.load(processed_dir / 'train.pt')
    #X_val = torch.load(processed_dir / 'val.pt')

    test_dl = DataLoader(TensorDataset(X_test), batch_size=10, shuffle=False)

    # Hyperparameters -- partially based on Ingraham
    vocab_size = args.vocab_size
    num_node_feats = args.num_node_feats
    num_edge_feats = args.num_edge_feats
    hidden_dim = args.hidden_dim
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    smoothing_weight = args.smoothing_weight
    k_nbrs = args.k_nbrs ## this is a good hyperparameter to tune based on the argsort of nearest contact distances
    
    #device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RNAStruct2Seq(vocab_size, num_node_feats, num_edge_feats, dist_map, hidden_dim, num_encoder_layers, num_decoder_layers, k_nbrs)

    checkpoint = torch.load(args.finetune_path, map_location=device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    optimizer = noam_opt.get_std_opt(model.parameters(), hidden_dim)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss(reduction='none')

    # Test
    with torch.no_grad():
        test_sum = 0.0
        test_weights = 0.0
        model.eval()
        for test_i, S in tqdm(enumerate(test_dl), total=len(test_dl)):
            S = S[0].to(device)
            mask = torch.ones_like(S)
            log_probs = model(S)
            loss, loss_av = loss_nll(S, log_probs, mask)
            test_sum += torch.sum(loss * mask).item()
            test_weights += torch.sum(mask).item()
    print('model:' + str(args.finetune_path))
    print(f"Test Loss: {test_sum / test_weights:.3f}")
    print(f"Test Perplexity: {np.exp(test_sum / test_weights):.3f}")
    #test_loss = test_sum / test_weights
    #wandb.log({"test_loss": test_loss, "test_perplexity": np.exp(test_loss)})
    #print("Finished training")
    #wandb.finish()
