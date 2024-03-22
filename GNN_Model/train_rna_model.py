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
    parser.add_argument('--distance_map', type=str, default='distance_map.npy', help='Name of distance map')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers') ## same as Ingraham
    parser.add_argument('--smoothing_weight', type=float, default=0.1, help='Smoothing weight') ## same as Ingraham
    parser.add_argument('--k_nbrs', type=int, default=10, help='Number of nearest neighbors')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--output_dir', type=str, default='test_rna', help='Output directory in log/ for saving models and log files')
    parser.add_argument('--train_data', type=str, default='thermophile_data_feb_24/train.pt', help='Training data file')
    parser.add_argument('--val_data', type=str, default='thermophile_data_feb_24/val.pt', help='Validation data file')
    parser.add_argument('--test_data', type=str, default='thermophile_data_feb_24/test.pt', help='Test data file')
    parser.add_argument('--finetune_path', type=str, default=None, help='Default none (assumes initializing from scratch); otherwise provide directory to finetune from')
    #parser.add_argument('--finetune_path', type=str, default='log/feb4_rna_hidden_dim_128_k_nbrs_50/models/checkpoint_32.pt', help='Directory to finetune from')
    ## dropout = 0.1 (fixed as per paper)

    args = parser.parse_args()
    wandb.init(project="feb4_structure-based-rna-model", entity="seyonec", config=args, name=args.output_dir)

    torch.manual_seed(args.seed)

    basedir = Path('.').resolve()
    processed_dir = basedir / 'data/rna/processed_for_ml'

    dist_map = torch.tensor(np.load(processed_dir / args.distance_map), device='cpu')
    
    X_train = torch.load(processed_dir / args.train_data) #'jan_2023_23s_data/new_23s_train.pt')
    X_val = torch.load(processed_dir / args.val_data) #'#jan_2023_23s_data/new_23s_val.pt')
    X_test = torch.load(processed_dir / args.test_data) #'#jan_2023_23s_data/new_23s_test.pt')
    #X_train = torch.load(processed_dir / 'train.pt')
    #X_val = torch.load(processed_dir / 'val.pt')

    train_dl = DataLoader(TensorDataset(X_train), batch_size=10)
    val_dl = DataLoader(TensorDataset(X_val), batch_size=10, shuffle=False)
    test_dl = DataLoader(TensorDataset(X_test), batch_size=10, shuffle=False)

    # Hyperparameters -- partially based on Ingraham (TODO: map to argparser or config file for easy experiment tracking)
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


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    if args.finetune_path is not None:
        checkpoint = torch.load(args.finetune_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    optimizer = noam_opt.get_std_opt(model.parameters(), hidden_dim)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss(reduction='none')

    # Log files
    log_folder = basedir / 'log' / args.output_dir
    log_folder.mkdir(exist_ok=True)
    model_folder = log_folder / 'models'
    model_folder.mkdir(exist_ok=True)

    logfile = log_folder / 'log.txt'
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

    start_train = time.time()
    epoch_losses_train, epoch_losses_valid = [], []
    patience = args.patience # TODO: hyperparameter
    epoch_checkpoints = []
    total_step = 0
    num_epochs=args.num_epochs
    for e in range(num_epochs):
        # Training epoch
        model.train()
        train_sum, train_weights = 0., 0.
        for train_i, S in tqdm(enumerate(train_dl), total=len(train_dl)):
            S = S[0].to(device)
            
            mask = torch.ones_like(S)
            start_batch = time.time()

            optimizer.zero_grad()
            log_probs = model(S)
            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask, weight=smoothing_weight)
            loss_av_smoothed.backward()
            optimizer.step()

            loss, loss_av = loss_nll(S, log_probs, mask)

            # Timing
            elapsed_batch = time.time() - start_batch
            elapsed_train = time.time() - start_train
            total_step += 1
            #print(total_step, elapsed_train, np.exp(loss_av.cpu().data.numpy()), np.exp(loss_av_smoothed.cpu().data.numpy()))


            # Accumulate true loss
            train_sum += torch.sum(loss * mask).cpu().data.numpy()
            train_weights += torch.sum(mask).cpu().data.numpy()
        print(f"Train Loss: {train_sum / train_weights:.3f}")

        with torch.no_grad():
            val_sum = 0.0
            val_weights = 0.0
            model.eval()
            for val_i, S in tqdm(enumerate(val_dl), total=len(val_dl)):
                S = S[0].to(device)
                mask = torch.ones_like(S)
                log_probs = model(S)
                loss, loss_av = loss_nll(S, log_probs, mask)
                val_sum += torch.sum(loss * mask).item()
                val_weights += torch.sum(mask).item()
        print(f"Val Loss: {val_sum / val_weights:.3f}")

        ## early stopping based on val loss
        train_loss = train_sum / train_weights
        val_loss = val_sum / val_weights
        epoch_losses_train.append(train_loss)
        epoch_losses_valid.append(val_loss)
        train_perplexity = np.exp(train_loss)
        val_perplexity = np.exp(val_loss)

        wandb.log({"train_loss": epoch_losses_train[-1], "val_loss": epoch_losses_valid[-1], "epoch": e, "val_perplexity": val_perplexity, "train_perplexity": train_perplexity})

        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_losses_valid[-1],
            }, str(model_folder) + f'/checkpoint_{e}.pt')

        with open(logfile, 'a') as f:
            f.write(f"{e}\t{epoch_losses_train[-1]:.3f}\t{epoch_losses_valid[-1]:.3f}\n")
        
        if e >= 1 and epoch_losses_valid[-1] > epoch_losses_valid[-2]:
            if patience == 0:
                print("Early stopping")
                break
            else:
                patience -= 1
                print("Validation loss increased, counting towards patience")

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
    print(f"Test Loss: {test_sum / test_weights:.3f}")
    test_loss = test_sum / test_weights
    wandb.log({"test_loss": test_loss, "test_perplexity": np.exp(test_loss)})
    print("Finished training")
    wandb.finish()
