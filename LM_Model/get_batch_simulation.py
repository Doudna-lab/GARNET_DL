"""
Look for input and output files definition in config file.
    # Construct the input filenames
    train_filename = f"{data_dir}{basename}_train.bin"
    val_filename = f"{data_dir}{basename}_val.bin"
    meta_filename = f"{data_dir}{basename}_meta.pkl"

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_RNA_rot import GPTConfig, GPT

# Disabling P2P may be required to get multiGPU training to work. JHDC
os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_IB_DISABLE"] = "1"
# -----------------------------------------------------------------------------
# default config values designed to train on Pfam sequences.
# Can override with a configuration file in the ./config directory.
# I/O
basename = 'Superset_triples'
#basename = '23S_triples'
data_dir = '/global/home/groups-sw/pc_rnallm/jamie/'
out_dir = '/global/home/groups-sw/pc_rnallm/jamie/'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = '23S'
wandb_run_name = '23S' # 'run' + str(time.time())
# data
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 18 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 384
# model
n_layer = 18
n_head = 6
n_embd = 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 3e-5 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 3e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
print(time.ctime())
print('grad_clip=', grad_clip)
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
#    gradient_accumulation_steps *= 1 # simulate 8 gpus

if master_process:
    os.makedirs(out_dir, exist_ok=True)
# Default seed. 
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
#if (device_type == 'cpu'):
#    ctx = nullcontext()
#elif ('cuda' in device_type):
#    ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16), torch.backends.cuda.sdp_kernel(enable_flash=False)

# poor man's data loader
# Construct the input filenames
train_filename = f"{data_dir}{basename}_train.bin"
val_filename = f"{data_dir}{basename}_val.bin"
meta_filename = f"{data_dir}{basename}_meta.pkl"

train_data = np.memmap(train_filename, dtype=np.uint16, mode='r')
val_data = np.memmap(val_filename, dtype=np.uint16, mode='r')

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = meta_filename
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
    print(f"Error: {meta_path} not found!")
    sys.exit(1)  # Exit with a non-zero status to indicate an error

stoi = meta['stoi']

# Assuming stoi is in the dictionary, extract integer values for beginning and end tokens.
beginning_tokens = torch.tensor([value for key, value in stoi.items() if key.startswith('5')])
end_tokens = torch.tensor([value for key, value in stoi.items() if key.endswith('3')])
pad_index = next((value for key, value in stoi.items() if '-' in key), None)
assert pad_index is not None, "Pad index is None!"

# Print the values of beginning_tokens and end_tokens
print("Beginning tokens (values):", beginning_tokens.tolist())
print("End tokens (values):", end_tokens.tolist())
print("Pad index (value):", pad_index)

def trim_sequence(sample_x, sample_y, beginning_tokens, end_tokens):
    '''
    This handles edge cases for >= 2 sequence segments in a sample.
    '''
    # Get the beginning and end indices
    begin_indices = torch.empty(0, dtype=torch.long)
    end_indices = torch.empty(0, dtype=torch.long)

    begin_indices = (sample_x.unsqueeze(1) == beginning_tokens).any(dim=1).nonzero(as_tuple=True)[0]
    end_indices = (sample_x.unsqueeze(1) == end_tokens).any(dim=1).nonzero(as_tuple=True)[0]

    #print("begin_indices:", begin_indices)
    #print("end_indices:", end_indices)

    # If no beginning or end tokens are found, return the whole sample
    if begin_indices.numel() == 0 and end_indices.numel() == 0:
        return sample_x, sample_y
    
    # If there's a beginning but no end, trim everything before the beginning and return.
    # This assumes there might be some padding between entries.
    elif begin_indices.numel() > 0 and end_indices.numel() == 0:
        return sample_x[begin_indices[0]:], sample_y[begin_indices[0]:]

    # If there's an end but no beginning, trim everything after the end and return.
    # This assumes there might be some padding between entries.
    elif end_indices.numel() > 0 and begin_indices.numel() == 0:
        return sample_x[:end_indices[0] + 1], sample_y[:end_indices[0] + 1]

    # If there are 2 segment ends, then...
    elif begin_indices.shape[0] == 1 and end_indices.shape[0] == 1:
        # If there's a complete sequence in a sample, but it's smaller than the sample, return the first complete sample.
        if begin_indices[0] < end_indices[0]:
            return sample_x[begin_indices[0]:end_indices[0] + 1], sample_y[begin_indices[0]:end_indices[0] + 1]

        # But if there's an end before a beginning, return the longer segment. 
        elif sample_x[begin_indices[0]:].shape[0] > sample_x[:end_indices[0] + 1].shape[0]:
            return sample_x[begin_indices[0]:], sample_y[begin_indices[0]:]
        else:
            return sample_x[:end_indices[0] + 1], sample_y[:end_indices[0] + 1]

    # If there are >= 3 segment ends, there has to be one complete sequence, i.e. a beginning followed by an end.
    # (b, e, b, ...)
    elif begin_indices.shape[0] >= 1 and end_indices.shape[0] >= 1 and begin_indices[0] < end_indices[0]:
        return sample_x[begin_indices[0]:end_indices[0] + 1], sample_y[begin_indices[0]:end_indices[0] + 1]
    # (e, b, e, ...)
    elif begin_indices.shape[0] >= 1 and end_indices.shape[0] >= 1 and begin_indices[0] > end_indices[0]:
        return sample_x[begin_indices[0]:end_indices[1] + 1], sample_y[begin_indices[0]:end_indices[1] + 1]

def get_batch(split, beginning_tokens, end_tokens, pad_index):
    '''
    This version of get_batch eliminates extraneous tokens at the beginning and end of an entry.
    It also reindexes an entry to start at an index of 0, with trailing padding tokens = pad_index.
    This version requires inputting a list of all beginning and all end tokens in a vocabulary.
    '''

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x_list, y_list = [], []
    for i in ix:
        sample_x = torch.from_numpy((data[i:i+block_size]).astype(np.int64))
        sample_y = torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))

        # Use the trimming function to trim the sequences
        trimmed_x, trimmed_y = trim_sequence(sample_x, sample_y, beginning_tokens, end_tokens)

        # Add padding to the trimmed sequences to make them block_size in length
        if trimmed_x.shape[0] < block_size:
            padding_length_x = block_size - trimmed_x.shape[0]
            trimmed_x = F.pad(trimmed_x, (0, padding_length_x), value=pad_index)
    
        if trimmed_y.shape[0] < block_size:
            padding_length_y = block_size - trimmed_y.shape[0]
            trimmed_y = F.pad(trimmed_y, (0, padding_length_y), value=pad_index)

        x_list.append(trimmed_x)
        y_list.append(trimmed_y)

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    #print(x[:,304:384])
    #print(y[:,304:384])

    #sys.exit()

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# Initialize variables
num_batches = 10000
incorrect_matches = []  # List to store all cases where x[b,:] = y[b,:]
correct_matches = [] # List of all cases where x[b,1:] = y[b,:-1]

# Run get_batch 10000 times
for batch_idx in range(num_batches):
    x, y = get_batch('train', beginning_tokens, end_tokens, pad_index)
    # Slice to exclude the last element along dim=1
    x_trimmed = x[:, :-1]
    y_trimmed = y[:, :-1]
    # Check row-wise equality for the trimmed tensors
    row_incorrect_matches = torch.where(torch.all(x_trimmed == y_trimmed, dim=1))[0]  # Indices where rows are incorrectly equal
    row_correct_matches = torch.where(torch.all(x[:, 1:187] == y[:, :186], dim=1))[0]  # Indices where rows are correctly equal
    if len(row_incorrect_matches) > 0:
        # Store the batch index and matching row indices
        incorrect_matches.append((batch_idx, row_incorrect_matches.tolist()))
    correct_matches.append((batch_idx, row_correct_matches.tolist()))

    # Convert to sets for efficient computation
    matched_rows = set(row_incorrect_matches.tolist()) | set(row_correct_matches.tolist())
    all_rows = set(range(x.shape[0]))
    unmatched_rows = all_rows - matched_rows

    # Print the unmatched rows and their x, y values
    #if unmatched_rows:
    #    unmatched_rows = sorted(list(unmatched_rows))  # Sort for consistent output
    #    print(f"Batch {batch_idx} had unmatched rows: {unmatched_rows}")
    #    for row in unmatched_rows:
    #        print(f"Row {row}: x = {x[row,:].tolist()}, y = {y[row,:].tolist()}")

# Print the results
for batch_idx, row_indices in incorrect_matches:
    print(f"Batch {batch_idx} had incorrect matches in rows: {row_indices}")

# Summary
total_incorrect_matches = sum(len(rows) for _, rows in incorrect_matches)
total_correct_matches = sum(len(rows) for _, rows in correct_matches)
print(f"Out of {num_batches} batches, there were {total_incorrect_matches} total incorrectly matching rows x[b,:-1] = y[b,:-1].")
print(f"Out of {num_batches} batches, there were {total_correct_matches} total correctly matching rows x[b,1:187] = y[b,:186].")
print(f"Note, the dimensions for x[b,1:187] = y[b,:186] reflect a padding of 10 between sequences, and choosing the longest seq.")

sys.exit()
