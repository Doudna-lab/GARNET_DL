"""
Idea for finetuning. First run a regular train.py job to generate a starting model, with
a defined n_layer, etc. Then, for finetuning, we will fix the gradients for all these
layers, except for the last layers.
===========Define the variable "n_fixed" here or in config file.==================
Note that n_layer will be overridden by the model input by the checkpoint file.

Look for input and output files definition in config file.
    # Construct the input filenames
    train_filename = f"{data_dir}{basename}_train.bin"
    val_filename = f"{data_dir}{basename}_val.bin"
    meta_filename = f"{data_dir}{basename}_meta.pkl"

    # Input checkpoint file
    in_ckpt = f"{out_dir}{pretrained}.pt"
    # Output checkpoint file
    out_ckpt = f"{out_dir}{basename}_finetune_{n_fixed}_{n_layer}_{n_head}_{n_embd}_rot_flash.pt"

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
basename = '23S'
data_dir = 'data/'
out_dir = 'out/'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = '23S'
wandb_run_name = '23S_finetune' # 'run' + str(time.time())
# data
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 384
# model
n_layer = 6
n_head = 6
n_embd = 768
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

def trim_sequence(sample, beginning_tokens, end_tokens):
    '''
    This handles edge cases for >= 2 sequence segments in a sample.
    '''
    # Get the beginning and end indices
    begin_indices = torch.empty(0, dtype=torch.long)
    end_indices = torch.empty(0, dtype=torch.long)

    begin_indices = (sample.unsqueeze(1) == beginning_tokens).any(dim=1).nonzero(as_tuple=True)[0]
    end_indices = (sample.unsqueeze(1) == end_tokens).any(dim=1).nonzero(as_tuple=True)[0]

    #print("begin_indices:", begin_indices)
    #print("end_indices:", end_indices)

    # If no beginning or end tokens are found, return the whole sample
    if begin_indices.numel() == 0 and end_indices.numel() == 0:
        return sample
    
    # If there's a beginning but no end, trim everything before the beginning and return.
    # This assumes there might be some padding between entries.
    elif begin_indices.numel() > 0 and end_indices.numel() == 0:
        return sample[begin_indices[0]:]

    # If there's an end but no beginning, trim everything after the end and return.
    # This assumes there might be some padding between entries.
    elif end_indices.numel() > 0 and begin_indices.numel() == 0:
        return sample[:end_indices[0] + 1]

    # If there are 2 segment ends, then...
    elif begin_indices.shape[0] == 1 and end_indices.shape[0] == 1:
        # If there's a complete sequence in a sample, but it's smaller than the sample, return the first complete sample.
        if begin_indices[0] < end_indices[0]:
            return sample[begin_indices[0]:end_indices[0] + 1]

        # But if there's an end before a beginning, return the longer segment. 
        elif sample[begin_indices[0]:].shape[0] > sample[:end_indices[0] + 1].shape[0]:
            return sample[begin_indices[0]:]
        else:
            return sample[:end_indices[0] + 1]

    # If there are >= 3 segment ends, there has to be one complete sequence, i.e. a beginning followed by an end.
    # (b, e, b, ...)
    elif begin_indices.shape[0] >= 1 and end_indices.shape[0] >= 1 and begin_indices[0] < end_indices[0]:
        return sample[begin_indices[0]:end_indices[0] + 1]
    # (e, b, e, ...)
    elif begin_indices.shape[0] >= 1 and end_indices.shape[0] >= 1 and begin_indices[0] > end_indices[0]:
        return sample[begin_indices[0]:end_indices[1] + 1]

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
        trimmed_x = trim_sequence(sample_x, beginning_tokens, end_tokens)
        trimmed_y = trim_sequence(sample_y, beginning_tokens, end_tokens)

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

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, flash=flash) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of 97 token library for RNA triples.")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 97
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_index=pad_index)
elif init_from == 'resume':
    # Input checkpoint file name
    in_ckpt = f"{out_dir}{pretrained}.pt"
    print(f"Resuming training from {in_ckpt}")
    # resume training from a checkpoint.
    checkpoint = torch.load(in_ckpt, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model_args['n_fixed'] = n_fixed # JHDC adding this to override default in GPTConfig.
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_index=pad_index)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'bfloat16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model, backend="eager") # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Need to use find_unused_parameters=True due to fixed layers in forward pass.
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(beginning_tokens, end_tokens, pad_index):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, beginning_tokens, end_tokens, pad_index)
            with ctx:
                logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train', beginning_tokens, end_tokens, pad_index) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(beginning_tokens, end_tokens, pad_index)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                # Output checkpoint file name
                out_ckpt = f"{out_dir}{basename}_finetune_{n_fixed}_{n_layer}_{n_head}_{n_embd}_rot_flash.pt"
                print(f"saving checkpoint to {out_ckpt}")
                torch.save(checkpoint, out_ckpt)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, targets=Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', beginning_tokens, end_tokens, pad_index)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
