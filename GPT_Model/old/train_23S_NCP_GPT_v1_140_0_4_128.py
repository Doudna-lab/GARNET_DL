# train 23S rRNA using token library from set {5, 25, 97, 385, 97-1skip}

init_from = 'scratch' # 'scratch' to init a new model from scratch, 'resume' to continue training.
device = 'cuda'
compile = False
flash = False  # Turn off scaled_dot_product_attention by default. JHDC

out_dir = 'out/' # Example for 23S_97tokens_train.bin, etc. as input.
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = '23S_NCP_GPT'
wandb_run_name = '23S_NCP_GPT'

# Base name for 23S_97tokens_train.bin, 23S_97tokens_val.bin, meta_23S_97tokens.pkl, for example.
basename = '23S_triples_GNN'
data_dir = '/home/ubuntu/software/jamie_rna_llm/data/'
batch_size = 60
block_size = 384 # context of up to block_size previous characters

# baby GPT model :) Set 0 < n_cfc <= n_layer for cfc component.
n_cfc = 0
n_layer = 2
n_head =  4
n_embd = 128
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 250000
lr_decay_iters = 50000 # make equal to max_iters usually, but not always.
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.998 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# Flags for CfC, some probably redundant with the above. Consolidate later, but make as close as possible.
in_features = 128 # Try a smaller model.
units = 140 # Has to be at least out_features + 2?
out_features = 128 # Make n_embd in the future?
use_mixed = False # These 3 are set so that we can use CFC_MIXED (i.e. CfC + LSTM)
no_gate = False
minimal = False

