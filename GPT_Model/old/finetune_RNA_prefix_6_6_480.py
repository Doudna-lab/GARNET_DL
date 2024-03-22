# finetune 23S rRNA using token library from set {single, pairs, triples, quadruples}

init_from = 'resume' # 'scratch' to init a new model from scratch, 'resume' to continue training.
device = 'cuda'
compile = True
flash = False  # Turn off scaled_dot_product_attention by default. JHDC

out_dir = '/home/ubuntu/software/jamie_rna_llm/out/' # Example for 23S_97tokens_train.bin, etc. as input.
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# finetuning might cause some issues with the val loss. Either reset best_val_loss during training or just run a defined
# number of iterations.
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = '23S'
wandb_run_name = 'triples-finetune'

# Base name for 23S_97tokens_train.bin, 23S_97tokens_val.bin, meta_23S_97tokens.pkl, for example.
pretrained = '23S_triples_0_6_6_480'
basename = '23S_thermo_triples_LM'
data_dir = '/home/ubuntu/software/jamie_rna_llm/data/'
batch_size = 36
block_size = 384 # context of up to block_size previous characters

# baby GPT model :) 
n_layer = 6
n_head =  6
n_embd = 480
dropout = 0.2

# For finetuning, use a prefix:
use_prefix_tuning = True # Set to true for fine-tuning.
prefix_length = 48 # Adjust and see what happens.

learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 1500000 # early 1277500 # 1269250 is stopping val in train_23S_triples_0_6_6_480.log
lr_decay_iters = 1500000 # make equal to max_iters usually, but not always.
min_lr = 5e-5 # learning_rate / 10 usually
beta2 = 0.998 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
