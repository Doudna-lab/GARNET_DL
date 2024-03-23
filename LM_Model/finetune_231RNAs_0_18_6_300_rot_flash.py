# finetune 23S rRNA using token library from set {single, pairs, triples, quadruples}

init_from = 'resume' # 'scratch' to init a new model from scratch, 'resume' to continue training.
device = 'cuda'
compile = True
flash = True  # Turn off scaled_dot_product_attention by default. JHDC

out_dir = '/home/ubuntu/software/jamie_rna_llm/out/' # Example for 23S_97tokens_train.bin, etc. as input.
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# finetuning might cause some issues with the val loss. Either reset best_val_loss during training or just run a defined
# number of iterations.
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = '231RNAs'
wandb_run_name = 'triples-finetune'

# Base name for 23S_97tokens_train.bin, 23S_97tokens_val.bin, meta_23S_97tokens.pkl, for example.
pretrained = '231RNAs_triples_0_18_6_300_rot_flash'
basename = '231RNAs_Thermo_triples'
data_dir = '/home/ubuntu/software/jamie_rna_llm/data/'
batch_size = 48
block_size = 384 # context of up to block_size previous characters

# baby GPT model :) Set 0 < n_fixed <= n_layer for finetuning.
n_fixed = 0
n_layer = 18
n_head =  6
n_embd = 300
dropout = 0.2

learning_rate = 2e-5 # with baby networks can afford to go a bit higher
max_iters = 1200000 # early 1277500 # 1269250 is stopping val in train_23S_triples_0_6_6_480.log
lr_decay_iters = 50000 # make equal to max_iters usually, but not always.
min_lr = 2e-6 # learning_rate / 10 usually
beta2 = 0.998 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
