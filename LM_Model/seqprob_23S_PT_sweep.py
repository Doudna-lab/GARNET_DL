"""
Sample from a trained or finetuned 23S model, using {single, pairs, triples, quadruples} tokens.
Note that the input file will have just the last block_size tokens made from it
for the decoder to then extend max_new_tokens. 

Configuration is handled at the top.
"""
import os
import time
import pickle
from contextlib import nullcontext
import torch

from model_RNA_rot_v2 import GPTConfig, GPT
print(time.ctime())
# -----------------------------------------------------------------------------
in_ckpt = '/global/scratch/users/jcate/23S_triples_resume_0_18_6_300_rot_flash' #23S_triples_0_18_6_300' # Input checkpoint file name, i.e. 23S_97tokens.pt, minus the suffix, with directory.
meta_filename = '/global/home/groups-sw/pc_rnallm/jamie/Superset_triples_meta.pkl' # Technically, should get from ckpt input, but putting it here for reference.
input_fasta = "FILE:Ecoli_23S_7K00.fasta"

mask = []
batch_size = 400
top_k = 97 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Including all tokenization functions for completeness.
def rna_tokens_single(train_data_string, stoi):
    '''
    Uses a 7-token library that includes '-' for padding.
    Returns the tokenized data as a list.
    '''

    train_stoi = []
    for i in range(0, len(train_data_string)):
        w = stoi.get(train_data_string[i])
        if not (isinstance(w, int)):
            print('At position: ', i, 'w = ', w)
            print('Value of train_data_string[i] is:', train_data_string[i],'!')
            quit()
        train_stoi.append(w)

    return train_stoi

def rna_tokens_pairs(train_data_string, stoi):
    '''
    Uses a paired-nt token library that includes '--' for padding.
    Returns the tokenized data as a list. 
    '''

    train_stoi = []
    for i in range(0, len(train_data_string) - 1):
        if not '--' in train_data_string[i:i+2]:
            w = stoi.get((train_data_string[i:i+2]))
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+2] is:', train_data_string[i:i+2],'!')
                quit()
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('--')
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+2] is:', train_data_string[i:i+2],'!')
                quit()
            train_stoi.append(w)

    return train_stoi

def rna_tokens_triples(train_data_string, stoi):
    '''
    Uses a triple-nt library that includes '---' for padding, and excludes single-base ends.
    Returns the tokenized data as a list.
    '''

    train_stoi = []
    for i in range(0, len(train_data_string) - 2):
        if not '--' in train_data_string[i:i+3]:
            w = stoi.get((train_data_string[i:i+3]))
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
                quit()
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('---')
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
                quit()
            train_stoi.append(w)

    return train_stoi

def rna_tokens_quadruples(train_data_string, stoi):
    '''
    Uses a quadruple-nt token library that includes '----' for padding, and excludes '--' and '---' ends.
    Returns the tokenized data as a list 
    '''

    train_stoi = []
    for i in range(0, len(train_data_string) - 3):
        if not '--' in train_data_string[i:i+4]:
            w = stoi.get((train_data_string[i:i+4]))
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+4] is:', train_data_string[i:i+4],'!')
                quit()
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('----')
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+4] is:', train_data_string[i:i+4],'!')
                quit()
            train_stoi.append(w)
    return  train_stoi

def RNA_tokens_triples_MSA(train_data_string, stoi):
    '''
    A token library for RNA nucleotide triples that includes '---' for padding.
    The MSA version includes tokens with dashes in them, i.e. '--A' and 'A--'.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach. See:
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    train_stoi = []
    for i in range(0, len(train_data_string) - 2):
        # First convert all triples lacking a 5' or 3' end.
        if not '5' in train_data_string[i:i+3] and not '3' in train_data_string[i:i+3]:
            w = stoi.get(train_data_string[i:i+3])
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
                quit()
            train_stoi.append(w)
        #Now deal with beginning and end tokens.
        elif '-5' in train_data_string[i:i+3] or '3-' in train_data_string[i:i+3]: # Just add padding.
            # This handles '--5', '-5N', '-5-', '3--', 'N3-', and '-3-'.
            w = stoi.get('---')
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
                quit()
            train_stoi.append(w)
        elif '5' in train_data_string[i] or '3' in train_data_string[i+2]:
            # Need to handle '5--', '5-N', '5NN', '--3', 'N-3' and 'NN3' cases.
            w = stoi.get(train_data_string[i:i+3])
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
                quit()
            train_stoi.append(w)
        # Any cases I missed?
        else:    # Just add padding.
            w = stoi.get('---')
            if not (isinstance(w, int)):
                print('At position: ', i, 'w = ', w)
                print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
                quit()
            train_stoi.append(w)

    return train_stoi

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# model
# init from a model saved in a specific directory
ckpt = f"{in_ckpt}.pt"
print(f"Reading from checkpoint {ckpt}")
checkpoint = torch.load(ckpt, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])

# Set the dropout rate to 0 for inference
gptconf.dropout = 0

model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
#if compile:
#    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
load_meta = os.path.exists(meta_filename)
if load_meta:
    print(f"Loading meta from {meta_filename}...")
    with open(meta_filename, 'rb') as f:
        meta = pickle.load(f)
    # Set up token dictionaries. 
    stoi, itos = meta['stoi'], meta['itos']
    print(stoi)
    print(itos)

# Assuming stoi is in the dictionary, extract integer value for the padding token (last in the stoi dictionary).
pad_index = next((value for key, value in stoi.items() if '-' in key), None)
assert pad_index is not None, "Pad index is None!"

# encode the sequence in input_fasta.
# Variables to temporarily hold sequence and header data
sequence = ''
header = ''

if input_fasta.startswith('FILE:'):
    with open(input_fasta[5:], 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('>'):
                header += line.strip() # Keep header lines
                continue
            sequence += line.strip()

# Generate all possible singl-nt variants of input sequence.
nucleotides = ['A', 'C', 'G', 'U']
all_sequences = [sequence]  # Start with the original sequence
headers = ['WT']  # Header for the original sequence

# Generate modified sequences
for i, original_nuc in enumerate(sequence):
    position = i + 1  # Convert from 0-indexed to 1-indexed position
    for nuc in nucleotides:
        if nuc != original_nuc:  # Exclude the original nucleotide at this position
            # Create a new sequence with the replacement
            new_seq = sequence[:i] + nuc + sequence[i+1:]
            all_sequences.append(new_seq)
            # Create a header indicating the position of the change and the new nucleotide
            header = f"{position}{nuc}"
            headers.append(header)

# Need to choose proper tokenization scheme, depending on token library used.
input_stoi = []
token_number = len(stoi)

# Create a dictionary that maps token_number to functions
tokenizer = {
    7: rna_tokens_single,
    25: rna_tokens_pairs,
    97: rna_tokens_triples, 
    385: rna_tokens_quadruples,
    175: RNA_tokens_triples_MSA,
}

# Choose the string tokenizer function based on len(stoi)
input_stoi = [tokenizer[token_number](seq, stoi) for seq in all_sequences]

# All the sequences are the same length, so it should be possible to convert input_stoi into a tensor directly.
input_ids = torch.LongTensor(input_stoi)
print(f"input has {len(input_ids):,} sequences and {input_ids.size(1)} tokens per sequence")

def process_batches(model, input_ids, mask, batch_size, headers):
    """
    Process input_ids in batches through a model.

    Parameters:
    - model: The decoder model to process input sequences.
    - input_ids: Tensor of tokenized sequences to be processed.
    - mask: List of positions to mask (Needs work before it can be used. JHDC)
    - batch_size: Size of batches to divide input_ids into.
    - headers: List of headers for each sequence.

    Prints:
    - header and log_prob for each sequence.

    Returns:
    - log_probs for each sequence in input_ids.
    """
    num_sequences = input_ids.size(0)
    log_probs = []

    print("Sequence Number\tLog Probability\tHeader")

    for start_idx in range(0, num_sequences, batch_size):
        end_idx = start_idx + batch_size
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_headers = headers[start_idx:end_idx]

        # Calculate the log probability of each sequence coming from the model.
        x = batch_input_ids.to(device).clone().detach()
        model.eval()
        batch_log_probs = model.sequence_probability(x)

        # Store the output for this batch
        log_probs.append(batch_log_probs)

        # Print out the results.
        batch_log_probs = batch_log_probs.cpu().numpy()  # Move to CPU and convert to a numpy array for easy manipulation
        for i, (header, log_prob) in enumerate(zip(batch_headers, batch_log_probs), start=1):  # Pair each header with its log_prob
            print(f"{i+start_idx}\t{log_prob:.4f}\t{header}",flush=True)

    return log_probs 

log_probs = process_batches(model, input_ids, mask, batch_size, headers)

# Print out the results.
#log_probs = log_probs.cpu().numpy()  # Move log_probs to CPU and convert to a numpy array for easy manipulation

#print("Header\tSequence Number\tLog Probability")
#for i, (header, log_prob) in enumerate(zip(headers, log_probs), start=1):  # Pair each header with its log_prob
#    print(f"{i}\t{log_prob:.4f}\t{header[13:]}")

