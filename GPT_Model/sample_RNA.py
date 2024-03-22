"""
Sample from a trained or finetuned RNA model, using {single, pairs, triples, quadruples} tokens.
Note that the input file will have just the last block_size tokens made from it
for the decoder to then extend max_new_tokens. 

Configuration is handled at the top.
"""
import os
import time
import pickle
from contextlib import nullcontext
import torch

from model_RNA_rot import GPTConfig, GPT
print(time.ctime())
# -----------------------------------------------------------------------------
in_ckpt = '/home/ubuntu/software/jamie_rna_llm/out/231RNAs_triples_0_18_6_300_rot_flash'
meta_filename = '/home/ubuntu/software/jamie_rna_llm/data/23S_triples_meta.pkl' # Technically, should get from ckpt input, but putting it here for reference.
out_fasta = 'out/23S_from_231RNAs_triples_0_18_6_300_rot_flash_new.0.5.fasta'
start = "FILE:data/Ecoli-23S-5p-386nts.txt" # Specify a file, use as: "FILE:prompt.txt" or "FILE:prompt.fasta"
start_string = 384
num_samples = 1000 # number of samples to draw
batch_size = 50 # Make num_samples % batch_size = 0
max_new_tokens = 2600 # number of tokens generated in each sample
temperature = 0.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 97 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
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

# model
# init from a model saved in a specific directory
ckpt = f"{in_ckpt}.pt"
print(f"Reading from checkpoint {ckpt}")
checkpoint = torch.load(ckpt, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

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

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        string = ''
        for line in f:
            if line.startswith('>'):
                continue  # Skip lines starting with '>'
            # Strip the line of newline characters and add to 'start'
            string += line.strip()
            # Check if the accumulated string exceeds the desired length
            if len(string) >= start_string:
                string = string[:start_string]
                break  # Exit the loop since we've reached the desired length

# Need to choose proper tokenization scheme, depending on token library used.

start_stoi = []
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
start_stoi = tokenizer[token_number](string, stoi)

start_ids = torch.LongTensor(start_stoi)
print(f"start has {len(start_ids):,} tokens")

x = start_ids.to(device)[None, ...].clone().detach()

# Print starting and ending token of start_ids.
print("First element in start_ids:", x[0, 0].item(), "Last element in start_ids:", x[0, -1].item())

# run generation depending on tokenizer.
if token_number == 7:
    decode = lambda l: ''.join([itos[i][0] for i in l])
elif token_number == 25:
    decode = lambda l: ''.join([itos[i][0] for i in l]) # Default to using 1st character in 2-nt string.
elif token_number == 97:
    decode = lambda l: ''.join([itos[i][1] for i in l]) # Default to using middle (2nd) character in 3-nt string, change to try 3rd.
elif token_number == 385:
    decode = lambda l: ''.join([itos[i][1] for i in l]) # Default to using 2nd character in 4-nt string.
elif token_number == 175:
    decode = lambda l: ''.join([itos[i][1] for i in l]) # Default to using middle (2nd) character in 3-nt string, change to try 3rd.
else:
    print('token number not supported')
    quit()

def generate_sequences(model, x, num_samples, batch_size, max_new_tokens, temperature, top_k, decode, ctx):
    # Lists to hold generated sequences
    generated_sequences = []

    with torch.no_grad():
        with ctx:
            # Generate in batches
            for _ in range(0, num_samples, batch_size):
                y = model.generate(x.repeat(batch_size, 1), max_new_tokens, temperature=temperature, top_k=top_k)
                for sample_idx in range(y.size(0)):
                    newseq = decode(y[sample_idx].tolist())
                    generated_sequences.append(newseq)
    return generated_sequences

# Actually generate the sequences
sequences = generate_sequences(model, x, num_samples, batch_size, max_new_tokens, temperature, top_k, decode, ctx)

# Write the generated sequences to the output file
with open(out_fasta, 'w') as f:
    for k, newseq in enumerate(sequences):
        # Find the index of '3' in newseq, if present
        index_of_3 = newseq.find('3')

        # If '3' is found, keep the string up to (but not including) '3'
        if index_of_3 != -1:
            newseq = newseq[:index_of_3]

        lines = [newseq[i:i + 80] for i in range(0, len(newseq), 80)]
        begin = '>' + str(k)
        f.write(f"{begin} sampled at nt [1] using {len(itos)} token library from {len(start)} starting nts. Model number: {k}\n")
        for line in lines:
            f.write(line + '\n')
