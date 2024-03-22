import os
import sys
import numpy as np
import torch
import pickle
import argparse
import distutils.util
import gzip

def rna_AUGCgap_only(string):
    ''' 
    Clean up RNA fasta and SILVA "fasta" formatting. Leave '=' as only
    remaining non-RNA character that could be used as padding later.
    Replace the rare "Purine" R with A and "Pyrimidine" Y with U.
    Replace K,B,H with U; M,S with C; Z,N,W,V,D,n with A. 
    Remove newline characters.
    Also replace spaces from SILVA, but keep dashes for MSA.
    '''
    for char in ['R', 'N', 'W', 'V', 'D', 'Z', 'n', 'a']:
        string = string.replace(char, 'A')
    for char in ['Y', 'K', 'B', 'H', 'u', 'T']:
        string = string.replace(char, 'U')
    for char in ['M', 'S', 'c']:
        string = string.replace(char, 'C')
    string = string.replace('g', 'G')
    for char in ['\n', ' ', '.', ]: # Keep '-' in string for MSA.
        string = string.replace(char, '')

    return(string)

def RNA_tokenize_single(train_data_string):
    '''
    A 7-token library for RNA nucleotides that includes '-' for padding.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach. See:
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    chars = ['A', 'U', 'G', 'C']
    chars += ['5', '3', '-']

    vocab_size = len(chars)
    print("all the unique characters:", ' '.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    print(stoi)
    print(itos)

    train_stoi = []
    for i in range(0, len(train_data_string)):
        w = stoi.get(train_data_string[i])
        assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i] is: {train_data_string[i]}!'
        #if not (isinstance(w, int)):
        #    print('At position: ', i, 'w = ', w)
        #    print('Value of train_data_string[i] is:', train_data_string[i],'!')
        #    quit()
        train_stoi.append(w)

    return stoi, itos, train_stoi

def RNA_tokenize_pairs(train_data_string):
    '''
    A 25-token library for RNA nucleotide pairs that includes '--' for padding.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach. See:
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    nucleotides = ['A', 'U', 'G', 'C']

    # Generate all possible pairs from nucleotides list
    chars = [a+b for a in nucleotides for b in nucleotides]

    # Now add the combinations with '5' in the first position and '3' in the second.
    chars.extend(['5' + b for b in nucleotides])
    chars.extend([a + '3' for a in nucleotides])
    chars += ['--']

    vocab_size = len(chars)
    print("all the unique characters:", ' '.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    print(stoi)
    print(itos)

    train_stoi = []
    for i in range(0, len(train_data_string) - 1):
        if not '--' in train_data_string[i:i+2] and not '-5' in train_data_string[i:i+2] and not '3-' in train_data_string[i:i+2]:
            w = stoi.get((train_data_string[i:i+2]))
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+2] is: {train_data_string[i:i+2]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+2] is:', train_data_string[i:i+2],'!')
            #    quit()
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('--')
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+2] is: {train_data_string[i:i+2]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+2] is:', train_data_string[i:i+2],'!')
            #    quit()
            train_stoi.append(w)

    return stoi, itos, train_stoi

def RNA_tokenize_triples(train_data_string):
    '''
    A token library for RNA nucleotide triples that includes '---' for padding and excludes double-dash ends like '--A' and 'A--'.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach. See:
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    nucleotides = ['A', 'U', 'G', 'C']

    # Generate all possible triples from nucleotides list
    chars = [a+b+c for a in nucleotides for b in nucleotides for c in nucleotides]

    # Add combinations with '5' in the first position and '3' in the third position.
    chars.extend(['5' + b + c for b in nucleotides for c in nucleotides])
    chars.extend([a + b + '3' for a in nucleotides for b in nucleotides])

    chars += ['---']

    vocab_size = len(chars)
    print("all the unique characters:", ' '.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    print(stoi)
    print(itos)

    train_stoi = []
    for i in range(0, len(train_data_string) - 2):
        if not '--' in train_data_string[i:i+3] and not '-5' in train_data_string[i:i+2] and not '3-' in train_data_string[i+1:i+3]:
            w = stoi.get((train_data_string[i:i+3]))
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+3] is: {train_data_string[i:i+3]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
            #    quit()
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('---')
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+3] is: {train_data_string[i:i+3]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
            #    quit()
            train_stoi.append(w)

    return stoi, itos, train_stoi

def RNA_tokenize_quadruples(train_data_string):
    '''
    A token library for RNA nucleotide quadruples that includes '----' for padding and excludes multiple-dash ends like '---A' and 'A---'.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach. See:
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    nucleotides = ['A', 'U', 'G', 'C']

    # Generate all possible quadruples from nucleotides list
    chars = [a+b+c+d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]

    # Add combinations with '5' in the first position and '3' in the third position.
    chars.extend(['5' + b + c + d for b in nucleotides for c in nucleotides for d in nucleotides])
    chars.extend([a + b + c + '3' for a in nucleotides for b in nucleotides for c in nucleotides])

    chars += ['----']

    vocab_size = len(chars)
    print("all the unique characters:", ' '.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    print(stoi)
    print(itos)

    train_stoi = []
    for i in range(0, len(train_data_string) - 3):
        if not '--' in train_data_string[i:i+4] and not '-5' in train_data_string[i:i+3] and not '3-' in train_data_string[i+1:i+4]:
            w = stoi.get((train_data_string[i:i+4]))
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+4] is: {train_data_string[i:i+4]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+4] is:', train_data_string[i:i+4],'!')
            #    quit()
            train_stoi.append(w)
        else:    # Just add padding.
            w = stoi.get('----')
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+4] is: {train_data_string[i:i+4]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+4] is:', train_data_string[i:i+4],'!')
            #    quit()
            train_stoi.append(w)

    return stoi, itos, train_stoi

def RNA_tokenize_triples_MSA(train_data_string):
    '''
    A token library for RNA nucleotide triples that includes '---' for padding.
    The MSA version includes tokens with dashes in them, i.e. '--A' and 'A--'.
    Returns the tokenized data as a list and the stoi and itos token libraries.
    Adding <BOS> and <EOS> tokens, per ESM-2 model approach. See:
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full
    I will use '5' and '3' to denote these, for ease of coding.
    '''

    nucleotides = ['A', 'U', 'G', 'C']

    # Generate all possible triples from nucleotides list
    chars = [a+b+c for a in nucleotides for b in nucleotides for c in nucleotides]

    # Add combinations with '5' in the first position and '3' in the third position.
    chars.extend(['5' + b + c for b in nucleotides for c in nucleotides])
    chars.extend([a + b + '3' for a in nucleotides for b in nucleotides])

    # Add padding token.
    chars += ['---']

    # Add combinations with dashes for MSA after all tokens in RNA_tokenize_triples.
    chars.extend(['-' + b + c for b in nucleotides for c in nucleotides])
    chars.extend([a + '-' + c for a in nucleotides for c in nucleotides])
    chars.extend([a + b + '-' for a in nucleotides for b in nucleotides])

    chars.extend(['--' + c for c in nucleotides])
    chars.extend(['-' + b + '-' for b in nucleotides])
    chars.extend([a + '--' for a in nucleotides])

    # Include 5' and 3' tokens with dashes. Use when 5 at pos[0] of token, or 3 at pos[2] of token.
    chars.extend(['5-' + c for c in nucleotides])
    chars.extend(['5' + b + '-' for b in nucleotides])
    chars.extend([a + '-3' for a in nucleotides])
    chars.extend(['-' + b + '3' for b in nucleotides])
    chars += ['5--', '--3']

    vocab_size = len(chars)
    print("all the unique characters:", ' '.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()} # inverse mapping

    print(stoi)
    print(itos)

    train_stoi = []
    for i in range(0, len(train_data_string) - 2):
        # First convert all triples lacking a 5' or 3' end.
        if not '5' in train_data_string[i:i+3] and not '3' in train_data_string[i:i+3]:
            w = stoi.get(train_data_string[i:i+3])
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+3] is: {train_data_string[i:i+3]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
            #    quit()
            train_stoi.append(w)
        #Now deal with beginning and end tokens.
        elif '-5' in train_data_string[i:i+3] or '3-' in train_data_string[i:i+3]: # Just add padding.
            # This handles '--5', '-5N', '-5-', '3--', 'N3-', and '-3-'.
            w = stoi.get('---')
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+3] is: {train_data_string[i:i+3]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
            #    quit()
            train_stoi.append(w)
        elif '5' in train_data_string[i] or '3' in train_data_string[i+2]:
            # Need to handle '5--', '5-N', '5NN', '--3', 'N-3' and 'NN3' cases.
            w = stoi.get(train_data_string[i:i+3])
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+3] is: {train_data_string[i:i+3]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
            #    quit()
            train_stoi.append(w)
        # Any cases I missed?
        else:    # Just add padding.
            w = stoi.get('---')
            assert isinstance(w, int), f'At position: {i}, expected w to be an int but got {w}. Value of train_data_string[i:i+3] is: {train_data_string[i:i+3]}!'
            #if not (isinstance(w, int)):
            #    print('At position: ', i, 'w = ', w)
            #    print('Value of train_data_string[i:i+3] is:', train_data_string[i:i+3],'!')
            #    quit()
            train_stoi.append(w)

    return stoi, itos, train_stoi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This reads in a gzipped .fasta file and outputs tokenized RNA data for training.')
    parser.add_argument('-i', required=True, help='Input file name (gzipped fasta file format only!)')
    parser.add_argument('-o', required=True, help='Output file base name')
    parser.add_argument('--type', type=str, choices=['single', 'pairs', 'triples', 'quadruples', 'triples_MSA'], required=True, 
                        help="Token type from set ['single', 'pairs', 'triples', 'quadruples', 'triples_MSA']")
    args = parser.parse_args()

    if args.i == "" or args.o == "":
        parser.print_help()
        sys.exit()
        
    # Create a dictionary that maps type strings to functions
    tokenizer = {
        'single': RNA_tokenize_single,
        'pairs': RNA_tokenize_pairs,
        'triples': RNA_tokenize_triples,
        'quadruples': RNA_tokenize_quadruples,
        'triples_MSA': RNA_tokenize_triples_MSA,
    }

    # Construct the output filenames
    train_filename = f"{args.o}_train.bin"
    meta_filename = f"{args.o}_meta.pkl"

    # Read in protein sequences.
    input_file_path_train = os.path.join(os.path.dirname(__file__), args.i)
    if not os.path.exists(input_file_path_train):
        print(f"{input_file_path_train} does not exist.")
        quit()

    with gzip.open(input_file_path_train, 'rt') as f:
        train_data = f.read()
    train_data = train_data.split('\n')

    # Convert string into a usable form for tokenization, i.e. only 4 natural RNA nts + '-', '5', '3'.
    train_data_string = ''

    #          1234567890
    padding = '==========' # Eventually use this as the padding!

    train_data[0] = padding + '5'
    for x in range(1, len(train_data) - 1):
        if (train_data[x].startswith('>')):
            train_data[x] = '3' + padding + '5'
    train_data.append('3')

    # Replace the rare "Purine" R with A and "Pyrimidine" Y with U.
    # Replace K,B,H with U; M,S with C; Z,N,W,V,D,n with A. Remove newline characters.
    # Also replace dots and spaces from compressed alignment with ''. Keep dashes for MSA.
    train_data_string = ''.join([str(elem) for i,elem in enumerate(train_data)])
    train_data_string = rna_AUGCgap_only(train_data_string)

    # Deal with padding separately, in case we want to change our approach later.
    train_data_string = train_data_string.replace('=', '-')

    # Make a simple token encoding here.
    # Choose the function based on args.type
    stoi, itos, train_stoi = tokenizer[args.type](train_data_string)

    # export to bin files
    train_ids = np.array(train_stoi, dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    train_ids.tofile(os.path.join(os.path.dirname(__file__), train_filename))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': len(itos),
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), meta_filename), 'wb') as f:
        pickle.dump(meta, f)
