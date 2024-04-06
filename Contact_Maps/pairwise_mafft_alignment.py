import sys
from Bio import SeqIO
import os
import subprocess
import io

def read_fasta(file_path):
    """Reads a FASTA file and returns a dictionary of sequences."""
    with open(file_path, 'r') as file:
        return {record.id: str(record.seq) for record in SeqIO.parse(file, "fasta")}

def run_mafft(seq1, seq2):
    """Runs MAFFT alignment on two sequences and returns the aligned second sequence."""
    # Write sequences to a temporary file
    with open("temp_input.fa", "w") as f:
        f.write(f">{seq1[0]}\n{seq1[1]}\n>{seq2[0]}\n{seq2[1]}\n")

    # Run MAFFT
    result = subprocess.run(["mafft", "--auto", "temp_input.fa"], capture_output=True, text=True, shell=True)
    
    # Use StringIO to convert string to file-like object
    aligned_seqs = list(SeqIO.parse(io.StringIO(result.stdout), "fasta"))

    # Return the aligned sequence of the second element
    return aligned_seqs[1].upper()

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py input1.fa input2.fa output.afa")
        sys.exit(1)

    file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # Read sequences from files
    seqs1 = read_fasta(file1)
    seqs2 = read_fasta(file2)

    # Find matching sequence names
    matching_names = set(seqs1.keys()).intersection(seqs2.keys())

    # Align and write to output
    with open(output_file, "w") as out_f:
        for name in matching_names:
            aligned_seq = run_mafft((name, seqs1[name]), (name, seqs2[name]))
            out_f.write(f">{name}\n{aligned_seq.seq}\n")

    # Remove temporary file
    os.remove("temp_input.fa")

if __name__ == "__main__":
    main()