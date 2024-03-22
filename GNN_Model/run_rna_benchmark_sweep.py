import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g", required=True)
parser.add_argument("--seed", "-s", type=int, required=True)
args = parser.parse_args()


for hidden_dim in [64, 128]:
    for k_nbrs in [5, 10, 20, 50, 100]:
        subprocess.run(f"CUDA_VISIBLE_DEVICES={args.gpu} python benchmark_models.py --test_data jan_2023_23s_data/new_23s_test.pt --seed {args.seed} --k_nbrs {k_nbrs} --hidden_dim {hidden_dim} --finetune_path log/jan_19_rna_hidden_dim_{hidden_dim}_k_nbrs_{k_nbrs}/models/checkpoint_19.pt", shell=True)
