import argparse
import os
import pickle
import pandas as pd
from tqdm import tqdm
from utils.utils import load_dataset

def create_kmer_dict(X, k, kmer_dict=None):
    if kmer_dict is None:
        kmer_dict = {}
    idx = len(kmer_dict)
    for seq in X:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer not in kmer_dict:
                kmer_dict[kmer] = idx
                idx += 1
    return kmer_dict

def m_neighbours(kmer, m):
    letters = ['A', 'C', 'G', 'T']
    results = {kmer}
    for _ in range(m):
        new_results = set(results)
        for s in results:
            for i in range(len(kmer)):
                for l in letters:
                    if s[i] != l:
                        new_results.add(s[:i] + l + s[i+1:])
        results = new_results
    return list(results)

def get_neighbours(kmer_dict, m):
    kmers_list = list(kmer_dict.keys())
    neighbours = {kmer: [] for kmer in kmers_list}
    for kmer in tqdm(kmers_list, desc="Processing kmers"):
        for neighbour in m_neighbours(kmer, m):
            if neighbour in kmer_dict:
                neighbours[kmer].append(neighbour)
    return neighbours

def main():
    parser = argparse.ArgumentParser(description="Generate kmer and neighbour cache files.")
    parser.add_argument("--k", type=int, required=True, help="Length of the kmer")
    parser.add_argument("--m", type=int, required=True, help="Number of mismatches allowed")
    parser.add_argument("--datasets", type=int, nargs='+', default=[0, 1, 2],
                        help="List of dataset IDs to process")
    parser.add_argument("--output_dir", type=str, default="saved_dictionnaries",
                        help="Directory to save output cache files")
    args = parser.parse_args()

    k = args.k
    m = args.m
    # Ensure k is greater than m.
    assert k > m, "Parameter 'k' must be greater than 'm'."

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for dataset_id in args.datasets:
        Xtrk, Xtek, _ = load_dataset(dataset_id)
        X_k = pd.concat([Xtrk['seq'], Xtek['seq']]).reset_index(drop=True)
        kmer_dict = create_kmer_dict(X_k, k)
        neighbors = get_neighbours(kmer_dict, m)

        kmer_file = os.path.join(args.output_dir, f"kmers_dict_data{dataset_id}_k{k}_m{m}.pkl")
        neighbors_file = os.path.join(args.output_dir, f"neighbors_dict_data{dataset_id}_k{k}_m{m}.pkl")
        with open(kmer_file, 'wb') as f:
            pickle.dump(kmer_dict, f)
        with open(neighbors_file, 'wb') as f:
            pickle.dump(neighbors, f)

if __name__ == "__main__":
    main()
