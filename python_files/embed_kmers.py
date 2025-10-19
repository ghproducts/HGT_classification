import json
import time
import os
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
import itertools
import hashlib

from pathlib import Path
from Bio import SeqIO
from collections import Counter

fasta_extensions = (".fasta", ".faa")
dna_alphabet = "ATCG"
protein_alphabet = "ARNDCEQGHILKMFPSTWYV" 

def extract_fasta(fname):
    ids, seqs = [], []
    seen = set()

    for record in SeqIO.parse(fname, "fasta"):
        s = str(record.seq).upper()
        key = (len(s), hashlib.blake2b(s.encode('ascii'), digest_size=16).digest()) # remove repeated sequences if they are exact
        if key in seen:
            continue
        seen.add(key)
        ids.append(str(record.id))
        seqs.append(s)

    return ids, seqs
    

def count_kmer(seq: str, k: int, alphabet: str, valid_kmers: list):
    if len(seq) < k:
        return {}

    s = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)

    valid = np.zeros(256, dtype=bool)
    for ch in alphabet:
        valid[ord(ch)] = True

    win = np.lib.stride_tricks.sliding_window_view(s, k)

    mask = valid[win].all(axis=1)
    if not mask.any():
        return {}

    win = win[mask]
    view = win.view(f"V{k}")
    uniq, counts = np.unique(view, return_counts=True)
    kmers = uniq.view(f"S{k}").astype(str)
    present = dict(zip(kmers.tolist(), counts.tolist()))
    return {km: present.get(km, 0) for km in valid_kmers}


def main(args):
    # find fasta files
    if os.path.isfile(args.input):
       fasta_files = [args.input]
    else:
        fasta_files = sorted(os.path.join(args.input, fname) for fname in os.listdir(args.input) if fname.lower().endswith(fasta_extensions))
    print(f"Input file: {fasta_files[:5]}")

    start = time.time()
    
    # extract sequences
    ids = []
    seqs = []
    for file in fasta_files:
        one_id, one_seq = extract_fasta(fasta_files[0])
        ids += one_id
        seqs += one_seq
    print(f"Total sequences extracted: {len(ids)}")

    # create pandas table with all permutations
    all_permutations = itertools.product(protein_alphabet, repeat = args.kmer_length)
    all_permutations = [''.join(kmer) for kmer in all_permutations]
    print(f"Total permutations: {len(all_permutations)}")

    # count kmers in each sequence
    kmers = {}
    for idx, seq in enumerate(seqs):
        seq_id = ids[idx]
        kmers[seq_id] = count_kmer(seq, args.kmer_length, protein_alphabet, all_permutations)

    # check for repeatsi
    #matching_keys = {}
    #for seq_id, target_value in kmers.items():
    #    matching_keys[seq_id]=[key for key, value in kmers.items() if value == target_value and key != seq_id]
    #with open("matching_keys.json", "w") as json_file:
    #    json.dump(matching_keys, json_file, indent=4)

    print(f"Counted kmers in {time.time() - start:.3f} seconds") 
    valid_kmers = all_permutations  # desired column order
    df = pd.DataFrame.from_dict(kmers, orient="index")
    df = df.reindex(columns=valid_kmers, fill_value=0).astype(np.int32)
    print(df.head(5))

    # combine and output file
    out = Path(args.output)
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"kmer_counts_k{args.kmer_length}.csv"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_file = out

    df.to_csv(out_file, index=True, index_label="ID")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fasta file Kmer counter with a fixed length")
    parser.add_argument("--input", type=str, required = True)
    parser.add_argument("--output", type=str, required = True)
    parser.add_argument("--kmer_length", type = int, default=2)
    parser.add_argument("--num_cores", type = int, default = 1)
    args=parser.parse_args()
    main(args)
