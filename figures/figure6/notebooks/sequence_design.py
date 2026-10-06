"""Greedy motif-insertion design of random sequences with Beluga for a pair of up/down tasks.

Outputs (--output_dir): motif_list_<id>.pkl, rand_filt_<id>.pkl, and per sequence
results_<id>_<i>.pkl, edited_seq_<id>_<i>.pkl, added_motifs_<id>_<i>.pkl.

Run (env: featleak-dl):
    python sequence_design.py --output_dir out --sequence_id 0 --task_id_up <idx> --task_id_down <idx>
"""

import os
import sys
#sys.path.insert(0,'/users/kobbad/feat_leakage/deepSEA_negatives')
sys.path.insert(0,'/users/kobbad/feat_leakage/all_ap1_motifs')
from model import Beluga
import torch
from ism_function import ism
import pandas as pd
from bpnetlite.io import extract_loci
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.ersatz import substitute
import numpy as np
from matplotlib import pyplot as plt
from tangermeme.ablate import ablate
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.design import greedy_substitution
import pickle
from tangermeme.predict import predict
import torch.nn
import matplotlib.pyplot as plt
from tangermeme.utils import random_one_hot
import custom_greedy
import argparse
from tangermeme.utils import characters

def get_motif_list():
    # next we need to get all the inserted motifs and their names
    annots = pd.read_excel("../ap1_counting/data/motif_annotations.xlsx",engine='openpyxl')
    archetypes = pd.read_csv("../ap1_counting/data/motifs.tsv", sep="\t")
    all_motifs = archetypes[archetypes["Motif"].isin(annots["Seed_motif"])][["Cluster_ID","Consensus"]]

    consensus_to_cluster = all_motifs.set_index("Consensus")["Cluster_ID"].to_dict()

    name_to_seq = {}

    for k,v in consensus_to_cluster.items():
        name = annots[annots.Cluster_ID == v]["Name"].values[0]

        if name=='REST/NRSF':
            continue

        name_to_seq[name] = k
    
    return [v for _,v in name_to_seq.items()]

class BelugaWrapper(torch.nn.Module):

    def __init__(self, beluga, task_idx):
        super().__init__()
        self.model = beluga
        self.task_idx = task_idx
    
    def forward(self, X):
        return self.model(X)[:,self.task_idx][:,None]

class BelugaDifferential(torch.nn.Module):

    def __init__(self, beluga, task_up, task_down):
        super().__init__()
        self.model = beluga
        self.task_up = task_up
        self.task_down = task_down
    
    def forward(self, X):
        return (self.model(X)[:,self.task_up][:,None], self.model(X)[:,self.task_down][:,None])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Sequence design script')

    parser.add_argument("--output_dir", type = str, help = "Output directory")
    parser.add_argument("--sequence_id", type = int, help = "Sequence ID")
    parser.add_argument("--task_id_down", type = int, help = "Task ID")
    parser.add_argument("--task_id_up", type = int, help = "Task ID")

    args = parser.parse_args()

    # make output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # load the model
    mdl = Beluga()
    mdl.load_state_dict(torch.load('../deepSEA_negatives/deepsea.beluga.pth')) 

    # load the motif list
    motif_list = get_motif_list()

    # save motif list to file
    motif_list_name = os.path.join(args.output_dir, f"motif_list_{args.sequence_id}.pkl")
    with open(motif_list_name, 'wb') as f:
        pickle.dump(motif_list, f)

    # wrap the model
    #wrapped_model = BelugaWrapper(mdl, args.task_id)

    wrapped_model = BelugaDifferential(mdl, args.task_id_up, args.task_id_down)

    # generate predictions and rand sequences
    rand = random_one_hot((2000, 4, 2000), random_state=0)
    predictions = predict(wrapped_model, rand)
    #rand_filt = rand[(predictions > 0).squeeze(), :, :]
    rand_filt = rand

    rand_filt_name = os.path.join(args.output_dir, f"rand_filt_{args.sequence_id}.pkl")
    with open(rand_filt_name, 'wb') as f:
        pickle.dump(rand_filt, f)

    y_target = torch.ones(1,1).to(device="cuda")
    y_target[0] = 5

    for i, seq in enumerate(rand_filt):

        seq = seq.unsqueeze(0).to(device="cuda")
        mask = torch.ones(1, dtype=bool).to(device="cuda")

        edited_seq, added_motifs, results = custom_greedy.greedy_substitution(wrapped_model,
                                                                     seq, motif_list,
                                                                     y_target, verbose=True,
                                                                     device="cuda", mask=mask, window=100)
        edited_seq = edited_seq.to("cpu").numpy()

        edited_seq_name = os.path.join(args.output_dir, f"edited_seq_{args.sequence_id}_{i}.pkl")

        results_name = os.path.join(args.output_dir, f"results_{args.sequence_id}_{i}.pkl")

        with open(results_name, 'wb') as f:
            pickle.dump(results, f)
        
        with open(edited_seq_name, 'wb') as f:
            pickle.dump(edited_seq, f)
        
        added_motifs_name = os.path.join(args.output_dir, f"added_motifs_{args.sequence_id}_{i}.pkl")

        with open(added_motifs_name, 'wb') as f:
            pickle.dump(added_motifs, f)
    






    


