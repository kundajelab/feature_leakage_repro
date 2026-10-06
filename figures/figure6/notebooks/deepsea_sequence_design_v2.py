import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-fbdc2a7b-c9f5-5e02-17f9-4e00342e866c'
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
from tangermeme.marginalize import marginalize
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.design import greedy_substitution
import pickle
from tangermeme.predict import predict
import torch.nn
import matplotlib.pyplot as plt
from tangermeme.utils import random_one_hot
import custom_greedy
import argparse
from tqdm import tqdm
import importlib
import custom_greedy

importlib.reload(custom_greedy)

from custom_greedy import greedy_substitution

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

deepsea_task_path = "/users/kobbad/feat_leakage/all_ap1_motifs/deepsea_beluga_names.txt"

with open(deepsea_task_path, 'r') as f:
    deepsea_task_names = f.read().splitlines()

mdl = Beluga()
mdl.load_state_dict(torch.load('../deepSEA_negatives/deepsea.beluga.pth'))

annots = pd.read_excel("../ap1_counting/data/motif_annotations.xlsx",engine='openpyxl')
archetypes = pd.read_csv("../ap1_counting/data/motifs.tsv", sep="\t")

all_motifs = archetypes[archetypes["Motif"].isin(annots["Seed_motif"])][["Cluster_ID","Consensus", "Motif"]]

consensus_to_cluster = all_motifs.set_index("Consensus")["Motif"].to_dict()
motif_to_seq = {v:k for k,v in consensus_to_cluster.items()}

target_first = "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02_DNase.all.peaks_None" 
target_second = 'Penis_Foreskin_Melanocyte_Primary_Cells_skin01_DNase.all.peaks_None' 

task_id_up = deepsea_task_names.index(target_first)
task_id_down = deepsea_task_names.index(target_second)

wrapped_model = BelugaDifferential(mdl, task_id_up, task_id_down)

# generate predictions and rand sequences
rand = random_one_hot((500, 4, 2000), random_state=0)
predictions = predict(wrapped_model, rand)
y_target = torch.ones(1,1).to(device="cuda")
y_target[0] = 6

motif_names = list(motif_to_seq.keys())
motifs = [motif_to_seq[name] for name in motif_names]

all_results = []

for i, seq in tqdm(enumerate(rand), total=len(rand)):

    seq = seq.unsqueeze(0).to(device="cuda")
    mask = torch.ones(1, dtype=bool).to(device="cuda")

    edited_seq, added_motifs, results, motif_names_added = custom_greedy.greedy_substitution(wrapped_model,
                                                                    seq, motifs,
                                                                    y_target, verbose=True,
                                                                    device="cuda", mask=mask, window=100,
                                                                    motif_names=motif_names)
    
    results['motif_names_added'] = motif_names_added
    all_results.append(results)

    # save all results to pickle
    with open("all_results_new.pkl", "wb") as f:
        pickle.dump(all_results, f)