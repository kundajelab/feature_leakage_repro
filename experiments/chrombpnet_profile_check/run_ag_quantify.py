#!/usr/bin/env python
"""Quantify AlphaGenome central fibroblast DNase signal at the top-ranked enhancers with 0-4 OCT-SOX insertions.

Outputs (outputs/): alphagenome_quant.csv.

Run (env: featleak-alphagenome):
    python run_ag_quantify.py
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from tangermeme.io import extract_loci
from tangermeme.ersatz import substitute
from tangermeme.utils import characters
from common import config
from common.models import load_alphagenome
from common.ag_score import _output_types, FIBROBLAST_ONTOLOGY
from common.motifs import OCT_SOX
OUT = Path(__file__).parent / "outputs"; SEQ_LEN = 16384; N_TOP = 6

def central(model, seq, OutputType):
    out = model.predict_sequence(sequence=seq, requested_outputs=[OutputType.DNASE],
                                 ontology_terms=[FIBROBLAST_ONTOLOGY])
    v = np.asarray(out.dnase.values); v = v.sum(1) if v.ndim==2 else v
    c=len(v)//2; w=int(150*len(v)/SEQ_LEN); return float(v[c-w:c+w].sum())

def main():
    ranked = pd.read_csv(OUT/"peak_ranking.csv").head(N_TOP).reset_index(drop=True)
    model = load_alphagenome(); OutputType=_output_types()
    mid=SEQ_LEN//2; offs=[mid-150,mid+150,mid-200,mid+200]; rows=[]
    for r in range(N_TOP):
        df=pd.DataFrame({"chrom":[ranked.chrom[r]],"start":[int(ranked.start[r])],"end":[int(ranked.end[r])]})
        base=extract_loci(df,str(config.resolve("hg38_fasta")),in_window=SEQ_LEN,verbose=False)
        vals=[central(model,characters(base[0]),OutputType)]; cur=base
        for o in offs:
            cur=substitute(cur,OCT_SOX,start=o); vals.append(central(model,characters(cur[0]),OutputType))
        rows.append({"locus":f"{ranked.chrom[r]}:{ranked.start[r]}","ag_0":vals[0],"ag_4":vals[-1],"ag_retained":vals[-1]/vals[0]})
    out=pd.DataFrame(rows); out.to_csv(OUT/"alphagenome_quant.csv",index=False)
    print("AlphaGenome central peak (retained = signal_4/signal_0):"); print(out.to_string(index=False))
main()
