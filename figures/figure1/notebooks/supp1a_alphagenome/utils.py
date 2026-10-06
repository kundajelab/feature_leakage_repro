"""Motif marginalization and ablation helpers for the AlphaGenome Figure 1 notebooks."""

import numpy as np
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import pandas as pd
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import one_hot_encode, characters, random_one_hot
from tangermeme.marginalize import marginalize
from tangermeme.ersatz import substitute
import random
from tqdm import tqdm
import time
from alphagenome.data import gene_annotation, genome, track_data, transcript
from alphagenome.models import dna_client
from alphagenome.visualization import plot_components
from tangermeme.io import extract_loci
from tqdm.auto import tqdm
from tangermeme.ablate import ablate
from tangermeme.ersatz import dinucleotide_shuffle

def create_marginalization_sequence(num_seqs, motifs_list, seq_len = 2048):
    sequences_with_motif = []
    sequences_with_shuffled_motif = []

    for i in tqdm(range(num_seqs)):
        motif = np.random.choice(motifs_list).upper()

        # hacky fix for when motif is not a valid DNA sequence
        try:
            dinuc_shuffled = dinucleotide_shuffle(one_hot_encode(motif).unsqueeze(0)).squeeze()
        except:
            i -= 1
            continue

        sample_idx = np.random.randint(dinuc_shuffled.shape[0])
        
        shuffled_ohe = dinuc_shuffled[sample_idx].squeeze()
        shuffled_motif = characters(shuffled_ohe)

        random_seq = random_one_hot((1,4,seq_len))
        with_motif = characters(substitute(random_seq, motif))
        with_shuffled_motif = characters(substitute(random_seq, shuffled_motif))

        sequences_with_motif.append(with_motif)
        sequences_with_shuffled_motif.append(with_shuffled_motif)
    
    return sequences_with_motif, sequences_with_shuffled_motif

def marginalize_motifs(sequences_with_motif, sequences_with_shuffled_motif, dna_model):
    # Make predictions.
    output_motifs = dna_model.predict_sequences(
        sequences = sequences_with_motif,
        requested_outputs={
            dna_client.OutputType.DNASE,
        },
        ontology_terms=None
    )


    # avoid rate limits
    time.sleep(60)

    output_shuffled_motifs = dna_model.predict_sequences(
        sequences = sequences_with_shuffled_motif,
        requested_outputs={
            dna_client.OutputType.DNASE,
        },
        ontology_terms=None
    )

    scores_with_motifs = [o.dnase.values.sum(axis=0) for o in output_motifs]
    scores_without_motifs = [o.dnase.values.sum(axis=0) for o in output_shuffled_motifs]

    scores_with_motifs = np.vstack(scores_with_motifs) 
    scores_without_motifs = np.vstack(scores_without_motifs)

    scores = scores_with_motifs - scores_without_motifs
    scores = scores.mean(axis=0)

    return scores, output_motifs[0].dnase.metadata

def ablate_motifs(motif_hits, model, motif_length, score_threshold = 0.9, 
                  seq_len = 2048, fasta = "/oak/stanford/groups/akundaje/jmschr/common/hg38/hg38.fa", 
                  num_loci = 200):
    # format bed file for extracting hits
    filt_hits_chroms = motif_hits
    hits = filt_hits_chroms.filter(['sequence_name','start','stop']).reset_index(drop=True)
    hits = hits.rename(columns={'sequence_name':'chromosome'})
    hits = hits.rename(columns={'stop':'end'})
    hits = hits.astype({'start':'int','end':'int'})
    hits_sampled = hits.sample(num_loci)

    loci = extract_loci(hits_sampled, fasta, in_window = seq_len, verbose=True, return_filtered=True)
    hits_sampled = hits_sampled.iloc[loci[1].numpy()]

    loci = loci[0]
    
    while loci.shape[0] != hits_sampled.shape[0]:
        hits_sampled = hits.sample(num_loci)
        loci = extract_loci(hits_sampled, fasta, in_window = seq_len, verbose=True)
    
    mask = loci.sum(axis=1).sum(axis=1) == seq_len
    loci = loci[mask]
    hits_sampled = hits_sampled.iloc[mask.numpy()]
    
    ablation_start = seq_len // 2 - motif_length // 2
    ablation_end = seq_len // 2 + motif_length // 2

    loci_ablated = dinucleotide_shuffle(loci, start = ablation_start, end = ablation_end, n=1).squeeze()

    loci_str = [characters(loci[i]) for i in range(len(loci))]
    loci_ablated_str = [characters(loci_ablated[i]) for i in range(len(loci_ablated))]

        # Make predictions.
    output_motifs = model.predict_sequences(
        sequences = loci_str,
        requested_outputs={
            dna_client.OutputType.DNASE,
        },
        ontology_terms=None
    )


    # avoid rate limits
    time.sleep(60)

    output_shuffled_motifs = model.predict_sequences(
        sequences = loci_ablated_str,
        requested_outputs={
            dna_client.OutputType.DNASE,
        },
        ontology_terms=None
    )

    scores_with_motifs = [o.dnase.values.sum(axis=0) for o in output_motifs]
    scores_without_motifs = [o.dnase.values.sum(axis=0) for o in output_shuffled_motifs]

    scores_with_motifs = np.vstack(scores_with_motifs) 
    scores_without_motifs = np.vstack(scores_without_motifs)

    scores = scores_with_motifs - scores_without_motifs
    scores = scores.mean(axis=0)

    return scores, output_motifs[0].dnase.metadata

def annotate_with_peak_distances(motif_hits_path, peaks_path):
    # Read files
    oct_sox_hits = pd.read_csv(motif_hits_path, sep='\t')
    cardiac_fibroblast_peaks = pd.read_csv(peaks_path, sep='\t', header=None)

    # Filter to normal chrom names
    oct_sox_chroms = oct_sox_hits[
        oct_sox_hits.sequence_name.astype(str).apply(lambda x: len(x) <= 5).values
    ]
    oct_sox_chroms.dropna(inplace=True)
    oct_sox_chroms['start'] = oct_sox_chroms['start'].astype(int)
    oct_sox_chroms['stop'] = oct_sox_chroms['stop'].astype(int)
    oct_sox_chroms.reset_index(drop=True, inplace=True)

    # Group peaks by chromosome with summit info
    peak_groups = {
        chrom: peaks[[1, 2, 9]].values  # 9 is the peak summit offset
        for chrom, peaks in cardiac_fibroblast_peaks.groupby(0)
    }

    # Preallocate lists
    closest_peak_start = []
    closest_peak_end = []
    closest_peak_distance = []

    # Loop through motif hits
    for row in tqdm(oct_sox_chroms.itertuples(), total=len(oct_sox_chroms)):
        chrom = row.sequence_name
        start = row.start
        end = row.stop

        if chrom not in peak_groups:
            closest_peak_start.append(None)
            closest_peak_end.append(None)
            closest_peak_distance.append(None)
            continue

        peaks = peak_groups[chrom]
        # Vectorized computation: summit absolute position = start + summit offset
        distances = np.minimum(
            np.abs((peaks[:, 0] + peaks[:, 2]) - start),
            np.abs((peaks[:, 0] + peaks[:, 2]) - end)
        )
        min_idx = np.argmin(distances)

        closest_peak_start.append(peaks[min_idx, 0])
        closest_peak_end.append(peaks[min_idx, 1])
        closest_peak_distance.append(distances[min_idx])

    # Add results to dataframe
    oct_sox_chroms['closest_enhancer_start'] = closest_peak_start
    oct_sox_chroms['closest_enhancer_end'] = closest_peak_end
    oct_sox_chroms['distance'] = closest_peak_distance

    return oct_sox_chroms