"""Keras helpers for training the Figure 4 panel b FOS/OCT4 CNNs."""

import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils import sequence
import pyBigWig
import pyfaidx
import pybedtools
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping


def create_train_val_test(genome, bed, labels, seq_size = 300, test_size=0.2, val_size=0.2, random_state=42):
    
    seqs = []

    # read in bed file and convert to df
    # bed = pybedtools.BedTool(bed).to_dataframe()

    flank_size = seq_size // 2

    # iterate over bed file and extract sequences from genome
    for _, row in bed.iterrows():
        
        summit = (row['start'] + row['end']) // 2
        seq = str(genome[row['chrom']][summit-flank_size:summit+flank_size])
        
        seqs.append(sequence.one_hot_encode_dna(seq))   

    # convert to numpy array
    seqs = np.array(seqs)

    # split labels and seqs into train, val, and test
    X_train, X_test, y_train, y_test = train_test_split(seqs, labels, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

     
def merge_bed_files_and_label(bedfile_path1, bedfile_path2, peak_size = 300):

    flank = peak_size // 2

    bed1 = pybedtools.BedTool(bedfile_path1)
    bed2 = pybedtools.BedTool(bedfile_path2)

    merged_peaks = bed1.cat(bed2, postmerge=False).sort().merge()

    merged_peaks = merged_peaks.to_dataframe()

    merged_peaks["summit"] = (merged_peaks["start"] + merged_peaks["end"]) // 2

    merged_peaks["start"] = merged_peaks["summit"] - flank
    merged_peaks["end"] = merged_peaks["summit"] + flank

    labels = []

    for _, row in tqdm(merged_peaks.iterrows(), total=merged_peaks.shape[0]):
        
        label = np.zeros(2)

        if bed1.any_hits(pybedtools.Interval(row['chrom'], row['start'], row['end'])):
            label[0] = 1
        
        if bed2.any_hits(pybedtools.Interval(row['chrom'], row['start'], row['end'])):
            label[1] = 1
    
        labels.append(label)
    
    return merged_peaks, np.array(labels)

def get_model(num_tasks=2, inp_len=300, num_filt=128, num_layers=6, kernel_size=17, use_bias=True):
    inp = tf.keras.Input((inp_len, 4))
    
    x = tf.keras.layers.Conv1D(num_filt,
                               kernel_size=kernel_size,
                               padding='valid',
                               activation='relu')(inp)
    
    for i in range(num_layers):
        
        if i < num_layers - 2:
            conv_x = tf.keras.layers.Conv1D(num_filt,
                                            kernel_size=kernel_size,
                                            padding='same',
                                            activation='relu')(x)
        else:
            conv_x = tf.keras.layers.Conv1D(num_filt,
                                            kernel_size=kernel_size,
                                            padding='same',
                                            activation='relu',
                                            use_bias=use_bias)(x)
        x = tf.keras.layers.add([conv_x, x])
        x = tf.keras.layers.MaxPool1D(2,padding='same')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    
    # Last dense layer
    x = tf.keras.layers.Dense(num_tasks)(x)
    
    # Apply sigmoid activation for probabilities
    output = tf.keras.layers.Activation('sigmoid')(x)
    
    return tf.keras.Model(inputs=inp, outputs=output)

# TODO: marginalization code to calculate how insertion of a motif effects the output
def run_marginalizations(model, sequence_list, motif, num_inserts = 1, task_idx=1, 
                            control=True, batchsize=32):
    """Run the marginalization experiment on the DeepSEA model.

    Args:
    model (tf.keras.Model): The DeepSEA model.
    sequence_list (numpy array): Array of one_hot encoded sequences.
    motif (numpy array): The motif to insert into the sequences.
    num_inserts (int): The number of insertions to perform.
    task_idx (int): The index of the task to marginalize over.
    control (bool): Whether to use a control sequence.
    batchsize (int): The batch size to use.
    """

    sequences_w_motif, sequences, control_sequences = sequence.insert_motif_mult(sequence_list, motif, num_inserts, control)

    # Get the predictions for the sequences with the motif
    predictions_w_motif = model.predict(sequences_w_motif, batch_size=batchsize)[:, task_idx]

    # Get the predictions for the original sequences
    predictions = model.predict(sequences, batch_size=batchsize)[:, task_idx]

    effect = predictions_w_motif - predictions

    if control:
        # Get the predictions for the control sequences
        predictions_control = model.predict(control_sequences, batch_size=batchsize)[:, task_idx]

        effect = effect - (predictions_control - predictions)

    return effect



def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64, epochs=10, verbose=1, patience=1, min_delta=0.01):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # Evaluate model on test set before training
    before = model.evaluate(X_test, y_test, verbose=0)
    print(f'Before AUC: {before[1]}')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, mode='min', verbose=verbose, restore_best_weights=True)

    # Train the model with validation data and early stopping
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[early_stopping])

    # Evaluate model on test set after training
    after = model.evaluate(X_test, y_test, batch_size = batch_size, verbose=0)
    print(f'After AUC: {after[1]}')

    return model

def run_leakage_effect_experiment(model, genome, peak_set, background_peak_set, motif, labels, relative_size = 1.0, task_idx = 1, single_task=False, return_model=False):

    num_peaks = peak_set.shape[0]

    num_background_peaks = int(num_peaks * relative_size)

    background_peak_set = background_peak_set.sample(n=num_background_peaks)
    background_labels = np.zeros((num_background_peaks, 2))

    if single_task:
        background_labels = np.zeros(num_background_peaks)

    data_set = pd.concat([peak_set, background_peak_set])
    labels = np.concatenate([labels, background_labels])

    # create train, val, test split
    try:
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test(genome, data_set, labels)
    
    except Exception as e:
        print(e)
        raise ValueError('Error creating train, val, test split')
    
    # train model
    model = train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, epochs=5)

    # one hot encode motif
    random_motif = sequence.random_one_hot_encode_dna(len(motif))
    motif = sequence.one_hot_encode_dna(motif)

    leakage = run_marginalizations(model, X_test, motif, task_idx = task_idx)
    random_leakage = run_marginalizations(model, X_test, random_motif, task_idx = task_idx)

    after = model.evaluate(X_test, y_test, verbose=0)

    if return_model:
        return {

            'leakage': leakage,
            'random_leakage': random_leakage,
            'AUC': after[1],
            'model': model
        }
    
    else:
        return {

            'leakage': leakage,
            'random_leakage': random_leakage,
            'AUC': after[1]
        }
    
    

