import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import pyBigWig
import pyfaidx
import pybedtools
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import random

# Tangermeme
from tangermeme.utils import one_hot_encode, random_one_hot, characters
from tangermeme.io import extract_loci
from tangermeme.ersatz import substitute, dinucleotide_shuffle
from tangermeme.marginalize import marginalize

# ----------------------------
# Helpers
# ----------------------------
def _ensure_channels_first(arr):
    """Ensure sequences are (N, 4, L). Accepts np.ndarray or torch.Tensor."""
    if isinstance(arr, torch.Tensor):
        if arr.ndim == 3 and arr.shape[-1] == 4 and arr.shape[1] != 4:
            return arr.permute(0, 2, 1).contiguous()
        return arr
    else:
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 4 and arr.shape[1] != 4:
            return np.transpose(arr, (0, 2, 1)).copy()
        return arr

def _normalize_labels_2d(labels):
    """Return labels as (N, C) np.ndarray."""
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels[:, None]
    return labels

def _stratify_vector(y_2d):
    """
    Build a 1D stratification vector for sklearn.
    - For single-task: returns the 0/1 vector.
    - For multi-label (C>1): stratify by 'has any positive' to stabilize splits.
    """
    y = np.asarray(y_2d)
    if y.ndim == 1 or y.shape[1] == 1:
        return y.reshape(-1)
    # multi-label: has any positive
    s = (y.sum(axis=1) > 0).astype(int)
    # Ensure both classes exist; otherwise return None (no stratify)
    if s.min() == s.max():
        return None
    return s

def _compute_pos_weight_tensor(y_train_t):
    """
    Compute per-task pos_weight for BCEWithLogitsLoss from TRAIN labels.
    y_train_t: torch.Tensor of shape (N, C) with 0/1 labels.
    Returns: torch.Tensor of shape (C,)
    """
    if y_train_t.ndim == 1:
        y_train_t = y_train_t.unsqueeze(-1)
    counts_pos = y_train_t.sum(dim=0)  # (C,)
    counts_neg = y_train_t.shape[0] - counts_pos
    pos_weight = counts_neg / torch.clamp(counts_pos, min=1)
    return pos_weight

# ----------------------------
# Data prep
# ----------------------------
def one_hot_encode_dna(sequence):
    """Convert DNA sequence to one-hot encoding using Tangermeme."""
    return one_hot_encode(sequence, alphabet=['A', 'C', 'G', 'T'])

def create_train_val_test(
    genome, bed, labels, seq_size=300, test_size=0.2, val_size=0.2, random_state=42
):
    """
    Create train/val/test splits with one-hot encoded sequences using extract_loci.
    Returns tensors X_train, X_val, X_test, y_train, y_val, y_test
    - X are float32, shape (N, 4, L)
    - y are float32, shape (N, C)
    """
    intervals = bed.copy()
    intervals['summit'] = (intervals['start'] + intervals['end']) // 2
    flank_size = seq_size // 2
    intervals['start'] = intervals['summit'] - flank_size
    intervals['end'] = intervals['summit'] + flank_size

    # Extract sequences + mask (filters out any with Ns etc.)
    seqs, keep_mask = extract_loci(
        intervals, genome, in_window=seq_size, return_filtered=True, verbose=True
    )

    labels = _normalize_labels_2d(labels)
    labels_filtered = labels[keep_mask]

    # Ensure channels-first and torch dtype
    seqs = _ensure_channels_first(seqs)
    seqs = torch.tensor(seqs, dtype=torch.float32)
    labels_filtered = torch.tensor(labels_filtered, dtype=torch.float32)

    # Build a robust stratification vector (or None if not feasible)
    strat = _stratify_vector(labels_filtered.numpy())

    # First split train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            seqs, labels_filtered, test_size=test_size, random_state=random_state,
            stratify=strat if strat is not None else None
        )
    except ValueError:
        # fallback (rare)
        X_train, X_test, y_train, y_test = train_test_split(
            seqs, labels_filtered, test_size=test_size, random_state=random_state
        )

    # Then split train/val using the same strategy
    strat2 = _stratify_vector(y_train.numpy())
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state,
            stratify=strat2 if strat2 is not None else None
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )

    # Final sanity: ensure (N, 4, L)
    X_train = _ensure_channels_first(X_train)
    X_val   = _ensure_channels_first(X_val)
    X_test  = _ensure_channels_first(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def merge_bed_files_and_label(bedfile_path1, bedfile_path2, peak_size=300):
    """Return merged_peaks DataFrame and labels (N,2) with 0/1 presence per source bed."""
    flank = peak_size // 2
    bed1 = pybedtools.BedTool(bedfile_path1)
    bed2 = pybedtools.BedTool(bedfile_path2)

    merged_peaks = bed1.cat(bed2, postmerge=False).sort().merge().to_dataframe()
    merged_peaks["summit"] = (merged_peaks["start"] + merged_peaks["end"]) // 2
    merged_peaks["start"] = merged_peaks["summit"] - flank
    merged_peaks["end"] = merged_peaks["summit"] + flank

    merged_bed = pybedtools.BedTool.from_dataframe(merged_peaks)
    df1 = merged_bed.intersect(bed1, wa=True, c=True).to_dataframe()
    df2 = merged_bed.intersect(bed2, wa=True, c=True).to_dataframe()
    assert df1.shape[0] == df2.shape[0] == merged_peaks.shape[0]

    labels = np.zeros((len(merged_peaks), 2), dtype=np.float32)
    labels[:, 0] = (df1.iloc[:, -1] > 0).astype(np.float32)
    labels[:, 1] = (df2.iloc[:, -1] > 0).astype(np.float32)
    return merged_peaks, labels

# ----------------------------
# Model
# ----------------------------
class CNNModel(nn.Module):
    """
    Simple CNN for (binary or multi-label) peak prediction.
    Outputs LOGITS of shape (B, num_tasks).
    """
    def __init__(self, num_tasks=2, inp_len=300, num_filt=128, num_layers=6,
                 kernel_size=17, use_bias=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.inp_len = inp_len
        self.num_filt = num_filt
        self.kernel_size = kernel_size

        self.initial_conv = nn.Conv1d(
            in_channels=4, out_channels=num_filt, kernel_size=kernel_size, padding=0
        )

        self.res_layers = nn.ModuleList()
        for i in range(num_layers):
            conv_layer = nn.Conv1d(
                in_channels=num_filt,
                out_channels=num_filt,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=use_bias if i >= num_layers - 2 else True
            )
            self.res_layers.append(nn.Sequential(
                conv_layer,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, padding=0)
            ))

        self._calculate_conv_output_size()

        self.dense1 = nn.Linear(self.conv_output_size, 100)
        self.dense2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, num_tasks)
        self.relu = nn.ReLU()

    def _calculate_conv_output_size(self):
        length = self.inp_len
        length = length - self.kernel_size + 1  # valid conv
        for _ in range(len(self.res_layers)):
            length = length // 2  # due to MaxPool1d(2)
        self.conv_output_size = length * self.num_filt

    def forward(self, x):
        # x: (B, 4, L)
        x = self.relu(self.initial_conv(x))
        for layer in self.res_layers:
            residual = x
            x = layer(x)
            if residual.shape == x.shape:
                x = x + residual
        x = x.view(x.size(0), -1)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        logits = self.output_layer(x)   # <-- LOGITS (no sigmoid)
        return logits

def get_model(num_tasks=2, inp_len=300, num_filt=128, num_layers=6, kernel_size=17, use_bias=True):
    return CNNModel(num_tasks, inp_len, num_filt, num_layers, kernel_size, use_bias)

# Wrapper used ONLY for marginalization (returns probabilities for the chosen task)
class _TaskWrapper(nn.Module):
    def __init__(self, base_model, task_idx=None, apply_sigmoid=True):
        super().__init__()
        self.base_model = base_model
        self.task_idx = task_idx
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        logits = self.base_model(x)
        if self.apply_sigmoid:
            outputs = torch.sigmoid(logits)
        else:
            outputs = logits
        if outputs.dim() == 1:
            return outputs
        elif outputs.dim() == 2:
            if outputs.shape[1] == 1:
                return outputs.squeeze(-1)
            else:
                return outputs[:, self.task_idx] if self.task_idx is not None else outputs
        else:
            raise ValueError(f"Unexpected output dimensions: {outputs.shape}")

# ----------------------------
# Marginalization
# ----------------------------
def run_marginalizations(model, sequence_list, motif, num_inserts=1, task_idx=1,
                         batchsize=32, device='cuda'):
    """
    Run marginalization experiment using Tangermeme's marginalize function.
    sequence_list: torch.Tensor (N, 4, L), float32
    motif: string; will be one-hot encoded to shape (4, mlen)
    """
    model.eval()

    # One-hot motif -> (L,4) or (4,L); ensure (4, L)
    m = one_hot_encode(motif, alphabet=['A', 'C', 'G', 'T'])
    if isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    motif_onehot = torch.tensor(m.T if m.shape[-1] == 4 else m, dtype=torch.float32)  # (4,L)

    # Ensure input dtype/shape
    sequence_list = _ensure_channels_first(sequence_list)
    if not isinstance(sequence_list, torch.Tensor):
        sequence_list = torch.tensor(sequence_list, dtype=torch.float32)

    # Wrap so tangermeme.predict can call .to()/.eval() etc.; return probabilities for the task
    wrapped = _TaskWrapper(model, task_idx=task_idx, apply_sigmoid=True)

    y_before, y_after = marginalize(
        wrapped,
        sequence_list,
        motif_onehot,
        alphabet=['A', 'C', 'G', 'T'],
        batch_size=batchsize,
        device=device
    )

    effects = y_after - y_before
    if isinstance(effects, torch.Tensor):
        effects = effects.detach().cpu().numpy()
    return effects

# ----------------------------
# Training
# ----------------------------
def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=64, epochs=15, learning_rate=1e-3, device='cuda',
                patience=5, min_delta=0.0):
    """
    Train with BCEWithLogitsLoss + per-task pos_weight.
    y_* must be shape (N, C); model outputs logits of shape (B, C).
    """
    model = model.to(device)

    # Ensure label shape (N, C)
    if y_train.ndim == 1: y_train = y_train.unsqueeze(-1)
    if y_val.ndim   == 1: y_val   = y_val.unsqueeze(-1)
    if y_test.ndim  == 1: y_test  = y_test.unsqueeze(-1)

    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, pin_memory=True)

    # Loss (+ pos_weight from TRAIN labels)
    pos_w = _compute_pos_weight_tensor(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Evaluate before training
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item()
        print(f'Before training - Test Loss: {test_loss/len(test_loader):.4f}')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item()

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader), 1)
        print(f'Epoch [{epoch+1}/{epochs}]  Train: {train_loss:.4f}  Val: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(torch.load('best_model.pth', map_location=device))
                break

    # Final test
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item()
        print(f'After training - Test Loss: {test_loss/len(test_loader):.4f}')

    return model

# ----------------------------
# Experiment driver
# ----------------------------
def run_leakage_effect_experiment(
    model, genome, peak_set, background_peak_set, motif, labels,
    relative_size=1.0, task_idx=0, single_task=False, return_model=False, device='cuda',
    random_state=42
):
    """
    - Adds `relative_size` * len(peak_set) background windows with 0 labels.
    - Trains model with BCEWithLogitsLoss(+pos_weight).
    - Runs marginalization on the test split for the given task.
    """
    num_peaks = peak_set.shape[0]
    num_background_peaks = int(num_peaks * relative_size)

    # Stable sampling for reproducibility
    background_sample = background_peak_set.sample(n=num_background_peaks, random_state=random_state)

    # Normalize labels to (N, C)
    labels = _normalize_labels_2d(labels)
    C = labels.shape[1] if labels.ndim == 2 else 1
    background_labels = np.zeros((num_background_peaks, C), dtype=labels.dtype)

    data_set = pd.concat([peak_set, background_sample], ignore_index=True)
    labels_all = np.concatenate([labels, background_labels], axis=0)

    # Split + tensors (X: (N,4,L), y: (N,C))
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test(
        genome, data_set, labels_all
    )

    # Train
    model = train_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32, epochs=15, device=device, patience=5, min_delta=0.0
    )

    # Marginalization on TEST set (returns prob deltas for selected task)
    leakage = run_marginalizations(model, X_test, motif, task_idx=task_idx, device=device)

    # Random motif as control
    random_motif = ''.join(random.choices(['A', 'C', 'G', 'T'], k=len(motif)))
    random_leakage = run_marginalizations(model, X_test, random_motif, task_idx=task_idx, device=device)

    # Final evaluation loss (reuse training criterion setup)
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    pos_w = _compute_pos_weight_tensor(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    test_loss = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item()
    final_test_loss = test_loss / max(len(test_loader), 1)

    out = {'leakage': leakage, 'random_leakage': random_leakage, 'loss': final_test_loss}
    if return_model:
        out['model'] = model
    return out
