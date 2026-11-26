"""Helpers for training and evaluating the Figure 4 single-task and multi-task FOS/OCT4 CNNs."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tangermeme.io import extract_loci
from tangermeme.marginalize import marginalize


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


def _seed_rngs(seed):
    """Seed torch/numpy reproducibly and return the caller's previous RNG states.

    Seeding the global generators in place would make every later random draw in
    the calling script depend on how much randomness this function consumed, so
    the previous state is captured here and restored by ``_restore_rngs``.
    """
    prev = (torch.get_rng_state(), np.random.get_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return prev


def _restore_rngs(prev):
    """Put back the RNG states captured by ``_seed_rngs``."""
    torch_state, numpy_state, cuda_state = prev
    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def train_multitask_model(peak_regions, labels, background_regions, genome_path, 
                         relative_proportion=0.2, test_split=0.2, val_split=0.2,
                         epochs=5, batch_size=64, lr=0.001, patience=5, 
                         min_delta=0.0, device='cuda', random_state=42):
    """
    Train a multitask CNN model on peak regions vs background.
    
    Args:
        peak_regions: DataFrame with columns ['chrom', 'start', 'end'] for peak regions
        labels: numpy array of shape (n_peaks, 2) with binary labels for each task
        background_regions: DataFrame with columns ['chrom', 'start', 'end'] for background regions
        genome_path: path to genome fasta file
        relative_proportion: proportion of background regions relative to peak regions
        test_split: proportion for test set
        val_split: proportion for validation set (from training set)
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        patience: early stopping patience
        min_delta: minimum improvement for early stopping
        device: 'cuda' or 'cpu'
        random_state: random seed; also seeds weight initialisation and
            batch shuffling, so a given seed reproduces the fit exactly.

    Returns:
        trained model
    """
    _prev_rng = _seed_rngs(random_state)

    # Sample background regions
    n_background = int(peak_regions.shape[0] * relative_proportion)
    background_sample = background_regions.sample(n=n_background, random_state=random_state)
    
    # Combine peak and background regions
    all_regions = pd.concat([peak_regions, background_sample], ignore_index=True)
    all_labels = np.concatenate([labels, np.zeros((n_background, 2))], axis=0)
    
    # Extract sequences
    loci, keep_mask = extract_loci(all_regions, genome_path, in_window=300, 
                                  return_filtered=True, verbose=True)
    
    # Filter labels and convert to tensors
    all_labels = all_labels[keep_mask]
    loci = torch.tensor(loci, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.float32)
    
    # Filter out sequences with N bases
    keep_mask = loci.sum(axis=1).sum(axis=1) == loci.shape[2]
    loci = loci[keep_mask]
    all_labels = all_labels[keep_mask]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        loci, all_labels, test_size=test_split, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=random_state
    )
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, 
                             shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, 
                           shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, 
                            shuffle=False, pin_memory=True)
    
    # Initialize model
    model = CNNModel(num_tasks=2, num_layers=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Evaluate before training
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item()
    print(f'Before training - Test Loss: {test_loss/len(test_loader):.4f}')
    
    # Training loop
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
        val_loss /= max(len(val_loader), 1)
        print(f'Epoch [{epoch+1}/{epochs}]  Train: {train_loss:.4f}  Val: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item()
        print(f'After training - Test Loss: {test_loss/len(test_loader):.4f}')
    
    auc = evaluate_auc_multitask(model, X_test, y_test, task_idx=0, device=device)
    
    _restore_rngs(_prev_rng)
    return model, auc


def train_singletask_model(peak_regions, labels, background_regions, genome_path,
                          task_idx=0, relative_proportion=0.2, test_split=0.2, val_split=0.2,
                          epochs=5, batch_size=64, lr=0.001, patience=5,
                          min_delta=0.0, device='cuda', random_state=42):
    """
    Train a single-task CNN model on peak regions vs background.
    
    Args:
        peak_regions: DataFrame with columns ['chrom', 'start', 'end'] for peak regions
        labels: numpy array of shape (n_peaks, 2) with binary labels for each task
        background_regions: DataFrame with columns ['chrom', 'start', 'end'] for background regions
        genome_path: path to genome fasta file
        task_idx: which task to use for single-task training (0 or 1)
        relative_proportion: proportion of background regions relative to peak regions
        test_split: proportion for test set
        val_split: proportion for validation set (from training set)
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        patience: early stopping patience
        min_delta: minimum improvement for early stopping
        device: 'cuda' or 'cpu'
        random_state: random seed; also seeds weight initialisation and
            batch shuffling, so a given seed reproduces the fit exactly.

    Returns:
        trained model
    """
    _prev_rng = _seed_rngs(random_state)

    # Sample background regions
    n_background = int(peak_regions.shape[0] * relative_proportion)
    background_sample = background_regions.sample(n=n_background, random_state=random_state)
    
    # Combine peak and background regions
    all_regions = pd.concat([peak_regions, background_sample], ignore_index=True)
    all_labels = np.concatenate([labels, np.zeros((n_background, 2))], axis=0)
    
    # Extract sequences
    loci, keep_mask = extract_loci(all_regions, genome_path, in_window=300, 
                                  return_filtered=True, verbose=True)
    
    # Filter labels and convert to tensors
    all_labels = all_labels[keep_mask]
    loci = torch.tensor(loci, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.float32)
    
    # Filter out sequences with N bases
    keep_mask = loci.sum(axis=1).sum(axis=1) == loci.shape[2]
    loci = loci[keep_mask]
    all_labels = all_labels[keep_mask][:, task_idx]  # Select single task
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        loci, all_labels, test_size=test_split, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=random_state
    )
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, 
                             shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, 
                           shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, 
                            shuffle=False, pin_memory=True)
    
    # Initialize model
    model = CNNModel(num_tasks=1, num_layers=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Evaluate before training
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb.unsqueeze(-1)).item()
    print(f'Before training - Test Loss: {test_loss/len(test_loader):.4f}')
    
    # Training loop
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
            loss = criterion(logits, yb.unsqueeze(-1))
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
                val_loss += criterion(logits, yb.unsqueeze(-1)).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f'Epoch [{epoch+1}/{epochs}]  Train: {train_loss:.4f}  Val: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb.unsqueeze(-1)).item()
        print(f'After training - Test Loss: {test_loss/len(test_loader):.4f}')
    
    auc = evaluate_auc_singletask(model, X_test, y_test, device=device)
    
    _restore_rngs(_prev_rng)
    return model, auc


def test_motif_effect_multitask(model, test_loci, motif, task_idx=0, device='cuda'):
    """
    Test motif effect on a multitask model.
    
    Args:
        model: trained multitask CNN model
        test_loci: numpy array or tensor of sequences to test on, shape (N, 4, L)
        motif: DNA motif string (e.g., "ATTGTTATGCAAAT")
        task_idx: which task to evaluate (0 or 1)
        device: 'cuda' or 'cpu'
        
    Returns:
        numpy array of motif effects for each sequence
    """
    model.eval()
    
    y_before, y_after = marginalize(model, test_loci, motif, device=device, verbose=True)
    
    motif_effect = y_after - y_before
    motif_effect = motif_effect[:, task_idx]  # Select task
    
    if isinstance(motif_effect, torch.Tensor):
        motif_effect = motif_effect.cpu().numpy()
        
    return motif_effect


def evaluate_auc_multitask(model, X_test, y_test, task_idx=0, device='cuda'):
    """
    Evaluate AUC for a multitask model on test data.
    
    Args:
        model: trained multitask CNN model
        X_test: test sequences, shape (N, 4, L) - tensor or numpy array
        y_test: test labels, shape (N, num_tasks) - tensor or numpy array
        task_idx: which task to evaluate AUC for (0 or 1)
        device: 'cuda' or 'cpu'
        
    Returns:
        float: AUC score for the specified task
    """
    model.eval()
    
    # Convert to tensors if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Move to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
    
    # Convert to numpy for sklearn
    y_true = y_test[:, task_idx].cpu().numpy()
    y_pred = probs[:, task_idx].cpu().numpy()
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    return auc


def test_motif_effect_singletask(model, test_loci, motif, device='cuda'):
    """
    Test motif effect on a single-task model.
    
    Args:
        model: trained single-task CNN model
        test_loci: numpy array or tensor of sequences to test on, shape (N, 4, L)
        motif: DNA motif string (e.g., "ATTGTTATGCAAAT")
        device: 'cuda' or 'cpu'
        
    Returns:
        numpy array of motif effects for each sequence
    """
    model.eval()
    
    y_before, y_after = marginalize(model, test_loci, motif, device=device, verbose=True)
    
    motif_effect = y_after - y_before
    motif_effect = motif_effect.squeeze()  # Remove extra dimensions
    
    if isinstance(motif_effect, torch.Tensor):
        motif_effect = motif_effect.cpu().numpy()
        
    return motif_effect


def evaluate_auc_singletask(model, X_test, y_test, device='cuda'):
    """
    Evaluate AUC for a single-task model on test data.
    
    Args:
        model: trained single-task CNN model
        X_test: test sequences, shape (N, 4, L) - tensor or numpy array
        y_test: test labels, shape (N,) - tensor or numpy array
        device: 'cuda' or 'cpu'
        
    Returns:
        float: AUC score
    """
    model.eval()
    
    # Convert to tensors if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Move to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).squeeze()
    
    # Convert to numpy for sklearn
    y_true = y_test.cpu().numpy()
    y_pred = probs.cpu().numpy()
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    return auc