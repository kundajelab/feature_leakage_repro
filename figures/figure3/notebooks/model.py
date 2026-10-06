"""Beluga model, AP-1 counting head, synthetic datasets and training/analysis helpers for Figure 3.

The Beluga architecture is adapted from the Kipoi model
https://github.com/kipoi/models/blob/master/DeepSEA/beluga/model.py
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import numpy as np
from scipy.stats import pearsonr
from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import wilcoxon, false_discovery_control
from fuzzywuzzy import process


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
    def __init__(self):
        super().__init__()  
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 320, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(320, 320, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(320, 480, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(480, 480, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(480, 640, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(640, 640, (1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(67840, 2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2003, 2002)),
            ),
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     return self.model(x.unsqueeze(2))

    def forward(self, x):
        # model inputs are A G C T instead of A C G T
        return self.model(x[:, [0, 2, 1, 3]].unsqueeze(2))


import torch
from torch import nn

COUNT_IDX = 2002  # 0-based index of the new counting task

class BelugaWithCounting(nn.Module):
    """
    Beluga wrapper that adds a new 2003rd task.
    - First 2002 outputs: passed through sigmoid (binary tasks).
    - 2003rd output (index 2002): raw logit for counting/regression.
    """
    def __init__(self, original_beluga_model, init="xavier_uniform"):
        super().__init__()

        # --- Pull the original submodules we need (everything before the final linear) ---
        self.conv    = original_beluga_model.model[0]   # conv tower
        self.drop    = original_beluga_model.model[1][0]  # Dropout(0.5)
        self.flatten = original_beluga_model.model[1][1]  # Lambda flatten
        self.fc1     = original_beluga_model.model[1][2]  # Linear(67840 -> 2003)
        self.relu    = original_beluga_model.model[1][3]  # ReLU

        # --- Expand the final linear layer: Linear(2003 -> 2002)  ->  Linear(2003 -> 2003) ---
        old_last: nn.Linear = original_beluga_model.model[1][4][1]
        new_last = nn.Linear(in_features=old_last.in_features,
                             out_features=old_last.out_features + 1)

        # copy old weights/bias & init the new row
        with torch.no_grad():
            # copy first 2002 rows
            new_last.weight[:old_last.out_features, :] = old_last.weight
            new_last.bias[:old_last.out_features] = old_last.bias
            # init the new (last) row
            if init == "xavier_uniform":
                nn.init.xavier_uniform_(new_last.weight[old_last.out_features:old_last.out_features+1])
                nn.init.zeros_(new_last.bias[old_last.out_features:old_last.out_features+1])
            elif init == "zeros":
                nn.init.zeros_(new_last.weight[old_last.out_features:old_last.out_features+1])
                nn.init.zeros_(new_last.bias[old_last.out_features:old_last.out_features+1])
            else:
                raise ValueError(f"Unknown init '{init}'")

        self.final_linear = new_last

        # --- Keep a compatibility handle: update the original model's final linear too ---
        # This lets existing code that references model.model[1][4][1] keep working.
        original_beluga_model.model[1][4] = nn.Sequential(
            original_beluga_model.model[1][4][0],  # the Lambda(view) stays
            self.final_linear
        )
        self.model = original_beluga_model.model  # NOTE: This still contains the trailing Sigmoid.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original Beluga expects AGCT channel order; inputs are ACGT.
        # Reorder to AGCT and add the extra spatial dim for Conv2d.
        x = x[:, [0, 2, 1, 3]].unsqueeze(2)  # (B,4,1,L)

        h = self.conv(x)
        h = self.drop(h)
        h = self.flatten(h)
        h = self.fc1(h)
        h = self.relu(h)                 # (B, 2003)

        logits = self.final_linear(h)    # (B, 2003)
        # First 2002 tasks: sigmoid; last task: raw logit for counting
        orig = torch.sigmoid(logits[:, :COUNT_IDX])
        count = logits[:, COUNT_IDX:COUNT_IDX+1]
        return torch.cat([orig, count], dim=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits for all 2003 tasks (no sigmoid on the first 2002, useful for custom losses)."""
        x = x[:, [0, 2, 1, 3]].unsqueeze(2)
        h = self.conv(x)
        h = self.drop(h)
        h = self.flatten(h)
        h = self.fc1(h)
        h = self.relu(h)
        return self.final_linear(h)  # (B, 2003)



def expand_tasks_to_2003(model):
    """
    Expand the Beluga model from 2002 to 2003 tasks by adding one more output
    for AP1 motif counting. The new task (index 2002) will be initialized randomly.
    
    Args:
        model: Beluga model instance
        
    Returns:
        Modified BelugaWithCounting model with 2003 outputs
    """
    return BelugaWithCounting(model)



def sigmoid_to_count(sigmoid_output, max_count=5):
    """
    Convert sigmoid output [0,1] to count values [0, max_count].
    This is used with the simple approach where sigmoid is still applied.
    
    Args:
        sigmoid_output: Sigmoid output in [0, 1]
        max_count: Maximum count value to scale to
        
    Returns:
        Count values scaled to [0, max_count]
    """
    return sigmoid_output * max_count


def calculate_pearson_r(predictions, targets):
    """
    Calculate Pearson correlation coefficient between predictions and targets.
    
    Args:
        predictions: Tensor of predictions
        targets: Tensor of true values
        
    Returns:
        Pearson R value (float), or 0.0 if calculation fails
    """
    try:
        # Convert to numpy arrays
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Calculate Pearson R
        r, _ = pearsonr(pred_np, target_np)
        
        # Handle NaN case (e.g., if one array has no variance)
        if np.isnan(r):
            return 0.0
        
        return float(r)
    except:
        # Return 0 if calculation fails for any reason
        return 0.0


class SyntheticMotifCountDataset(Dataset):
    """
    Dataset that generates synthetic DNA sequences with a specified motif
    and returns the count of non-overlapping motif occurrences.
    """
    def __init__(self, N, L=2000, motif="ATGAGTCAC", max_count=5, seed=0):
        self.N = N
        self.L = L
        self.motif = motif
        self.max_count = max_count
        self.rng = random.Random(seed)
        self.K = len(motif) if isinstance(motif, str) else getattr(motif, "shape", [None, None])[-1]
        if self.K is None:
            raise ValueError("Can't infer motif length; pass a string motif or tensor with shape (..., K).")

    def __len__(self): 
        return self.N

    def _nonoverlapping_starts(self, n):
        """Pick n start positions in [0, L-K] without overlap; simple rejection (n<=5)."""
        starts = []
        tries = 0
        while len(starts) < n and tries < 5000:
            s = self.rng.randrange(0, self.L - self.K + 1)
            ok = True
            for t in starts:
                if not (s + self.K <= t or t + self.K <= s):
                    ok = False
                    break
            if ok:
                starts.append(s)
            tries += 1
        return sorted(starts)

    def __getitem__(self, idx):
        # background: (1, 4, L)
        X = random_one_hot((1, 4, self.L)).type(torch.float32)
        # number of insertions
        n = self.rng.randint(0, self.max_count)
        if n > 0:
            for s in self._nonoverlapping_starts(n):
                X = substitute(X, self.motif, start=s)

        # Return (4, L) so DataLoader gives (B,4,L)
        return X[0], torch.tensor(float(n), dtype=torch.float32)


class SyntheticMotifClassificationDataset(Dataset):
    """
    Dataset that generates synthetic DNA sequences for binary motif classification.
    Each sequence either contains exactly one motif (label=1) or no motif (label=0).
    The dataset is balanced with equal numbers of positive and negative examples.
    """
    def __init__(self, N, L=2000, motif="ATGAGTCAC", seed=0):
        """
        Args:
            N: Total number of samples (must be even for balanced dataset)
            L: Sequence length
            motif: Motif sequence to classify
            seed: Random seed for reproducibility
        """
        if N % 2 != 0:
            raise ValueError("N must be even to create a balanced dataset")
            
        self.N = N
        self.L = L
        self.motif = motif
        self.rng = random.Random(seed)
        self.K = len(motif) if isinstance(motif, str) else getattr(motif, "shape", [None, None])[-1]
        if self.K is None:
            raise ValueError("Can't infer motif length; pass a string motif or tensor with shape (..., K).")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Generate background sequence: (1, 4, L)
        X = random_one_hot((1, 4, self.L), random_state=self.rng.randint(0, 1000000)).type(torch.float32)
        
        # First half of dataset: sequences with motif (label=1)
        # Second half of dataset: sequences without motif (label=0)
        if idx < self.N // 2:
            # Insert exactly one motif at a random position
            start_pos = self.rng.randrange(0, self.L - self.K + 1)
            X = substitute(X, self.motif, start=start_pos)
            label = torch.tensor(1.0, dtype=torch.float32)
        else:
            # No motif insertion - keep background sequence
            label = torch.tensor(0.0, dtype=torch.float32)
        
        # Return (4, L) so DataLoader gives (B,4,L)
        return X[0], label


def train_ap1_motif_counting(model, num_epochs=10, batch_size=32, learning_rate=1e-4, 
                           motif="ATGAGTCAC", max_count=5, train_size=10000, 
                           val_size=2000, device='cuda', freeze_backbone=True):
    """
    Train the AP1 motif counting task (task index 2002) on synthetic data.
    
    Args:
        model: Beluga model with 2003 tasks
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        motif: AP1 motif sequence to count
        max_count: Maximum number of motifs per sequence
        train_size: Number of training samples
        val_size: Number of validation samples
        device: Device to train on ('cuda' or 'cpu')
        freeze_backbone: Whether to freeze the backbone and only train the final layer
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Create datasets
    train_dataset = SyntheticMotifCountDataset(
        N=train_size, motif=motif, max_count=max_count, seed=42
    )
    val_dataset = SyntheticMotifCountDataset(
        N=val_size, motif=motif, max_count=max_count, seed=123
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer
    if freeze_backbone:
        # Only train the final linear layer
        params_to_train = model.model[1][4][1].parameters()
        print("Freezing backbone - only training final layer")
    else:
        # Train the entire model
        params_to_train = model.parameters()
        print("Training entire model")
    
    optimizer = torch.optim.Adam(params_to_train, lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_pearson_r': [],
        'val_pearson_r': []
    }
    
    print(f"Training AP1 motif counting for {num_epochs} epochs...")
    print(f"Motif: {motif}, Max count: {max_count}")
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    model.train()
    for epoch in range(num_epochs):
        # Training phase
        train_loss = 0.0
        train_mae = 0.0
        num_train_batches = 0
        train_predictions_all = []
        train_targets_all = []
        
        for batch_idx, (sequences, counts) in enumerate(train_loader):
            sequences = sequences.to(device)
            counts = counts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get all predictions but only use task 2002 (AP1 counting)
            predictions = model(sequences)  # Shape: [batch_size, 2003]
            ap1_predictions = predictions[:, 2002]  # AP1 motif counting task
            
            # Apply ReLU to ensure non-negative counts (optional but recommended)
            ap1_predictions = torch.relu(ap1_predictions)
            
            # Compute loss
            loss = criterion(ap1_predictions, counts)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_mae += F.l1_loss(ap1_predictions, counts).item()
            num_train_batches += 1
            
            # Collect predictions and targets for Pearson R calculation
            train_predictions_all.append(ap1_predictions.detach())
            train_targets_all.append(counts.detach())
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        num_val_batches = 0
        val_predictions_all = []
        val_targets_all = []
        
        with torch.no_grad():
            for sequences, counts in val_loader:
                sequences = sequences.to(device)
                counts = counts.to(device)
                
                predictions = model(sequences)
                ap1_predictions = predictions[:, 2002]
                
                # Apply ReLU to ensure non-negative counts
                ap1_predictions = torch.relu(ap1_predictions)
                
                loss = criterion(ap1_predictions, counts)
                val_loss += loss.item()
                val_mae += F.l1_loss(ap1_predictions, counts).item()
                num_val_batches += 1
                
                # Collect predictions and targets for Pearson R calculation
                val_predictions_all.append(ap1_predictions.detach())
                val_targets_all.append(counts.detach())
        
        # Calculate average metrics
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        avg_train_mae = train_mae / num_train_batches
        avg_val_mae = val_mae / num_val_batches
        
        # Calculate Pearson R for train and validation
        train_predictions_concat = torch.cat(train_predictions_all, dim=0)
        train_targets_concat = torch.cat(train_targets_all, dim=0)
        train_pearson_r = calculate_pearson_r(train_predictions_concat, train_targets_concat)
        
        val_predictions_concat = torch.cat(val_predictions_all, dim=0)
        val_targets_concat = torch.cat(val_targets_all, dim=0)
        val_pearson_r = calculate_pearson_r(val_predictions_concat, val_targets_concat)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['train_pearson_r'].append(train_pearson_r)
        history['val_pearson_r'].append(val_pearson_r)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Train R: {train_pearson_r:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}, Val R: {val_pearson_r:.4f}')
        
        model.train()
    
    print("Training completed!")
    return model, history


def evaluate_ap1_counting(model, motif="ATGAGTCAC", max_count=5, test_size=1000, 
                         batch_size=32, device='cuda'):
    """
    Evaluate the AP1 motif counting performance on test data.
    
    Args:
        model: Trained Beluga model
        motif: AP1 motif sequence
        max_count: Maximum number of motifs per sequence
        test_size: Number of test samples
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = SyntheticMotifCountDataset(
        N=test_size, motif=motif, max_count=max_count, seed=999
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_true_counts = []
    
    print(f"Evaluating AP1 motif counting on {test_size} test samples...")
    
    with torch.no_grad():
        for sequences, counts in test_loader:
            sequences = sequences.to(device)
            
            predictions = model(sequences)
            ap1_predictions = predictions[:, 2002]  # AP1 motif counting task
            
            # Apply ReLU to ensure non-negative counts
            ap1_predictions = torch.relu(ap1_predictions).cpu()
            
            all_predictions.extend(ap1_predictions.tolist())
            all_true_counts.extend(counts.tolist())
    
    all_predictions = torch.tensor(all_predictions)
    all_true_counts = torch.tensor(all_true_counts)
    
    # Calculate metrics
    mse = F.mse_loss(all_predictions, all_true_counts).item()
    mae = F.l1_loss(all_predictions, all_true_counts).item()
    pearson_r = calculate_pearson_r(all_predictions, all_true_counts)
    
    # Calculate accuracy for exact count prediction
    exact_accuracy = (torch.round(all_predictions) == all_true_counts).float().mean().item()
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'pearson_r': pearson_r,
        'exact_accuracy': exact_accuracy,
        'predictions': all_predictions.tolist(),
        'true_counts': all_true_counts.tolist()
    }
    
    print(f"Test Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Pearson R: {pearson_r:.4f}")
    print(f"  Exact Count Accuracy: {exact_accuracy:.4f}")
    
    return metrics


def plot_training_curves(history, save_path=None):
    """
    Plot training curves including loss, MAE, and Pearson R.
    
    Args:
        history: Training history dictionary from train_ap1_motif_counting
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        axes[1].plot(epochs, history['train_mae'], 'b-', label='Train MAE')
        axes[1].plot(epochs, history['val_mae'], 'r-', label='Val MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot Pearson R
        axes[2].plot(epochs, history['train_pearson_r'], 'b-', label='Train Pearson R')
        axes[2].plot(epochs, history['val_pearson_r'], 'r-', label='Val Pearson R')
        axes[2].set_title('Training and Validation Pearson R')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Pearson Correlation')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_ylim([0, 1])  # Pearson R is typically [0, 1] for this task
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def measure_motif_insertion_effect(model, test_motif, num_sequences=100, 
                                  sequence_length=2000, device='cuda', seed=42):
    """
    Generate sequences with 2 AP1 motifs at fixed positions (500 and 1500), 
    then insert 1 test motif in the middle and measure prediction changes.
    
    Args:
        model: Trained Beluga model with AP1 counting capability
        test_motif: The motif sequence to test (different from AP1)
        num_sequences: Number of sequences to generate (default 100)
        sequence_length: Length of sequences to generate
        device: Device to run inference on
        seed: Random seed for reproducibility
        
    Returns:
        numpy array of prediction changes (after - before) for the AP1 counting task
    """
    model = model.to(device)
    model.eval()
    
    ap1_motif = "ATGAGTCAC"  # Hardcoded AP1 motif
    prediction_changes = []
    middle_position = sequence_length // 2  # Position 1000 for 2000bp sequences
    
    print(f"Generating {num_sequences} sequences with 2 AP1 motifs at positions 500 and 1500...")
    print(f"Testing insertion of motif: {test_motif}")
    
    with torch.no_grad():
        for i in range(num_sequences):
            # Generate background sequence
            X = random_one_hot((1, 4, sequence_length), random_state=seed + i).type(torch.float32)
            
            # Insert 2 AP1 motifs at fixed positions 500 and 1500
            X = substitute(X, ap1_motif, start=500)
            X = substitute(X, ap1_motif, start=1500)
            
            # Get prediction before inserting test motif
            X_input = X[0].unsqueeze(0).to(device)  # Shape: (1, 4, L)
            pred_before = model(X_input)[:, 2002].cpu()  # AP1 counting task prediction
            
            # Insert the test motif in the middle
            X_modified = substitute(X, test_motif, start=middle_position)
            
            # Get prediction after inserting test motif
            X_modified_input = X_modified[0].unsqueeze(0).to(device)  # Shape: (1, 4, L)
            pred_after = model(X_modified_input)[:, 2002].cpu()  # AP1 counting task prediction
            
            # Calculate the change in prediction
            change = pred_after.item() - pred_before.item()
            prediction_changes.append(change)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{num_sequences} sequences")
    
    prediction_changes = np.array(prediction_changes)
    
    print(f"Generated {len(prediction_changes)} sequences successfully")
    print(f"Mean prediction change: {prediction_changes.mean():.4f} ± {prediction_changes.std():.4f}")
    print(f"Min change: {prediction_changes.min():.4f}, Max change: {prediction_changes.max():.4f}")
    
    return prediction_changes


def fuzzy_substring_match(main_string, target_substring, threshold=90):
    # Create a list of all possible substrings
    substrings = [main_string[i:j] for i in range(len(main_string)) for j in range(i + 1, len(main_string) + 1)]

    # Use fuzzy matching to find the closest match to the target substring
    match, score = process.extractOne(target_substring, substrings)

    # Check if the match score meets the threshold
    # Adjust the threshold according to your need. Higher value means more strict matching.
    return score >= threshold


def extract_activations_batch(model, layer_name, input_data_list, batch_size=32, device='cuda'):
    """
    Extract activations from a specific layer for many predictions efficiently.
    
    Args:
        model: Your Beluga or BelugaWithCounting model
        layer_name: String path to the layer (e.g., 'model.0.0')
        input_data_list: input tensors, each with shape (num_sequences, 4, sequence_length)
        batch_size: Number of sequences to process at once
        device: Device to run on
    
    Returns:
        all_activations: List of activation tensors, one per input sequence
    """
    model = model.to(device)
    model.eval()
    
    all_activations = []
    
    # Set up the hook
    activations_storage = {}
    
    def hook_fn(module, input, output):
        activations_storage['batch_activations'] = output.detach().cpu()
    
    # Get the layer and register hook
    layer = model
    for attr in layer_name.split('.'):
        layer = getattr(layer, attr)
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        # Process in batches
        for i in range(0, len(input_data_list), batch_size):
            batch_end = min(i + batch_size, len(input_data_list))
            batch_inputs = input_data_list[i:batch_end]
            
            # Stack into batch tensor: (batch_size, 4, sequence_length)
            batch_tensor = batch_inputs.to(device)
            
            with torch.no_grad():
                _ = model(batch_tensor)
            
            # Extract activations for this batch
            batch_activations = activations_storage['batch_activations']
            
            # Split back into individual sequences
            for j in range(batch_activations.shape[0]):
                all_activations.append(batch_activations[j])
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {batch_end}/{len(input_data_list)} sequences")
    
    finally:
        # Always remove the hook
        handle.remove()
    
    return all_activations

class BelugaProbeWithClampedActivations(nn.Module):
    """
    Probe that operates on clamped activations from Beluga's model.0.14 layer.
    """
    def __init__(self, beluga_model, clamp_indices, probe_output_size):
        super().__init__()
        
        self.beluga_model = beluga_model
        self.clamp_indices = clamp_indices
        
        # Freeze the Beluga model
        for param in self.beluga_model.parameters():
            param.requires_grad = False
        
        # Storage for extracted activations
        self.extracted_activations = None
        
        # The activations from model.0.14 will have shape (batch, 640, 1, width)
        # We'll need to process them for the probe
        
        # Probe architecture - adapt based on your needs
        self.probe = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling: (batch, 640, 1, 1)
            nn.Flatten(),                   # (batch, 640)
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, probe_output_size)
        )
        
        # Register hooks
        self.hook_handle = self._register_clamping_and_extraction_hook()
    
    def _register_clamping_and_extraction_hook(self):
        """Register hook to clamp activations and extract them"""
        def clamping_extraction_hook(module, input, output):
            # Clamp specified channels to 0
            # output shape: (batch, 640, 1, width)
            # channels = self.clamp_indices[:, 0]
            # dim1s = self.clamp_indices[:, 1] 
            # dim2s = self.clamp_indices[:, 2]
            # output[:, channels, dim1s, dim2s] = 1 # channels, dim1s, dim2s
            
            # Store the clamped activations
            self.extracted_activations = output.clone()
            
            return output  # Return modified output for the rest of the network
        
        # Get the target layer: model.0.14
        target_layer = self.beluga_model.model[0][14]
        return target_layer.register_forward_hook(clamping_extraction_hook)
    
    def forward(self, x):
        # Forward through Beluga model (this will trigger the hook)
        with torch.no_grad():
            _ = self.beluga_model(x)
        
        # Use the extracted clamped activations for the probe
        probe_output = self.probe(self.extracted_activations)
        return probe_output
        #return self.extracted_activations
    
    def __del__(self):
        # Clean up hook
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


