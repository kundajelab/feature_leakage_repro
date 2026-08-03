from tangermeme.utils import one_hot_encode, random_one_hot, characters
from tangermeme.ersatz import substitute, dinucleotide_shuffle
from tangermeme.predict import predict
from tangermeme.marginalize import marginalize
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.stats import wilcoxon


class MotifEffectSum(nn.Module):
    def __init__(self, num_tasks=50, kernel=8, C=(320,480,960)):
        super().__init__()
        c1,c2,c3 = C
        self.conv1 = nn.Conv1d(4,  c1, kernel, padding='same', bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel, padding='same', bias=False)
        self.conv3 = nn.Conv1d(c2, c3, kernel, padding='same', bias=False)
        self.act   = nn.ReLU()
        # map channels → task contribution at each position
        self.to_tasks = nn.Conv1d(c3, num_tasks, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))     # (B, 960, L)
        x = self.to_tasks(x)            # (B, num_tasks, L)
        out = x.sum(dim=-1)             # additive across positions
        return out

def make_motif_dict(num_motifs, motif_length, min_effect = 0.2, max_effect = 2.0, num_tasks = 50, min_sparsity = 0.05, max_sparsity = 0.4):
    motif_dict = {}
    for i in range(num_motifs):
        motif = characters(random_one_hot((1,4,motif_length)))
        effect = np.random.uniform(min_effect, max_effect)
        sparsity = np.zeros(num_tasks)
        # make vector of 0s and random 1s
        num_active = np.random.binomial(num_tasks, np.random.uniform(min_sparsity, max_sparsity))
        sparsity[np.random.choice(num_tasks, num_active, replace=False)] = 1

        motif_dict[motif] = {'effect': effect, 'sparsity': sparsity}
    return motif_dict

def generate_dataset(motif_dict, num_samples, sequence_length = 2000, num_tasks = 50, max_motifs_inserted=5):
    """"Use tangermeme to generate random one hot encoded sequences and labels

        Insert up to max_motifs_inserted motifs into the sequence
        and generate labels based on the motifs inserted
    """
    sequences = []
    labels = []

    for i in tqdm(range(num_samples)):
        num_motifs_inserted = np.random.randint(1, max_motifs_inserted + 1)
        sequence = random_one_hot((1,4,sequence_length))
        label = np.zeros(num_tasks)
        for j in range(num_motifs_inserted):
            motif = np.random.choice(list(motif_dict.keys()))
            label += motif_dict[motif]['effect'] * motif_dict[motif]['sparsity']
            position = np.random.randint(0, sequence_length - len(motif))
            sequence = substitute(sequence, motif, start=position)
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels


class SequenceDataset(Dataset):
    """PyTorch Dataset class for sequence data"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert numpy arrays to tensors and ensure correct shape
        # sequences come as (1, 4, seq_len), we want (4, seq_len) for Conv1d
        sequence = torch.FloatTensor(self.sequences[idx]).squeeze(0)  # Remove batch dim
        label = torch.FloatTensor(self.labels[idx])
        return sequence, label


def create_dataloaders(sequences, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                      batch_size=32, shuffle=True, random_state=42):
    """
    Create train, validation, and test dataloaders from generate_dataset output.
    
    Args:
        sequences: List of numpy arrays from generate_dataset
        labels: List of numpy arrays from generate_dataset  
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        test_ratio: Fraction of data for testing (default: 0.15)
        batch_size: Batch size for dataloaders (default: 32)
        shuffle: Whether to shuffle training data (default: True)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Convert to numpy arrays for easier handling
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=test_ratio, random_state=random_state, stratify=None
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
    )
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


class CustomModelPyTorch(nn.Module):
    """
    PyTorch equivalent of the TensorFlow CustomModel.
    
    Architecture:
    - Conv1D(320 filters, kernel_size=8) + MaxPool1D(4) + Dropout(0.2)
    - Conv1D(480 filters, kernel_size=8) + MaxPool1D(4) + Dropout(0.2)  
    - Conv1D(960 filters, kernel_size=8) + Dropout(0.5)
    - Flatten + Dense(50)
    """
    def __init__(self, num_tasks=50):
        super(CustomModelPyTorch, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, stride=1)
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, stride=1)
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, stride=1)
        
        # Pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Dropout layers
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Starting with 2000 length sequence
        # After conv1 (kernel=8): 2000 - 8 + 1 = 1993
        # After pool1 (pool=4, stride=4): 1993 // 4 = 498
        # After conv2 (kernel=8): 498 - 8 + 1 = 491  
        # After pool2 (pool=4, stride=4): 491 // 4 = 122
        # After conv3 (kernel=8): 122 - 8 + 1 = 115
        # Final feature size: 960 * 115 = 110400
        
        self.fc = nn.Linear(960 * 115, num_tasks)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 4, sequence_length)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.relu(self.conv3(x))
        x = self.drop3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Final dense layer
        x = self.fc(x)
        
        return x


def generate_simple_dataset(motif_dict, num_samples, sequence_length=200, num_tasks=10, motif_probability=0.5):
    """
    Generate a simple dataset where each sequence has a motif with given probability.
    
    Args:
        motif_dict: Dictionary with motif information (should contain exactly 1 motif)
        num_samples: Number of sequences to generate
        sequence_length: Length of each sequence
        num_tasks: Number of tasks
        motif_probability: Probability that a sequence contains the motif (0.5 = 50/50)
    
    Returns:
        sequences, labels: Lists of numpy arrays
    """
    assert len(motif_dict) == 1, "This function is designed for exactly 1 motif"
    
    motif = list(motif_dict.keys())[0]
    motif_info = motif_dict[motif]
    
    sequences = []
    labels = []
    
    for i in tqdm(range(num_samples)):
        # Start with random sequence
        sequence = random_one_hot((1, 4, sequence_length))
        label = np.zeros(num_tasks)
        
        # Add motif with given probability
        if np.random.random() < motif_probability:
            label = motif_info['effect'] * motif_info['sparsity']
            position = np.random.randint(0, sequence_length - len(motif))
            sequence = substitute(sequence, motif, start=position)
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels


def train_simple_model(train_loader, val_loader, num_tasks=10, num_epochs=100, lr=1e-3, device='cpu', patience = 10):
    """
    Simple training function for the PyTorch model.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader  
        num_tasks: Number of output tasks
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        model, train_losses, val_losses
    """
    model = MotifEffectSum(num_tasks=num_tasks).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_sequences, batch_labels in tqdm(train_loader, total=len(train_loader)):
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:2d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")

            return model, train_losses, val_losses

    return model, train_losses, val_losses


def evaluate_motif_learning(model, test_loader, motif_dict, device='cpu'):
    """
    Evaluate how well the model learned the motif pattern.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test dataloader
        motif_dict: Dictionary with motif info (should contain exactly 1 motif)
        device: Device to run evaluation on
    
    Returns:
        dict with evaluation metrics
    """
    assert len(motif_dict) == 1, "This function is designed for exactly 1 motif"
    
    motif_info = list(motif_dict.values())[0]
    active_tasks = np.where(motif_info['sparsity'] > 0)[0]
    inactive_tasks = np.where(motif_info['sparsity'] == 0)[0]
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_sequences, batch_labels in test_loader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            outputs = model(batch_sequences)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate correlations
    correlations = {}
    for task in range(all_predictions.shape[1]):
        corr = np.corrcoef(all_predictions[:, task], all_targets[:, task])[0, 1]
        correlations[task] = corr if not np.isnan(corr) else 0.0
    
    # Separate sequences with/without motif
    has_motif = np.any(all_targets > 0, axis=1)
    
    results = {
        'active_tasks': active_tasks.tolist(),
        'inactive_tasks': inactive_tasks.tolist(),
        'correlations': correlations,
        'avg_correlation_active': np.mean([correlations[t] for t in active_tasks]) if len(active_tasks) > 0 else 0.0,
        'avg_correlation_inactive': np.mean([correlations[t] for t in inactive_tasks]) if len(inactive_tasks) > 0 else 0.0,
        'num_with_motif': int(np.sum(has_motif)),
        'num_without_motif': int(np.sum(~has_motif)),
        'total_samples': len(all_targets)
    }
    
    return results


def test_single_motif():
    """
    Simple test function for single motif learning.
    Returns True if model successfully learned the motif.
    """
    print("Testing single motif learning...")
    
    # Create single motif
    motif_dict = make_motif_dict(
        num_motifs=1, motif_length=8, min_effect=2.0, max_effect=2.0,
        num_tasks=5, min_sparsity=0.6, max_sparsity=0.6
    )
    
    motif_name = list(motif_dict.keys())[0]
    print(f"Motif: {motif_name}")
    print(f"Active tasks: {np.where(motif_dict[motif_name]['sparsity'] > 0)[0]}")
    
    # Generate data
    sequences, labels = generate_simple_dataset(
        motif_dict, num_samples=500, sequence_length=200, 
        num_tasks=5, motif_probability=0.5
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        sequences, labels, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_losses, val_losses = train_simple_model(
        train_loader, val_loader, num_tasks=5, num_epochs=30, device=device
    )
    
    # Evaluate
    results = evaluate_motif_learning(model, test_loader, motif_dict, device)
    
    print(f"\nResults:")
    print(f"Sequences with motif: {results['num_with_motif']}")
    print(f"Sequences without motif: {results['num_without_motif']}")
    print(f"Average correlation (active tasks): {results['avg_correlation_active']:.3f}")
    print(f"Average correlation (inactive tasks): {results['avg_correlation_inactive']:.3f}")
    
    # Success if active tasks have good correlation
    success = results['avg_correlation_active'] > 0.5
    print(f"Success: {success}")
    
    return success


def measure_feature_leakage(motif, model, sparsity_array, actual_effect_size, num_sequences=100, seq_length=2000):
    """
    Measure feature leakage of a motif on active vs inactive tasks using tangermeme marginalization.
    
    Args:
        motif: String motif sequence (e.g., 'ATCGATCG')
        model: Trained PyTorch model
        sparsity_array: Array indicating which tasks should be affected by this motif (1=active, 0=inactive)
        actual_effect_size: The true effect size this motif should have
        num_sequences: Number of random sequences to test on (default: 100)
        seq_length: Length of sequences to generate (default: 2000)
    
    Returns:
        dict: Contains actual_effect_size, proportion_tasks_active, leakage_effect, predicted_effect
    """
    # Generate random sequences
    random_sequences = random_one_hot((num_sequences, 4, seq_length))
    
    # Use tangermeme marginalize to get predictions before and after motif insertion
    y_before, y_after = marginalize(model, random_sequences, motif)
    
    # Calculate motif effect by subtracting before from after
    motif_effects = y_after - y_before  # Shape: (num_sequences, num_tasks)

    # Average effects across sequences
    avg_effects = motif_effects.mean(axis=0).numpy() # Shape: (num_tasks,)
    
    # Separate active and inactive tasks
    active_tasks = sparsity_array > 0
    inactive_tasks = sparsity_array == 0
    
    # Calculate effects on active vs inactive tasks
    predicted_effect = np.mean(avg_effects[active_tasks]) if np.any(active_tasks) else 0.0
    leakage_effect = np.mean(avg_effects[inactive_tasks]) if np.any(inactive_tasks) else 0.0
    
    statistic, p_value = wilcoxon(avg_effects[inactive_tasks])
    # Calculate proportion of tasks that are active
    proportion_tasks_active = np.mean(active_tasks)
    
    return {
        'actual_effect_size': actual_effect_size,
        'proportion_tasks_active': proportion_tasks_active.item(),
        'leakage_effect': leakage_effect.item(),
        'predicted_effect': predicted_effect,
        'statistic': statistic.item(),
        'p_value': p_value.item()
    }