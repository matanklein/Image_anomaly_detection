import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

import config as Config
from dataset import TrafficImageDataset 
from cnn_model import FlowPicCNN


def compute_energy_score(logits):
    return -logits.squeeze()


def _best_threshold(labels, scores, strategy='youden'):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return Config.OOD_THRESHOLD

    candidates = np.r_[unique_scores[0] - 1e-6, (unique_scores[:-1] + unique_scores[1:]) / 2.0, unique_scores[-1] + 1e-6]
    best_thr = Config.OOD_THRESHOLD
    best_metric = -1e9

    for thr in candidates:
        preds = (scores > thr).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if strategy == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            metric = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        else:
            metric = tpr - fpr

        if metric > best_metric:
            best_metric = metric
            best_thr = float(thr)

    return best_thr


def calibrate_threshold(model, benign_dataset, oe_dataset, device):
    model.eval()

    benign_loader = DataLoader(benign_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    oe_loader = DataLoader(oe_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    benign_scores = []
    oe_scores = []

    with torch.no_grad():
        for images, _ in benign_loader:
            images = images.to(device).float()
            logits, _ = model(images)
            energy = compute_energy_score(logits)
            if energy.dim() == 0:
                energy = energy.unsqueeze(0)
            benign_scores.extend(energy.cpu().numpy())

        for images, _ in oe_loader:
            images = images.to(device).float()
            logits, _ = model(images)
            energy = compute_energy_score(logits)
            if energy.dim() == 0:
                energy = energy.unsqueeze(0)
            oe_scores.extend(energy.cpu().numpy())

    benign_scores = np.array(benign_scores, dtype=float)
    oe_scores = np.array(oe_scores, dtype=float)

    labels = np.concatenate([
        np.zeros_like(benign_scores, dtype=int),
        np.ones_like(oe_scores, dtype=int)
    ])
    scores = np.concatenate([benign_scores, oe_scores])

    calibrated_thr = _best_threshold(labels, scores, strategy=Config.CALIBRATION_STRATEGY)
    stats = {
        'benign_mean': float(np.mean(benign_scores)) if benign_scores.size > 0 else float('nan'),
        'benign_std': float(np.std(benign_scores)) if benign_scores.size > 0 else float('nan'),
        'oe_mean': float(np.mean(oe_scores)) if oe_scores.size > 0 else float('nan'),
        'oe_std': float(np.std(oe_scores)) if oe_scores.size > 0 else float('nan'),
    }

    model.train()
    return calibrated_thr, stats

def pure_energy_ood_loss(logits, labels):
    """
    Pure Energy-Bounded Loss (adapted from Liu et al. 2020 for 1-class formulation).
    Here, the network outputs a single scalar logit.
    Energy E(x) is defined as -logit.
    """
    # E(x) = -f(x)
    energy = -logits.squeeze()
    
    mask_in = (labels == Config.BENIGN_LABEL)
    mask_out = (labels == Config.MALICIOUS_LABEL)
    
    total_loss = 0.0
    
    # 1. Minimize Energy for Benign (In-Distribution)
    if mask_in.sum() > 0:
        # Penalize if Energy exceeds M_IN
        loss_in = torch.pow(F.relu(energy[mask_in] - Config.M_IN), 2).mean()
        total_loss += loss_in
        
    # 2. Maximize Energy for OE/OOD Proxy Data
    if mask_out.sum() > 0:
        # Penalize if Energy falls below M_OUT
        loss_out = torch.pow(F.relu(Config.M_OUT - energy[mask_out]), 2).mean()
        total_loss += Config.OE_LAMBDA * loss_out
        
    return total_loss

def train_model():
    """
    Trains the CNN using a pure Energy-Bounded OOD approach.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Preparing Data for Training on {device} ---")

    benign_dataset = TrafficImageDataset(Config.TRAIN_DIR, label=Config.BENIGN_LABEL)
    oe_dataset = TrafficImageDataset(Config.TRAIN_DIR, label=Config.MALICIOUS_LABEL)
    
    print(f"Benign Samples: {len(benign_dataset)}")
    print(f"OE/OOD Proxy Samples: {len(oe_dataset)}")
    
    if len(benign_dataset) == 0 or len(oe_dataset) == 0:
        print("Error: One of the datasets is empty. Check your directories.")
        return None

    full_dataset = ConcatDataset([benign_dataset, oe_dataset])

    if Config.BALANCED_SAMPLING:
        benign_len = len(benign_dataset)
        oe_len = len(oe_dataset)
        sample_weights = [1.0 / benign_len] * benign_len + [1.0 / oe_len] * oe_len
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
        print("Using balanced sampling (oversampling minority class).")
    else:
        train_loader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    
    model = FlowPicCNN(num_classes=1).to(device)
    model.train()
    
    # We use a lower learning rate for energy margin tuning
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print("--- Starting Training (Pure Energy-Based OOD) ---")
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        benign_count = 0
        oe_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in pbar:
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            
            # Forward Pass (Single scalar logit)
            logits, _ = model(images)
            
            # Compute Pure Energy Loss
            loss = pure_energy_ood_loss(logits, labels)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            benign_count += (labels == Config.BENIGN_LABEL).sum().item()
            oe_count += (labels == Config.MALICIOUS_LABEL).sum().item()
            
            pbar.set_postfix({'loss': f"{total_loss/(benign_count+oe_count+1e-5):.4f}"})
            
    # Calibrate threshold from training distributions (benign vs OE proxy)
    calibrated_threshold, energy_stats = calibrate_threshold(model, benign_dataset, oe_dataset, device)

    # Save checkpoint (backward-compatible loading handled in test.py)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ood_threshold': calibrated_threshold,
        'energy_stats': energy_stats,
    }
    torch.save(checkpoint, Config.MODEL_DIR)

    print(f"Model saved to {Config.MODEL_DIR}")
    print(f"Calibrated OOD Threshold ({Config.CALIBRATION_STRATEGY}): {calibrated_threshold:.6f}")
    print(
        "Energy stats | "
        f"Benign mean={energy_stats['benign_mean']:.4f}, std={energy_stats['benign_std']:.4f} | "
        f"OE mean={energy_stats['oe_mean']:.4f}, std={energy_stats['oe_std']:.4f}"
    )
    
    return model