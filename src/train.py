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

def compute_distance_score(embeddings, center):
    """Energy E(x) is defined as the squared Euclidean distance to center c."""
    return torch.sum((embeddings - center) ** 2, dim=1)

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

def calibrate_threshold(model, benign_dataset, oe_dataset, device, center_c):
    model.eval()
    benign_loader = DataLoader(benign_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    oe_loader = DataLoader(oe_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    benign_scores = []
    oe_scores = []

    with torch.no_grad():
        for images, _ in benign_loader:
            images = images.to(device).float()
            embeddings = model(images)
            distance = compute_distance_score(embeddings, center_c)
            if distance.dim() == 0:
                distance = distance.unsqueeze(0)
            benign_scores.extend(distance.cpu().numpy())

        for images, _ in oe_loader:
            images = images.to(device).float()
            embeddings = model(images)
            distance = compute_distance_score(embeddings, center_c)
            if distance.dim() == 0:
                distance = distance.unsqueeze(0)
            oe_scores.extend(distance.cpu().numpy())

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

def init_center_c(train_loader, model, device, eps=0.1):
    """Initializes the hypersphere center c as the mean of the benign data features."""
    print("Initializing Deep SVDD center 'c'...")
    model.eval()
    center = torch.zeros(Config.LATENT_DIM, device=device)
    n_samples = 0
    with torch.no_grad():
        for images, labels in train_loader:
            mask_in = (labels == Config.BENIGN_LABEL)
            if mask_in.sum() == 0:
                continue
            images = images[mask_in].to(device).float()
            embeddings = model(images)
            center += torch.sum(embeddings, dim=0)
            n_samples += embeddings.size(0)
    
    # Avoid trivial zero collapse
    center = center / (n_samples + 1e-6)
    center[(abs(center) < eps) & (center < 0)] = -eps
    center[(abs(center) < eps) & (center > 0)] = eps
    return center

def deep_svdd_loss(embeddings, labels, center):
    """
    Minimizes distance for benign samples; maximizes (up to margin) for OE samples.
    """
    dist = torch.sum((embeddings - center) ** 2, dim=1)
    
    mask_in = (labels == Config.BENIGN_LABEL)
    mask_out = (labels == Config.MALICIOUS_LABEL)
    
    total_loss = 0.0
    
    # 1. Minimize Distance for Benign (In-Distribution)
    if mask_in.sum() > 0:
        loss_in = dist[mask_in].mean()
        total_loss += loss_in
        
    # 2. Maximize Distance for OE/OOD Proxy Data
    if mask_out.sum() > 0:
        loss_out = torch.pow(F.relu(Config.SVDD_MARGIN - dist[mask_out]), 2).mean()
        total_loss += Config.OE_LAMBDA * loss_out
        
    return total_loss

def train_model():
    """
    Trains the CNN using a Deep SVDD Outlier Exposure approach.
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
        print("Using balanced sampling.")
    else:
        train_loader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = FlowPicCNN(
        input_dim=Config.FLOWPIC_DIM, 
        latent_dim=Config.LATENT_DIM
    ).to(device)
    
    # Initialize SVDD Center using Benign Data
    center_c = init_center_c(train_loader, model, device)
    
    # Weight decay regularizes W, helping prevent unbounded feature expansion
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    model.train()
    
    print("--- Starting Training (Deep SVDD with OE) ---")
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        benign_count = 0
        oe_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in pbar:
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            embeddings = model(images)
            
            # Compute SVDD Distance Loss
            loss = deep_svdd_loss(embeddings, labels, center_c)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            benign_count += (labels == Config.BENIGN_LABEL).sum().item()
            oe_count += (labels == Config.MALICIOUS_LABEL).sum().item()
            
            pbar.set_postfix({'loss': f"{total_loss/(benign_count+oe_count+1e-5):.4f}"})
            
    # Calibrate threshold from training distributions
    calibrated_threshold, energy_stats = calibrate_threshold(model, benign_dataset, oe_dataset, device, center_c)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'center_c': center_c.cpu(),
        'ood_threshold': calibrated_threshold,
        'energy_stats': energy_stats,
    }
    torch.save(checkpoint, Config.MODEL_DIR)

    print(f"Model saved to {Config.MODEL_DIR}")
    print(f"Calibrated SVDD Distance Threshold ({Config.CALIBRATION_STRATEGY}): {calibrated_threshold:.6f}")
    print(
        "Distance Stats | "
        f"Benign mean={energy_stats['benign_mean']:.4f}, std={energy_stats['benign_std']:.4f} | "
        f"OE mean={energy_stats['oe_mean']:.4f}, std={energy_stats['oe_std']:.4f}"
    )
    
    return model