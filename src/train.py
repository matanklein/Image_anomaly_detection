import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

import config as Config
from dataset import TrafficImageDataset 

def outlier_exposure_loss(logits_in, logits_out):
    """
    OE Loss: CrossEntropy for ID data + KL Divergence (to uniform) for OOD data.
    
    Args:
        logits_in: Logits for Benign samples (Target: Class 0)
        logits_out: Logits for Anomaly/OE samples (Target: Uniform Distribution)
    """
    # 1. In-Distribution Loss: Minimize CE to class 0 (Benign)
    if len(logits_in) > 0:
        # Target for Benign is always 0
        targets_in = torch.zeros(logits_in.shape[0], dtype=torch.long, device=logits_in.device)
        loss_ce = F.cross_entropy(logits_in, targets_in)
    else:
        loss_ce = torch.tensor(0.0, device=logits_in.device)

    # 2. Out-of-Distribution Loss: Force uniform distribution
    if len(logits_out) > 0:
        outputs_out = F.softmax(logits_out, dim=1)
        # Entropy maximization (KL to uniform)
        # We want probability 0.5 / 0.5 for binary classes
        # Formula: -mean(sum(p * log(p)))
        loss_oe = -torch.mean(torch.sum(outputs_out * torch.log(outputs_out + 1e-6), dim=1))
    else:
        loss_oe = torch.tensor(0.0, device=logits_in.device)
    
    return loss_ce + Config.OE_LAMBDA * loss_oe

def train_model(model, device):
    """
    Trains the model using Outlier Exposure.
    Data loading is handled internally to match CNN_anomaly_detection architecture.
    """
    print(f"--- Preparing Data for Training on {device} ---")

    # Initialize Datasets
    # Label 0: Benign (In-Distribution)
    # Label 1: Malicious (Out-of-Distribution / Outlier Exposure)
    benign_dataset = TrafficImageDataset(Config.TRAIN_BENIGN_DIR, label=Config.BENIGN_LABEL)
    oe_dataset = TrafficImageDataset(Config.TRAIN_OE_DIR, label=Config.MALICIOUS_LABEL)
    
    print(f"Benign Samples: {len(benign_dataset)}")
    print(f"OE Samples: {len(oe_dataset)}")
    
    if len(benign_dataset) == 0 or len(oe_dataset) == 0:
        print("Error: One of the datasets is empty. Check your directories.")
        return None

    # Create DataLoader
    # Concatenate datasets to mix Benign and OE samples in the same batch
    full_dataset = ConcatDataset([benign_dataset, oe_dataset])
    train_loader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Optimizer
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Fallback for pure batches
    
    print("--- Starting Training (Outlier Exposure) ---")
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in pbar:
            # Type casting: CNN inputs must be Float, Labels must be Long
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            
            # Forward Pass (Returns Logits)
            logits = model(images)
            
            # --- OE Logic Adaptation ---
            # Separate the batch based on labels
            mask_normal = (labels == 0)
            mask_anomaly = (labels == 1)
            
            # Extract logits for specific groups
            logits_in = logits[mask_normal]
            logits_out = logits[mask_anomaly]
            
            # Calculate Loss
            if mask_anomaly.sum() > 0 and mask_normal.sum() > 0:
                # Mixed batch: Use specialized OE Loss
                loss = outlier_exposure_loss(logits_in, logits_out)
            
            elif mask_normal.sum() > 0:
                # Pure Benign batch: Standard Cross Entropy
                loss = criterion(logits, labels)
            
            elif mask_anomaly.sum() > 0:
                # Pure Anomaly batch: Only OE term (maximize entropy)
                outputs_out = F.softmax(logits, dim=1)
                loss = -torch.mean(torch.sum(outputs_out * torch.log(outputs_out + 1e-6), dim=1)) * Config.OE_LAMBDA
            
            else:
                continue # Should not happen

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy Metric
            # Note: For OE samples, "Accuracy" is whether the model predicted Class 1? 
            # In OE training, we usually just want to know if it classifies Benign correctly 
            # and if Loss is going down. Here we calculate standard acc for predicted labels.
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss/(total/Config.BATCH_SIZE), 'acc': correct/total})
            
    # Save Model
    # Ensure directory exists
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Construct save path using Config variables
    save_path = Config.MODELS_DIR / "model.pth" # Or match config.MODEL_DIR if defined directly
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model