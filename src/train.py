import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .config import Config

def outlier_exposure_loss(logits_in, logits_out):
    """
    OE Loss: CrossEntropy for ID data + KL Divergence (to uniform) for OOD data.
    """
    # 1. Standard Cross Entropy for In-Distribution (Normal) labels
    # Assuming Benign=0
    targets_in = torch.zeros(logits_in.shape[0], dtype=torch.long, device=logits_in.device)
    loss_ce = F.cross_entropy(logits_in, targets_in)
    
    # 2. OE Term: Force OOD (Anomaly) inputs to have uniform distribution (high entropy)
    # We want the model to be "confused" or output low confidence for OOD
    # Uniform target: 1/num_classes
    outputs_out = F.softmax(logits_out, dim=1)
    # Mean of probabilities should be uniform
    loss_oe = -torch.mean(torch.sum(outputs_out * torch.log(outputs_out + 1e-6), dim=1))
    
    return loss_ce + Config.OE_LAMBDA * loss_oe

def train_model(model, train_loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Fallback if standard training
    
    print("Starting Training...")
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(images)
            
            # --- OE Logic Adaptation ---
            # If batch contains both Normal (0) and Anomaly (1)
            # We can use the Anomaly samples as 'Outliers' for OE
            
            mask_normal = (labels == 0)
            mask_anomaly = (labels == 1)
            
            if mask_anomaly.sum() > 0 and mask_normal.sum() > 0:
                # We have both, use OE Loss
                loss = outlier_exposure_loss(logits[mask_normal], logits[mask_anomaly])
            else:
                # Fallback to standard CE if batch is pure
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss/(total/Config.BATCH_SIZE), 'acc': correct/total})
            
    # Save Model
    torch.save(model.state_dict(), Config.MODELS_DIR / f"{Config.MODEL_NAME}.pt")
    print(f"Model saved to {Config.MODELS_DIR}")
    return model