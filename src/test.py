import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os

from cnn_model import FlowPicCNN
from dataset import TrafficImageDataset
import config
from results import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_energy_score(logits):
    """
    Compute Energy Score for single-class OOD output.
    Energy E(x) = -logit
    """
    return -logits.squeeze()

def test_model():
    print(f"--- Testing Model on {device} ---")
    
    # Must match training: num_classes=1
    model = FlowPicCNN( 
        num_classes=1, 
    ).to(device)
    
    if not os.path.exists(config.MODEL_DIR):
        print(f"CRITICAL ERROR: Model file not found at {config.MODEL_DIR}")
        return

    loaded_threshold = config.OOD_THRESHOLD
    try:
        checkpoint = torch.load(config.MODEL_DIR, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'ood_threshold' in checkpoint:
                loaded_threshold = float(checkpoint['ood_threshold'])
            print("Model checkpoint loaded successfully.")
        else:
            model.load_state_dict(checkpoint)
            print("Model weights loaded successfully (legacy format).")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}\nEnsure num_classes=1 matches your training architecture.")
        return
        
    model.eval()

    try:
        benign_dataset = TrafficImageDataset(config.TEST_BENIGN_DIR, config.BENIGN_LABEL)
        malicious_dataset = TrafficImageDataset(config.TEST_MALICIOUS_DIR, config.MALICIOUS_LABEL)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"Data Loaded: {len(benign_dataset)} Benign samples, {len(malicious_dataset)} Malicious samples.")
    
    full_dataset = ConcatDataset([benign_dataset, malicious_dataset])
    loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    energy_scores = []
    true_labels = []

    print("Running Inference (Pure Energy OOD)...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device).float()
            
            logits, _ = model(images)
            
            # Calculate Scalar Energy
            batch_energy = compute_energy_score(logits)
            
            # Handle edge case where batch size is 1
            if batch_energy.dim() == 0:
                batch_energy = batch_energy.unsqueeze(0)
                
            energy_scores.extend(batch_energy.cpu().numpy())
            true_labels.extend(labels.numpy())

    energy_scores = np.array(energy_scores)
    true_labels = np.array(true_labels)

    benign_mask = (true_labels == config.BENIGN_LABEL)
    malicious_mask = (true_labels == config.MALICIOUS_LABEL)
    if benign_mask.any():
        print(
            f"Benign Energy: mean={energy_scores[benign_mask].mean():.4f}, "
            f"std={energy_scores[benign_mask].std():.4f}"
        )
    if malicious_mask.any():
        print(
            f"Malicious Energy: mean={energy_scores[malicious_mask].mean():.4f}, "
            f"std={energy_scores[malicious_mask].std():.4f}"
        )

    evaluate_model(true_labels, energy_scores, loaded_threshold)