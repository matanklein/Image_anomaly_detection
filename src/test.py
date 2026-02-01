import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os

from cnn_model import FlowPicCNN
from dataset import TrafficImageDataset
import config
from results import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_energy_score(logits, temperature=1.0):
    """
    Compute Energy Score for Multi-class (2-class) output.
    
    Formula: E(x) = -T * log( sum( exp( f(x) / T ) ) )
    Ref: "Energy-based Out-of-distribution Detection" (Liu et al. 2020)
    
    Args:
        logits (Tensor): Shape [Batch_Size, 2]
        temperature (float): Scaling factor (usually 1.0)
    
    Returns:
        Tensor: Energy scores [Batch_Size]
    """
    # Scale by temperature
    logits = logits / temperature
    
    # LogSumExp along the class dimension (dim=1)
    # This aggregates confidence across both 'Benign' and 'Malicious' classes
    energy = -temperature * torch.logsumexp(logits, dim=1)
    
    return energy

def test_model():
    print(f"--- Testing Model on {device} ---")
    
    # Load Model
    model = FlowPicCNN(
        input_dim=config.FLOWPIC_DIM, 
        num_classes=config.NUM_CLASSES, 
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    
    if not os.path.exists(config.MODEL_DIR):
        print(f"CRITICAL ERROR: Model file not found at {config.MODEL_DIR}")
        return

    try:
        # Load weights
        model.load_state_dict(torch.load(config.MODEL_DIR, map_location=device))
        print("Model weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("Hint: Check if 'num_classes' in test.py matches train.py (should be 2).")
        return
        
    model.eval()

    # Load Data
    try:
        # Assuming dataset.py returns (image, label)
        benign_dataset = TrafficImageDataset(config.TEST_BENIGN_DIR, config.BENIGN_LABEL)
        malicious_dataset = TrafficImageDataset(config.TEST_MALICIOUS_DIR, config.MALICIOUS_LABEL)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"Data Loaded: {len(benign_dataset)} Benign samples, {len(malicious_dataset)} Malicious samples.")
    
    # Concatenate for batch processing
    full_dataset = ConcatDataset([benign_dataset, malicious_dataset])
    loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    energy_scores = []
    true_labels = []

    # Inference
    print("Running Inference (Energy-Based OOD)...")
    
    # Get Temperature from config or default to 1.0
    temp = getattr(config, 'T', 1.0) 

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device).float()
            
            # Forward pass (Get Logits)
            logits = model(images)
            
            # Calculate Energy
            # Note: We do NOT apply Softmax here. Energy is calculated on raw logits.
            batch_energy = compute_energy_score(logits, temperature=temp)
            
            # Store results
            energy_scores.extend(batch_energy.cpu().numpy())
            true_labels.extend(labels.numpy())

    energy_scores = np.array(energy_scores)
    true_labels = np.array(true_labels)

    # Evaluation
    # Using the OOD_THRESHOLD from config to calculate binary metrics
    evaluate_model(true_labels, energy_scores, config.OOD_THRESHOLD)

if __name__ == "__main__":
    test_model()