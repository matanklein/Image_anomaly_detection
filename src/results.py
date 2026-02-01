import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(true_labels, energy_scores, threshold):
    """
    Calculates and prints evaluation metrics for the Energy-Based Model.
    
    Args:
        true_labels (np.array): Ground truth labels (0=Benign, 1=Malicious).
        energy_scores (np.array): Calculated energy scores.
        threshold (float): Threshold to decide classification.
    """
    # 1. Convert Energy Scores to Binary Predictions
    # If Energy > Threshold => Malicious (1)
    # If Energy <= Threshold => Benign (0)
    preds = (energy_scores > threshold).astype(int)

    # 2. Calculate Core Metrics
    acc = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, zero_division=0)
    recall = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)

    # 3. Calculate AUROC
    # AUROC is threshold-independent. Handle case where only 1 class exists in test set.
    try:
        if len(np.unique(true_labels)) > 1:
            auroc = roc_auc_score(true_labels, energy_scores)
        else:
            auroc = -1.0 # Indicates N/A
    except ValueError:
        auroc = -1.0

    # 4. Print Summary
    print("\n" + "="*30)
    print("   FINAL EVALUATION RESULTS   ")
    print("="*30)
    print(f"Threshold Used: {threshold}")
    print("-" * 30)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    
    if auroc != -1.0:
        print(f"AUROC:          {auroc:.4f}")
    else:
        print("AUROC:          N/A (Only one class in dataset)")

    # 5. Confusion Matrix & FPR
    cm = confusion_matrix(true_labels, preds)
    print("-" * 30)
    print("Confusion Matrix:")
    print(cm)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTP (Attacks Caught): {tp}")
        print(f"TN (Benign Allowed): {tn}")
        print(f"FP (False Alarms):   {fp}")
        print(f"FN (Missed Attacks): {fn}")

        # False Positive Rate (FPR)
        if (tn + fp) > 0:
            fpr = fp / (tn + fp)
            print(f"FPR: {fpr:.4f}")
    
    print("-" * 30)
    print("Full Classification Report:")
    print(classification_report(true_labels, preds, target_names=["Benign", "Malicious"], zero_division=0))
    print("="*30)