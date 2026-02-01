import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from .config import Config

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits, _ = model(images)
            
            # Anomaly Score based on Energy (Liu et al. 2020)
            # E = -T * logsumexp(x / T)
            # Lower energy = In-distribution (Normal), Higher = Anomaly
            # But here we have a binary classifier, so we can just use Argmax or Softmax(class 1)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=["Benign", "Malicious"], output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save results
    df = pd.DataFrame(report).transpose()
    df.to_csv(Config.RESULTS_DIR / "metrics_report.csv")
    
    print("\n--- Evaluation Results ---")
    print(df)
    print("\nConfusion Matrix:")
    print(cm)