import pandas as pd
import numpy as np
import re

def calculate_weighted_metrics(file_path):
    """
    Reads the confusion matrices and calculates the Weighted Precision, 
    Weighted Recall, Weighted F1-Score, and Weighted Accuracy for each model.
    Weights are based on the actual support (number of samples) in each test set.
    """
    print(f"Loading data from {file_path} to calculate weighted metrics...")
    df = pd.read_csv(file_path)
    attacks = df['Attacks'].dropna().tolist()
    
    results = []
    
    # Iterate over the file in pairs of rows (each pair represents one training attack)
    for i in range(0, len(df), 2):
        row_top = df.iloc[i]
        row_bottom = df.iloc[i+1]
        train_attack = attacks[i//2]
        
        # Accumulators for the weighted averages
        weighted_precision_sum = 0
        weighted_recall_sum = 0
        weighted_f1_sum = 0
        weighted_acc_sum = 0
        
        total_pos_support = 0  # Support for Positive class (Actual Attacks)
        total_support = 0      # Total support (Attacks + Benign)
        
        for test_attack in df.columns[1:]:
            # Ignore self-testing (main diagonal)
            if train_attack == test_attack:
                continue
                
            val_top = str(row_top[test_attack])
            val_bottom = str(row_bottom[test_attack])
            
            if pd.isna(row_top[test_attack]) or val_top == 'nan':
                continue
                
            # Extract [TN FP] and [FN TP]
            m_top = re.findall(r'\d+', val_top)
            m_bot = re.findall(r'\d+', val_bottom)
            
            if len(m_top) == 2 and len(m_bot) == 2:
                tn, fp = int(m_top[0]), int(m_top[1])
                fn, tp = int(m_bot[0]), int(m_bot[1])
                
                # Calculate Support
                pos_support = tp + fn
                all_support = tp + tn + fp + fn
                
                if all_support == 0:
                    continue
                    
                # Calculate Base Metrics for this specific test
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / all_support
                
                # Add to weighted sums
                # P, R, F1 are weighted by the number of actual attack samples
                weighted_precision_sum += precision * pos_support
                weighted_recall_sum += recall * pos_support
                weighted_f1_sum += f1 * pos_support
                total_pos_support += pos_support
                
                # Accuracy is weighted by the total number of samples (Benign + Attack)
                weighted_acc_sum += accuracy * all_support
                total_support += all_support
        
        # Calculate the final weighted averages for this training model
        final_precision = weighted_precision_sum / total_pos_support if total_pos_support > 0 else 0
        final_recall = weighted_recall_sum / total_pos_support if total_pos_support > 0 else 0
        final_f1 = weighted_f1_sum / total_pos_support if total_pos_support > 0 else 0
        final_acc = weighted_acc_sum / total_support if total_support > 0 else 0
        
        results.append({
            'Train Attack (OE)': train_attack,
            'Weighted Precision': final_precision,
            'Weighted Recall': final_recall,
            'Weighted F1-Score': final_f1,
            'Weighted Accuracy': final_acc
        })
        
    metrics_df = pd.DataFrame(results)
    metrics_df.set_index('Train Attack (OE)', inplace=True)
    return metrics_df

if __name__ == "__main__":
    # Ensure this points to your original CSV
    file_path = "../results/oe_results.csv" 
    
    weighted_metrics_df = calculate_weighted_metrics(file_path)
    
    # Print the results nicely rounded to 4 decimal places
    print("\n" + "="*70)
    print("WEIGHTED METRICS PER TRAINING ATTACK (GENERALIZATION) 🌟")
    print("="*70)
    print(weighted_metrics_df.round(4))
    
    # Save to CSV
    output_csv_path = "../results/weighted_metrics_summary.csv"
    weighted_metrics_df.to_csv(output_csv_path)
    print(f"\nWeighted metrics successfully saved to '{output_csv_path}'")
    
    # Quick Insights
    print("\n🏆 Top 3 Models by Weighted F1-Score:")
    print(weighted_metrics_df['Weighted F1-Score'].sort_values(ascending=False).head(3).round(4))
    
    print("\n🏆 Top 3 Models by Weighted Accuracy:")
    print(weighted_metrics_df['Weighted Accuracy'].sort_values(ascending=False).head(3).round(4))