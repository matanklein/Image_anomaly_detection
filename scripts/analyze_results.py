import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_parse_weighted_data(file_path):
    """
    Reads the results file and converts the cells (confusion matrices) 
    to WEIGHTED Recall and F1-Score metrics based on class support.
    Ignores the main diagonal (self-testing) for correct generalization metrics.
    """
    print("Loading and parsing data for WEIGHTED Recall and F1-Score analysis...")
    df = pd.read_csv(file_path)
    attacks = df['Attacks'].dropna().tolist()
    
    # --- Step 1: Calculate the support (number of actual attacks) for each test column ---
    # We assume the test dataset for a specific attack is constant across all training models.
    # We take the first training model's row to extract the support sizes.
    supports = {}
    row_bottom_first = df.iloc[1]
    
    for col in df.columns[1:]:
        val_bottom = str(row_bottom_first[col])
        m_bot = re.findall(r'\d+', val_bottom)
        if len(m_bot) == 2:
            fn, tp = int(m_bot[0]), int(m_bot[1])
            supports[col] = tp + fn
        else:
            supports[col] = 0

    total_support = sum(supports.values())
    if total_support == 0:
        print("Warning: Total support is 0. Check data formatting.")
        total_support = 1 # Prevent division by zero later
        
    recall_results = []
    f1_results = []
    
    # --- Step 2: Calculate metrics and apply weights ---
    for i in range(0, len(df), 2):
        row_top = df.iloc[i]
        row_bottom = df.iloc[i+1]
        
        row_recall = {}
        row_f1 = {}
        
        for col in df.columns[1:]:
            val_top = str(row_top[col])
            val_bottom = str(row_bottom[col])
            
            if pd.isna(row_top[col]) or val_top == 'nan':
                row_recall[col] = np.nan
                row_f1[col] = np.nan
                continue
                
            m_top = re.findall(r'\d+', val_top)
            m_bot = re.findall(r'\d+', val_bottom)
            
            if len(m_top) == 2 and len(m_bot) == 2:
                tn, fp = int(m_top[0]), int(m_top[1])
                fn, tp = int(m_bot[0]), int(m_bot[1])
                
                # Raw metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # We store the RAW values in the DataFrame for the heatmap to look normal (0 to 1),
                # but we will calculate the weighted averages during the Insights phase.
                row_recall[col] = recall
                row_f1[col] = f1
            else:
                row_recall[col] = np.nan
                row_f1[col] = np.nan
                
        recall_results.append(row_recall)
        f1_results.append(row_f1)

    recall_df = pd.DataFrame(recall_results, index=attacks)
    f1_df = pd.DataFrame(f1_results, index=attacks)
    
    # Ignore self-testing results (train == test)
    for attack in attacks:
        if attack in recall_df.columns:
            recall_df.loc[attack, attack] = np.nan
        if attack in f1_df.columns:
            f1_df.loc[attack, attack] = np.nan
            
    return recall_df, f1_df, attacks, supports

def prove_weighted_insights(df, attacks, supports, metric_name="Recall"):
    print("\n" + "="*50)
    print(f"PROVING WEIGHTED INSIGHTS FOR: {metric_name.upper()}")
    print("="*50)
    
    # Calculate column averages (Test Attacks difficulty) - simple average is usually fine here
    # as we are looking at the attack itself across different models.
    col_means = df.mean()
    
    print(f"\n--- Insight 1: Easiest Attacks to Detect (Average {metric_name}) ---")
    print(col_means.sort_values(ascending=False).head(3).round(3))

    print(f"\n--- Insight 2: Hardest Attacks to Detect (Average {metric_name}) ---")
    print(col_means.sort_values(ascending=True).head(3).round(3))

    # --- Insight 3: Best single attack for training (WEIGHTED AVERAGE) ---
    print(f"\n--- Insight 3: Best Single Attack for Outlier Exposure (WEIGHTED {metric_name}) ---")
    
    weighted_row_means = {}
    for train_attack in df.index:
        weighted_sum = 0
        valid_support_sum = 0
        
        for test_attack in df.columns:
            val = df.loc[train_attack, test_attack]
            if not pd.isna(val):
                weight = supports.get(test_attack, 0)
                weighted_sum += val * weight
                valid_support_sum += weight
                
        if valid_support_sum > 0:
            weighted_row_means[train_attack] = weighted_sum / valid_support_sum
        else:
            weighted_row_means[train_attack] = 0

    weighted_row_series = pd.Series(weighted_row_means)
    best_trainers = weighted_row_series.sort_values(ascending=False).head(3)
    print(f"Top 3 training attacks by overall WEIGHTED average {metric_name}:")
    print(best_trainers.round(3))

    # --- Insight 4: Complementary pairs (using RAW values threshold, but evaluating by WEIGHTED coverage) ---
    print(f"\n--- Insight 4: The Perfect Complementary Pair ({metric_name} >= 0.8) ---")
    best_pair = None
    max_covered_support = 0
    threshold = 0.8 
    
    total_valid_support = sum(supports.values())

    for i in range(len(attacks)):
        for j in range(i+1, len(attacks)):
            attack1 = attacks[i]
            attack2 = attacks[j]
            
            row1 = df.loc[attack1].fillna(1.0)
            row2 = df.loc[attack2].fillna(1.0)
            
            combined_metric = np.maximum(row1, row2)
            
            # Calculate how much *support* (actual attacks) is covered by this pair
            covered_support = sum(supports[col] for col in df.columns if combined_metric[col] >= threshold)
            
            if covered_support > max_covered_support:
                max_covered_support = covered_support
                best_pair = (attack1, attack2)

    coverage_percent = (max_covered_support / total_valid_support) * 100 if total_valid_support > 0 else 0

    print(f"Best Pair Found: {best_pair[0]} & {best_pair[1]}")
    print(f"Percentage of actual attack samples covered (>{threshold*100}% {metric_name}): {coverage_percent:.2f}%")
    if coverage_percent == 100:
        print("PROVED: This pair perfectly covers 100% of the attack samples!")

def generate_heatmap(df, metric_name, output_filename):
    print(f"\nGenerating {metric_name} Heatmap...")
    plt.figure(figsize=(16, 10))
    
    sns.heatmap(df, 
                annot=False,          
                cmap="YlGnBu",        
                cbar_kws={'label': metric_name},
                vmin=0, vmax=1)
    
    plt.title(f"Anomaly Detection {metric_name}: Train (OE) vs Test Attacks", fontsize=16, pad=20)
    plt.xlabel("Tested Attacks (Generalization)", fontsize=12)
    plt.ylabel("Trained Attack (Outlier Exposure)", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap successfully saved as '{output_filename}'")

if __name__ == "__main__":
    file_path = "../results/oe_results.csv"
    
    recall_df, f1_df, attacks, supports = load_and_parse_weighted_data(file_path)
    
    recall_csv_path = "../results/recall_matrix.csv"
    f1_csv_path = "../results/f1_matrix.csv"
    
    recall_df.to_csv(recall_csv_path)
    f1_df.to_csv(f1_csv_path)
    print(f"\nMatrices successfully saved to '{recall_csv_path}' and '{f1_csv_path}'")
    
    # Print WEIGHTED insights
    prove_weighted_insights(recall_df, attacks, supports, metric_name="Recall")
    prove_weighted_insights(f1_df, attacks, supports, metric_name="F1-Score")
    
    generate_heatmap(recall_df, "Recall", "../results/recall_heatmap.png")
    generate_heatmap(f1_df, "F1-Score", "../results/f1_heatmap.png")