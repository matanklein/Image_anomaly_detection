import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_weighted_f1(csv_path, output_image_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please ensure the previous script generated it.")
        return

    # Sort the dataframe by Weighted F1-Score in descending order
    df_sorted = df.sort_values(by='Weighted F1-Score', ascending=False)

    # Set up the plot style and figure size
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Create a horizontal bar chart with a nice color palette
    ax = sns.barplot(
        x='Weighted F1-Score', 
        y='Train Attack (OE)', 
        data=df_sorted, 
        palette="viridis"
    )

    # Add titles and axis labels
    plt.title('Anomaly Detection Performance by OE Training Attack\n(Weighted F1-Score)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Weighted F1-Score (Generalization on Unseen Attacks)', fontsize=14)
    plt.ylabel('Training Attack (Outlier Exposure)', fontsize=14)

    # Add the exact value labels to the end of each bar
    for p in ax.patches:
        width = p.get_width()
        # Only add text if the width is greater than 0
        if width > 0:
            plt.text(width + 0.01, p.get_y() + p.get_height() / 2, 
                     f'{width:.3f}', 
                     ha='left', va='center', fontsize=10, color='black')

    # Extend x-axis limit slightly to make room for the text labels
    plt.xlim(0, max(df['Weighted F1-Score']) + 0.1)
    
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_image_path, dpi=300)
    print(f"🎯 Bar chart successfully saved to '{output_image_path}'")

if __name__ == "__main__":
    # Input file from the previous step
    input_csv = "../results/weighted_metrics_summary.csv"
    # Output image file
    output_image = "../results/weighted_f1_barchart.png"
    
    plot_weighted_f1(input_csv, output_image)