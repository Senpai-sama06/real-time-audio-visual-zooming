import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src import config

def analyze():
    csv_path = os.path.join(config.RESULTS_DIR, "batch_metrics.csv")
    
    if not os.path.exists(csv_path):
        print("No batch results found. Run batch_run.py first.")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    
    print("\n=== SUMMARY STATISTICS ===")
    print(df.describe().transpose()[['mean', 'std', 'min', 'max']])

    # Create Visualization Folder
    viz_dir = os.path.join(config.RESULTS_DIR, "analysis_plots")
    os.makedirs(viz_dir, exist_ok=True)

    # --- PLOT 1: Box Plots of Metrics ---
    plt.figure(figsize=(12, 6))
    
    # Melt dataframe for seaborn
    metrics_to_plot = ['SIR_Imp', 'PESQ_WB', 'STOI']
    df_melt = df.melt(value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    sns.boxplot(x="Metric", y="Score", data=df_melt)
    plt.title(f"Performance Distribution (N={len(df)})")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(viz_dir, "metrics_boxplot.png")
    plt.savefig(save_path)
    print(f"\n[PLOT] Saved Boxplot: {save_path}")

    # --- PLOT 2: SIR Improvement Histogram ---
    plt.figure(figsize=(8, 5))
    sns.histplot(df['SIR_Imp'], kde=True, color='green')
    plt.title("Distribution of SIR Improvement")
    plt.xlabel("Improvement (dB)")
    
    save_path = os.path.join(viz_dir, "sir_hist.png")
    plt.savefig(save_path)
    print(f"[PLOT] Saved Histogram: {save_path}")
    
    # --- PLOT 3: Scatter (Input SIR vs Output SIR) ---
    plt.figure(figsize=(8, 5))
    plt.scatter(df['SIR_Base'], df['SIR_Enh'], alpha=0.6)
    plt.plot([df['SIR_Base'].min(), df['SIR_Base'].max()], 
             [df['SIR_Base'].min(), df['SIR_Base'].max()], 'r--', label='No Improvement')
    plt.xlabel("Input SIR (dB)")
    plt.ylabel("Output SIR (dB)")
    plt.title("Input vs Output SIR")
    plt.legend()
    
    save_path = os.path.join(viz_dir, "sir_scatter.png")
    plt.savefig(save_path)
    print(f"[PLOT] Saved Scatter: {save_path}")

if __name__ == "__main__":
    analyze()

### How to use this workflow

# 1.  **Install Analysis Libs:**
#     ```bash
#     pip install pandas matplotlib seaborn tqdm
#     ```

# 2.  **Run the Batch:**
#     Run 50 experiments with 2 interferers.
#     ```bash
#     python batch_run.py --n 50 --interferers 2
#     ```
#     *This will take some time. It will populate `data/results/batch_metrics.csv`.*

# 3.  **Analyze:**
#     Once the batch is done (or even while it's running), run:
#     ```bash
#     python analyze_results.py