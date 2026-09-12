import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
# List the full paths to the timestamp folders you want to aggregate
SELECTED_TIMESTAMPS = [
    "./results_sc_experiments/20260506_233231",
]

# 1. Collect files from all selected timestamp folders
all_files = []
for base_dir in SELECTED_TIMESTAMPS:
    if not os.path.exists(base_dir):
        print(f"Warning: Directory not found: {base_dir}")
        continue
    
    # Updated pattern: Find all CSVs starting with 'benchmark_metrics_seed'
    # This will crawl through seed_42, seed_123, etc., and all batch-pair subfolders
    found_files = glob.glob(os.path.join(base_dir, "**/benchmark_metrics_seed*.csv"), recursive=True)
    all_files.extend(found_files)

# 2. Process and Aggregate
if not all_files:
    print("No results found in the provided timestamp folders.")
    print("Check if the directory structure is: base_dir/seed_X/pair_name/benchmark_metrics_seedX.csv")
else:
    # Load and stack all dataframes
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f, index_col=0)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    full_df = pd.concat(df_list)

    # 3. Aggregate by Method
    # We group by 'Method' (the index) and calculate the mean across all seeds and pairs
    summary = full_df.groupby(full_df.index).mean(numeric_only=True)
    
    # Remove the 'Seed' column from the final summary averages if it exists
    if "Seed" in summary.columns:
        summary = summary.drop(columns=["Seed"])

    # Optional: Sort by Total mean to make the plot easier to read
    if "Total mean" in summary.columns:
        summary = summary.sort_values("Total mean", ascending=False)

    # 4. Output LaTeX Table
    print(f"\n--- Aggregated Results ---")
    print(f"Total CSVs found: {len(all_files)}")
    print(f"Timestamps included: {[os.path.basename(t) for t in SELECTED_TIMESTAMPS]}\n")
    
    # Using specific columns for the printed table to keep it clean
    cols_to_show = ["Bio conservation mean", "Batch correction mean", "Total mean"]
    print(summary[cols_to_show].to_latex(float_format="%.3f", bold_rows=True))

    # 5. --- PLOTTING ---
    metrics_to_plot = ["Bio conservation mean", "Batch correction mean"]
    
    # Filter to ensure we only plot columns that exist
    metrics_to_plot = [m for m in metrics_to_plot if m in summary.columns]
    
    ax = summary[metrics_to_plot].plot(
        kind='bar', 
        figsize=(10, 6), 
        width=0.8,
        color=['#2ca02c', '#1f77b4'] # Green for Bio, Blue for Batch
    )

    plt.title("Integration Performance (Aggregated over Seeds and Pairs)", fontsize=14)
    plt.ylabel("Score (Mean)", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Save the plot back to the first timestamp directory
    plot_path = os.path.join(SELECTED_TIMESTAMPS[0], "aggregated_summary_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.show()