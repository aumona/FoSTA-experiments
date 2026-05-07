import pandas as pd
import glob
import os

# --- CONFIG ---
# List the full paths to the timestamp folders you want to aggregate
SELECTED_TIMESTAMPS = [
    "/Users/aumona/Projects/RF-MALI/result_sc_ad/20260506_192321_scvi_anvi_liger_fosta_seed42_batchA",
    "/Users/aumona/Projects/RF-MALI/result_sc_ad/20260506_202623", # Example of another run
]

# 1. Collect files from all selected timestamp folders
all_files = []
for base_dir in SELECTED_TIMESTAMPS:
    if not os.path.exists(base_dir):
        print(f"Warning: Directory not found: {base_dir}")
        continue
        
    # Find all benchmark CSVs within this specific timestamp folder
    found_files = glob.glob(os.path.join(base_dir, "**/benchmark_metrics_progressive.csv"), recursive=True)
    all_files.extend(found_files)

# 2. Process and Aggregate
if not all_files:
    print("No results found in the provided timestamp folders.")
else:
    # Load and stack all dataframes
    df_list = [pd.read_csv(f, index_col=0) for f in all_files]
    full_df = pd.concat(df_list)

    # 3. Aggregate by Method
    # This takes the mean across every batch pair found in every selected timestamp
    summary = full_df.groupby("Method").mean(numeric_only=True)

    # 4. Output LaTeX Table
    print(f"\n--- Aggregated Results ---")
    print(f"Total CSVs found: {len(all_files)}")
    print(f"Timestamps included: {[os.path.basename(t) for t in SELECTED_TIMESTAMPS]}\n")
    
    print(summary.to_latex(float_format="%.3f", bold_rows=True))