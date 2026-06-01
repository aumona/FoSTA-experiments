import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================


# results_csv = "results_uci/results_20260503_015528_general.csv"

results_csv = "results_uci/results_20260503_122708_general_distort05.csv"


# Set this to 1, 2, or 3 to control how many top ranks are colored
desired_top = 3

metrics = {
    "label_transfer": "Acc",
    "alignment_score": "AS",
    "foscttm": "FOS"
}

lower_is_better = ["foscttm"]

# split_order = [
#     "add_gaussian_noise_features",
#     "alternate_importance",
#     "distort",
#     "importance",
#     "random",
#     "rotate"
# ]

split_order = ["distort"]


split_display_map = {
    "add_gaussian_noise_features": "Noise",
    "alternate_importance": "Alt.~Imp.",
    "distort": "Distort",
    "importance": "Imp.",
    "random": "Random",
    "rotate": "Rotate"
}

method_order = [
    "FoSTA_gap_auto",

    # "FoSTA_gap_t2",
    # "FoSTA_gap_auto",
    # "FoSTA_kerf_t2",
    # "FoSTA_kerf_auto",



    "MALI", 
    # "MALI_nodpt",
    "Pamona", 
    "KEMAlin", 
    "KEMArbf"
]

method_display_map = {
    "FoSTA_ICML_auto": "FoSTA_ICML ($t=\\texttt{auto}$)",
    "FoSTA_ICML_t2": "FoSTA_ICML ($t=2$)",
    "FoSTA_gap_t2": "FoSTA ($t=2$)",
    "FoSTA_gap_auto": "FoSTA ($t=\\texttt{auto}$)",
    "FoSTA_gap_mauto_t2": "FoSTA (mauto, $t=2$)",
    "FoSTA_kerf_t2": "FoSTA-KeRF ($t=2$)",
    "FoSTA_kerf_auto": "FoSTA-KeRF ($t=\\texttt{auto}$)",
    "FoSTA_kerf_mauto_t2": "FoSTA-KeRF (mauto, $t=2$)",
    "MALI": "MALI", 
    "MALI_nodpt": "MALI (w/o DPT)",
    "Pamona": "Pamona", 
    "KEMAlin": "KEMAlin", 
    "KEMArbf": "KEMArbf"
}



# =========================================================
# HIGHLIGHTING LOGIC
# =========================================================
def get_highlighted_value(val, m_key, split_name):
    if pd.isna(val): return "---"
    all_scores = all_summaries[m_key][split_name].sort_values(ascending=(m_key in lower_is_better))
    unique_vals = all_scores.unique()
    formatted = f"{val:.3f}"
    
    if desired_top >= 1 and val == unique_vals[0]:
        return f"\\gold{{{formatted}}}" 
    elif desired_top >= 2 and len(unique_vals) > 1 and val == unique_vals[1]:
        return f"\\silver{{{formatted}}}" 
    elif desired_top >= 3 and len(unique_vals) > 2 and val == unique_vals[2]:
        return f"\\bronze{{{formatted}}}" 
    return formatted

# =========================================================
# PROCESSING
# =========================================================
df = pd.read_csv(results_csv)
if "status" in df.columns:
    df = df[df["status"] == "ok"].copy()

all_summaries = {}
all_stds = {} 

for m_key in metrics.keys():
    x = df[df["method"].isin(method_order)].copy()
    piv = x.pivot_table(index=["dataset", "split", "seed"], columns="method", values=m_key)
    valid_idx = piv.index[piv[method_order].notna().all(axis=1)]
    
    # Filter for valid runs
    filtered = x.set_index(["dataset", "split", "seed"]).loc[valid_idx].reset_index()
    
    # 1. Calculate Mean (Average of all scores)
    all_summaries[m_key] = filtered.groupby(["split", "method"])[m_key].mean().unstack(level=0)
    
    # 2. NEW LOGIC: Calculate STD per dataset, then take the Mean of those STDs
    stds_per_dataset = filtered.groupby(["split", "method", "dataset"])[m_key].std()
    mean_of_stds = stds_per_dataset.groupby(["split", "method"]).mean()
    all_stds[m_key] = mean_of_stds.unstack(level=0)

# =========================================================
# LATEX GENERATION - TWO-ROW FORMAT
# =========================================================
print("\n" + "="*50)
print(f"LATEX TABLE: TWO-ROW FORMAT (SCORES + STDS)")
print("="*50 + "\n")

header_row = "Model "
sub_header = " "
for s in split_order:
    header_row += f"& \multicolumn{{3}}{{c}}{{{split_display_map[s]}}} "
    sub_header += "& Acc & AS & FOS "

print(header_row + "\\\\")
print(sub_header + "\\\\")
print("\\midrule")

for m in method_order:
    # --- ROW 1: MEANS ---
    row_means = [method_display_map[m]]
    for s in split_order:
        row_means.append(get_highlighted_value(all_summaries["label_transfer"].loc[m, s], "label_transfer", s))
        row_means.append(get_highlighted_value(all_summaries["alignment_score"].loc[m, s], "alignment_score", s))
        row_means.append(get_highlighted_value(all_summaries["foscttm"].loc[m, s], "foscttm", s))
    print(" & ".join(row_means) + " \\\\")

    # --- ROW 2: ERRORS (Standard Deviation) ---
    # UPDATED: Label changed to "std." to reflect the metric change
    row_errs = ["\\scriptsize{$\\pm$ std.}"] 
    for s in split_order:
        acc_e = all_stds["label_transfer"].loc[m, s]
        as_e = all_stds["alignment_score"].loc[m, s]
        fos_e = all_stds["foscttm"].loc[m, s]
        
        # Formatting to 2 decimals with math-mode \pm
        row_errs.append(f"\\scriptsize{{$\\pm${acc_e:.2f}}}" if not pd.isna(acc_e) else " ")
        row_errs.append(f"\\scriptsize{{$\\pm${as_e:.2f}}}" if not pd.isna(as_e) else " ")
        row_errs.append(f"\\scriptsize{{$\\pm${fos_e:.2f}}}" if not pd.isna(fos_e) else " ")
    
    print(" & ".join(row_errs) + " \\\\[0.5ex]")

print("\\bottomrule")