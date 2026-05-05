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

split_order = [

    "distort",
 
]

split_display_map = {
    "add_gaussian_noise_features": "Noise",
    "alternate_importance": "Alt.~Imp.",
    "distort": "Distort",
    "importance": "Imp.",
    "random": "Random",
    "rotate": "Rotate"
}

method_order = [
    # "FoSTA_gap_auto",

    "FoSTA_gap_t2",
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
# PROCESSING
# =========================================================
df = pd.read_csv(results_csv)
if "status" in df.columns:
    df = df[df["status"] == "ok"].copy()

all_summaries = {}
for m_key in metrics.keys():
    x = df[df["method"].isin(method_order)].copy()
    piv = x.pivot_table(index=["dataset", "split", "seed"], columns="method", values=m_key)
    valid_idx = piv.index[piv[method_order].notna().all(axis=1)]
    means = x.set_index(["dataset", "split", "seed"]).loc[valid_idx].reset_index()
    all_summaries[m_key] = means.groupby(["split", "method"])[m_key].mean().unstack(level=0)

# =========================================================
# HIGHLIGHTING LOGIC
# =========================================================
def get_highlighted_value(val, m_key, split_name):
    """Returns formatted value with leading zeros and ranking highlights."""
    if pd.isna(val): return "---"
    
    # Get all scores for this metric and split
    all_scores = all_summaries[m_key][split_name].sort_values(ascending=(m_key in lower_is_better))
    unique_vals = all_scores.unique()
    
    # Keeping the leading zero (standard 0.3f format)
    formatted = f"{val:.3f}"
    
    # Check ranks against thresholds
    if desired_top >= 1 and val == unique_vals[0]:
        return f"\\gold{{{formatted}}}" 
    elif desired_top >= 2 and len(unique_vals) > 1 and val == unique_vals[1]:
        return f"\\silver{{{formatted}}}" 
    elif desired_top >= 3 and len(unique_vals) > 2 and val == unique_vals[2]:
        return f"\\bronze{{{formatted}}}" 
    
    return formatted

# =========================================================
# LATEX GENERATION
# =========================================================
print("\n" + "="*30)
print(f"LATEX TABLE OUTPUT (LEADING ZEROS INCLUDED)")
print("="*30 + "\n")

header = "Model "
for s in split_order:
    header += f"& \multicolumn{{3}}{{c}}{{{split_display_map[s]}}} "
print(header + "\\\\")

sub_header = " "
for _ in split_order:
    sub_header += "& Acc & AS & FOS "
print(sub_header + "\\\\")
print("\\midrule")

for m in method_order:
    row_parts = [method_display_map[m]]
    for s in split_order:
        acc_val = all_summaries["label_transfer"].loc[m, s]
        as_val = all_summaries["alignment_score"].loc[m, s]
        fos_val = all_summaries["foscttm"].loc[m, s]
        
        row_parts.append(get_highlighted_value(acc_val, "label_transfer", s))
        row_parts.append(get_highlighted_value(as_val, "alignment_score", s))
        row_parts.append(get_highlighted_value(fos_val, "foscttm", s))
    
    print(" & ".join(row_parts) + " \\\\")

print("\\bottomrule")