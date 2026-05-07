import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

# results_csv = "results_uci/results_20260503_015528_general.csv"
results_csv = "results_uci/results_20260503_122708_general_distort05.csv"
# results_csv = "results_uci/results_20260503_201720_other_ablation.csv"

desired_top = 3
metrics = {"label_transfer": "Acc", "alignment_score": "AS", "foscttm": "FOS"}
lower_is_better = ["foscttm"]

# split_order = ["add_gaussian_noise_features", "alternate_importance", "distort", "importance", "random", "rotate"]

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
    # "FoSTA_et",

    "FoSTA_gap_t2",
    "FoSTA_kerf_auto",

    # "FoSTA_dense",

    # "FoSTA_umap",
]

method_display_map = {
    "FoSTA_gap_auto": "FoSTA (Default)",
    "FoSTA_gap_t2": "FoSTA ($t=2$)",
    "FoSTA_kerf_auto": "FoSTA-KeRF (auto)",
    "FoSTA_kerf_t2": "FoSTA-KeRF ($t=2$)",
    "FoSTA_umap": "FoSTA-UMAP",
    "FoSTA_dense": "FoSTA-Dense",
    "FoSTA_et": "FoSTA-et",
}

# =========================================================
# REFERENCE DATA
# =========================================================
ref_scores = {
    "label_transfer": {
        "add_gaussian_noise_features": [0.776, 0.760, 0.787, 0.776],
        "alternate_importance": [0.722, 0.721, 0.753, 0.767],
        "distort": [0.785, 0.811, 0.783, 0.794],
        "importance": [0.708, 0.710, 0.720, 0.782],
        "random": [0.682, 0.689, 0.732, 0.760],
        "rotate": [0.818, 0.834, 0.795, 0.808],
    },
    "alignment_score": {
        "add_gaussian_noise_features": [0.867, 0.865, 0.385, 0.430],
        "alternate_importance": [0.988, 0.523, 0.308, 0.568],
        "distort": [1.027, 0.939, 0.427, 0.568],
        "importance": [0.919, 0.471, 0.308, 0.500],
        "random": [0.982, 0.507, 0.312, 0.562],
        "rotate": [1.044, 1.017, 0.447, 0.597],
    },
    "foscttm": {
        "add_gaussian_noise_features": [0.303, 0.400, 0.301, 0.325],
        "alternate_importance": [0.364, 0.358, 0.373, 0.331],
        "distort": [0.204, 0.164, 0.310, 0.307],
        "importance": [0.374, 0.341, 0.385, 0.337],
        "random": [0.384, 0.350, 0.379, 0.336],
        "rotate": [0.098, 0.075, 0.233, 0.263],
    }
}


def get_highlighted_value(val, m_key, split_name):
    if pd.isna(val): return "---"
    baselines = ref_scores[m_key][split_name]
    competition = np.array(baselines + [val])
    is_lower = (m_key in lower_is_better)
    sorted_unique = sorted(np.unique(competition), reverse=not is_lower)
    
    formatted = f"{val:.3f}"
    
    if val == sorted_unique[0]:
        return f"\\gold{{{formatted}}}"
    elif desired_top >= 2 and len(sorted_unique) > 1 and val == sorted_unique[1]:
        return f"\\silver{{{formatted}}}"
    elif desired_top >= 3 and len(sorted_unique) > 2 and val == sorted_unique[2]:
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
    # Filter methods
    x = df[df["method"].isin(method_order)].copy()
    
    # Calculate Mean (Global average for the top row)
    all_summaries[m_key] = x.groupby(["split", "method"])[m_key].mean().unstack(level=0)
    
    # Calculate Mean of STDs per dataset (for the error row)
    # 1. STD per dataset
    stds_per_dataset = x.groupby(["split", "method", "dataset"])[m_key].std()
    # 2. Mean of those STDs
    mean_of_stds = stds_per_dataset.groupby(["split", "method"]).mean()
    all_stds[m_key] = mean_of_stds.unstack(level=0)

# =========================================================
# LATEX GENERATION
# =========================================================
print("\n" + "="*40)
print("ABLATION TABLE (SCORES + MEAN-OF-STDS)")
print("="*40 + "\n")

header = "Model "
sub_header = " "
for s in split_order:
    header += fr"& \multicolumn{{3}}{{c}}{{{split_display_map[s]}}} "
    sub_header += "& Acc & AS & FOS "

print(header + " \\\\")
print(sub_header + " \\\\")
print("\\midrule")

for m in method_order:
    if m not in all_summaries["label_transfer"].index:
        continue
        
    # --- ROW 1: MEANS ---
    row_means = [method_display_map.get(m, m)]
    for s in split_order:
        acc_v = all_summaries["label_transfer"].loc[m, s]
        as_v = all_summaries["alignment_score"].loc[m, s]
        fos_v = all_summaries["foscttm"].loc[m, s]
        
        row_means.append(get_highlighted_value(acc_v, "label_transfer", s))
        row_means.append(get_highlighted_value(as_v, "alignment_score", s))
        row_means.append(get_highlighted_value(fos_v, "foscttm", s))
    
    print(" & ".join(row_means) + " \\\\")

    # --- ROW 2: ERRORS (Mean of STDs) ---
    row_errs = [r"\scriptsize{$\pm$ std.}"]
    for s in split_order:
        acc_e = all_stds["label_transfer"].loc[m, s]
        as_e = all_stds["alignment_score"].loc[m, s]
        fos_e = all_stds["foscttm"].loc[m, s]
        
        # Formatting to 2 decimals with math-mode \pm
        row_errs.append(fr"\scriptsize{{$\pm${acc_e:.2f}}}" if not pd.isna(acc_e) else " ")
        row_errs.append(fr"\scriptsize{{$\pm${as_e:.2f}}}" if not pd.isna(as_e) else " ")
        row_errs.append(fr"\scriptsize{{$\pm${fos_e:.2f}}}" if not pd.isna(fos_e) else " ")
    
    print(" & ".join(row_errs) + r" \\[0.5ex]")

print("\\bottomrule")