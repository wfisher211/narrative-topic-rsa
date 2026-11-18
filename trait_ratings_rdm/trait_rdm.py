# %%
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import argparse


# %%
# SET MAIN HYPERPARAMETERS
p = argparse.ArgumentParser()
p.add_argument("--stim", required=True)
args = p.parse_args()
STIMULUS_LABEL_SAVE_STRING = args.stim

# Dictionary mapping save string to full label
stimulus_map = {
    "pieman": "1Pieman",
    "slumlordreach": "2Slumlord",
    "reachforstars": "3ReachStars",
    "notthefallintact": "4NotTheFall",
    "black": "5Black",
    "forgot": "6forgot",
    }

# Automatically set STIMULUS_LABEL based on STIMULUS_LABEL_SAVE_STRING
STIMULUS_LABEL = stimulus_map.get(STIMULUS_LABEL_SAVE_STRING)    

CSV_FILE = "/burg/psych/users/wf2315/narratives-fmri/fmriprep/trait_ratings_data/chujun_data.csv"
OUT_DIR = Path("/burg/psych/users/wf2315/narratives-fmri/fmriprep/derivatives/RDMs_behavior")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Define the traits to be used

TRAIT_LIST = [
    "Open-minded",
    "feeling Affectionate",
    "Attentive",
    "Assertive",
    "feeling Gloomy",
    "feeling Peaceful",
    "Agreeable",
    "Judging",
    "feeling Angry",
    "feeling Bewildered",
    "Impulsive",
    "Self-disciplined",
    "Contemplating",
]


# %%
# Define the smoothing window for each trait
WIN_DICT: dict[str, int] = {
    "Open-minded"      : 200,
    "feeling Affectionate": 200,
    "Attentive"        : 200,
    "Assertive"        : 200,
    "feeling Gloomy"   : 200,
    "feeling Peaceful" : 175,
    "Agreeable"        : 200,
    "Judging"          : 150,
    "feeling Angry"    : 0,
    "feeling Bewildered": 50,
    "Impulsive"        : 25,
    "Self-disciplined" : 0,
    "Contemplating"    : 0,
}

# %%
# Toggle smoothing on/off 
ENABLE_SMOOTHING = False

# %%
# --------------------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------------------

def build_behavior_template(
        csv_path       : str | Path,
        stim_label     : str,
        trait_label    : str,
        win_dict       : dict[str, int],
        apply_smoothing: bool = True,
        tr_sec         : float = 1.5,
        n_tr           : int = 160,
        sd_cutoff      : float = 5.0,
        max_sec        : float = 240.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # (1) Load and filter data
    df = pd.read_csv(csv_path)
    df = df[(df["stim"] == stim_label)
            & (df["task"] == trait_label)
            & (df["time"]  < max_sec)].copy()

    # (2) exclude raters whose SD < sd_cutoff
    sd = df.groupby("subject")["rating"].std()
    bad_sd = sd[sd < sd_cutoff].index
    df = df[~df["subject"].isin(bad_sd)]
    # print(f"Excluded {len(bad_sd)} raters (SD < {sd_cutoff}). Kept {df['subject'].nunique()} subjects.")

    # (3) Map samples to TR bins (0‥159)
    df["TR"] = (df["time"] / tr_sec).round().astype(int)
    df = df[(df["TR"] >= 0) & (df["TR"] < n_tr)]

    # (4) Smooth ratings only if enabled
    window = win_dict.get(trait_label, 0)
    if not apply_smoothing or window < 1:
        # print(f"[{trait_label}] smoothing skipped (enabled={apply_smoothing}, window={window})")
        df["rating_sm"] = df["rating"]
    else:
        # print(f"[{trait_label}] smoothing window={window} samples")
        df["rating_sm"] = (
            df.groupby("subject")["rating"]
              .apply(lambda s: s.rolling(window, center=True, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )

    # (5) Average across raters within each TR
    pivot = (df.pivot_table(index="TR", columns="subject",
                            values="rating_sm", aggfunc="mean")
               .reindex(range(n_tr))
               .interpolate(limit_direction="both"))
    vec = pivot.mean(axis=1).values

    # (6) Compute dissimilarity matrix (Manhattan distance)
    RDM = squareform(pdist(vec[:, None], metric="cityblock"))

    return vec, RDM, df

# %% Loop over traits: build & save
for trait in TRAIT_LIST:
    trait_save_str = trait.replace(" ", "_").replace("-", "_")
    # print(f"\n=== Processing trait: {trait} ===")
    vec, RDM, df = build_behavior_template(
        csv_path        = CSV_FILE,
        stim_label      = STIMULUS_LABEL,
        trait_label     = trait,
        win_dict        = WIN_DICT,
        apply_smoothing = ENABLE_SMOOTHING,
    )
    smoothing_setting = "" if ENABLE_SMOOTHING else "_no_smoothing"
    np.save(OUT_DIR / f"{STIMULUS_LABEL_SAVE_STRING}_{trait_save_str}_vec{smoothing_setting}.npy", vec)
    np.save(OUT_DIR / f"{STIMULUS_LABEL_SAVE_STRING}_{trait_save_str}_RDM{smoothing_setting}.npy", RDM)
    print(f"Saved vector and RDM for {trait} with suffix '{smoothing_setting}'")



# %% Sanity-check
assert vec.shape == (160,),  "Vector must be length 160"
assert RDM.shape == (160,160), "RDM must be 160×160"
assert np.allclose(RDM, RDM.T), "RDM not symmetric"
assert np.allclose(np.diag(RDM), 0), "Diagonal must be zeros"
assert np.allclose(RDM, squareform(pdist(vec[:,None], 'cityblock'))), \
       "RDM mismatch"
print("All checks passed ✔️")


