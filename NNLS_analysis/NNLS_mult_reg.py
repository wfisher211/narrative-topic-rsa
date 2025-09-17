# %%
import os, glob, re, csv
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from nilearn import datasets
from nilearn.image import resample_to_img
import pandas as pd
from scipy.stats import ttest_1samp
from nilearn import plotting, datasets
import scipy.stats as stats
from scipy.optimize import nnls   
from statsmodels.stats.multitest import fdrcorrection
import sys
import argparse


# %%
# SET MAIN HYPERPARAMETERS
# TRAIT_LABEL = "Contemplating"  

TRAIT_SETS = {
    "all_13": [
        "Open-minded","feeling Affectionate","Attentive","Assertive",
        "feeling Gloomy","feeling Peaceful","Agreeable","Judging",
        "feeling Angry","feeling Bewildered","Impulsive",
        "Self-disciplined","Contemplating"
    ],
    "mental_8": [
        "feeling Affectionate","feeling Gloomy","feeling Peaceful",
        "feeling Angry","feeling Bewildered","Judging",
        "Contemplating","Attentive"
    ],
    "personality_5": [
        "Open-minded","Agreeable","Assertive",
        "Self-disciplined","Impulsive"
    ],
    "trait_9": [
        "Open-minded","feeling Affectionate","Attentive","Assertive",
        "Agreeable","Judging","feeling Angry","Self-disciplined","Contemplating"
    ]
}

# Select model here: choose one key from TRAIT_SETS
model_key = "trait_9"   # options: all_13, mental_8, personality_5, trait_9
traits    = TRAIT_SETS[model_key]

# ALL_TRAIT_LABELS = [
    #"Open-minded","feeling Affectionate","Attentive","Assertive",
    #"feeling Gloomy","feeling Peaceful","Agreeable","Judging",
    #"feeling Angry","feeling Bewildered","Impulsive",
    #"Self-disciplined","Contemplating"
#]
ALL_TRAIT_SAVE_STRS = [t.replace(" ","_").replace("-","_")
                       for t in traits]
# Our 13 trait labels 
# ["Open-minded", "feeling Affectionate", "Attentive", "Assertive", "feeling Gloomy", "feeling Peaceful", "Agreeable", "Judging", "feeling Angry", "feeling Bewildered", "Impulsive", "Self-disciplined", "Contemplating"]

#TRAIT_LABEL_SAVE_STRING = TRAIT_LABEL.replace(" ", "_").replace("-", "_")
p = argparse.ArgumentParser()
p.add_argument("--stim", required=True)
args = p.parse_args()
STIMULUS_LABEL_SAVE_STRING = args.stim

# Set smoothing setting to either use smoothed trait RDMs or the un-smoothed RDMs
smoothing_setting = "_no_smoothing"    # set to _no_smoothing or set to "" for smoothed RDMs

# %%
# ──────────────────────────────────────────────────────────────
# 0) PATHS & I/O
# ──────────────────────────────────────────────────────────────
root_dir  = "/burg/psych/users/wf2315/narratives-fmri/fmriprep"          # ←  same as in cleaning script
deriv_dir = os.path.join(root_dir, "derivatives") #   (don’t hard-code “subjects” yet)



# output from your behaviour-model RSA
for trait_long, trait_save in zip(traits, ALL_TRAIT_SAVE_STRS):
    rdm_path = os.path.join(
        deriv_dir, "RDMs_behavior",
        f"{STIMULUS_LABEL_SAVE_STRING}_{trait_save}_RDM{smoothing_setting}.npy"
    )
    model_rdm = np.load(rdm_path)

# %%
# ──────────────────────────────────────────────────────────────
# 1) SUBJECT / RUN FILTERS    ─────────────
# ──────────────────────────────────────────────────────────────
# Exclusions
EXCLUSIONS = {
    "pieman": {
        "subs": {'sub-001','sub-021','sub-022','sub-038','sub-056','sub-068','sub-069'},
        "sub_runs": {
            ('sub-002','2'),('sub-003','2'),('sub-004','2'),
            ('sub-005','2'),('sub-006','2'),('sub-008','2'),
            ('sub-010','2'),('sub-011','2'),('sub-012','2'),
            ('sub-013','2'),('sub-014','2'),('sub-015','2'),
            ('sub-016','2')
        },
    },

    "reachforstars": {"subs": {"sub-139"}},                 
    "slumlordreach": {"subs": {"sub-139"}},    
    "black": {},        # no exclusions
    "forgot": {},       # no exclusions
    "notthefallintact": {"subs": {"sub-317","sub-335"}},
}

# Fail fast if stim not configured at all
assert STIMULUS_LABEL_SAVE_STRING in EXCLUSIONS, (
    f"Stimulus '{STIMULUS_LABEL_SAVE_STRING}' is not configured in EXCLUSIONS."
)

_ex = EXCLUSIONS[STIMULUS_LABEL_SAVE_STRING]
exclude_subs     = set(_ex.get("subs", set()))
exclude_sub_runs = set(_ex.get("sub_runs", set()))

target_subject = None     # e.g. "sub-002" to run a single person


# %%
# ─── find a subject that has a cleaned BOLD file ─────────────────────────
cleaned_root = os.path.join(
    deriv_dir, f"{STIMULUS_LABEL_SAVE_STRING}_cleaned"
)

# build a sorted list of candidate subjects, excluding any in exclude_subs
candidates = sorted(
    s for s in os.listdir(cleaned_root)
    if s.startswith("sub-") and s not in exclude_subs
)

bold_img_path = None
bold_sub      = None

for sub in candidates:
    func_dir = os.path.join(cleaned_root, sub, "func")

    # (a) single‐run file
    single_pattern = os.path.join(
        func_dir,
        f"{sub}_task-{STIMULUS_LABEL_SAVE_STRING}_cleaned_desc-masked_bold.nii.gz"
    )
    # (b) multi-run files (_run-01_, _run-02_, …)
    multi_pattern  = os.path.join(
        func_dir,
        f"{sub}_task-{STIMULUS_LABEL_SAVE_STRING}_run-*_cleaned_desc-masked_bold.nii.gz"
    )

    hits = glob.glob(single_pattern) or glob.glob(multi_pattern)
    if hits:                         # found at least one file → stop searching
        bold_img_path = hits[0]      # take the first match
        bold_sub      = sub
        break

if bold_img_path is None:
    raise RuntimeError(
        f"No cleaned BOLD files found for stimulus '{STIMULUS_LABEL_SAVE_STRING}'."
    )

# ─── load the image and report dimensions ───────────────────────────────
bold_img = nib.load(bold_img_path)
zooms    = bold_img.header.get_zooms()

print(f"Using subject: {bold_sub}")
print(f"Voxel size (mm): {zooms[:3]}")
print(f"TR (s): {zooms[3]}")
print(f"Shape: {bold_img.shape}")

# %%
# ----------------------------------------------------------------
# 2)  BUILD SUBJECT LIST  (from cleaned derivatives)  ------------
# ----------------------------------------------------------------
cleaned_root = os.path.join(deriv_dir, f"{STIMULUS_LABEL_SAVE_STRING}_cleaned")
all_subs     = sorted(
    d for d in os.listdir(cleaned_root) if d.startswith("sub-")
)
if target_subject:
    if target_subject not in all_subs:
        raise ValueError(f"{target_subject} not found in {cleaned_root}")
    subjects = [target_subject]
else:
    subjects = [s for s in all_subs if s not in exclude_subs]

print("Subjects to process →", ", ".join(subjects))

# %%
# ──────────────────────────────────────────────────────────────
# 3) FETCH SCHAEFER ATLAS  ─────────────────────────────────────
# ──────────────────────────────────────────────────────────────

# Schaefer parcel/atlas parameters
n_rois = 200
yeo_networks = 17
resolution_mm = 2                   # resolution of your Schaefer atlas (double check!)

schaefer    = datasets.fetch_atlas_schaefer_2018(
                 n_rois=n_rois,
                 yeo_networks=yeo_networks,
                 resolution_mm=resolution_mm
             )
atlas_img   = nib.load(schaefer['maps'])  # default 2mm MNI - but our images 3x3x4 (Pieman and others) OR 2.5^3 (ie., Black and Forgot)

atlas_resampled = resample_to_img(atlas_img, bold_img, interpolation='nearest')
atlas_data     = atlas_resampled.get_fdata()



# Change Schaeffer Labels so 0 is whole brain and 1 corresponds to 1st ROI
labels = schaefer['labels']
# change to string and remove excess
labels = [l.replace(b'17Networks_', b'').decode('utf-8') for l in labels]
# Prepend background label
labels = np.insert(labels, 0, "Background")

# %% [markdown]
# # Multiple Regression

# %%
def compute_r2(X, y):
    """
    Compute the coefficient of determination (R²) for a linear regression model.
    Parameters
    ----------"""
    betas, _ = nnls(X, y)
    y_pred = X @ betas
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def permutation_test(X, y, n_permutations=10):

    # compute observed R2
    r2_observed = compute_r2(X, y)
    
    # Permutation test for R² - create null distribution from N permutations
    r2_values = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)
        r2_values[i] = compute_r2(X, y_permuted)
    

    # retrurn p-value and pseudo-t value
    pseudo_t = (r2_observed - np.mean(r2_values)) / np.std(r2_values, ddof=1)
    p_value = np.mean(r2_values >= r2_observed)
    return pseudo_t, p_value    


# %%


# 1) Load all the single-trait .npy RDMs 
model_rdms = {}
for trait, sstr in zip(traits, ALL_TRAIT_SAVE_STRS):
    filepath = os.path.join(
        deriv_dir, "RDMs_behavior",
        f"{STIMULUS_LABEL_SAVE_STRING}_{sstr}_RDM{smoothing_setting}.npy"
    )
    model_rdms[trait] = np.load(filepath)

# 2) multi-regression RSA 
def rsa_multi_nnls(neural_rdm, model_rdms, traits):
    idx = np.tril_indices(neural_rdm.shape[0], k=-1)
    y   = neural_rdm[idx]
    
    # design matrix of all behavioural RDMs
    Xs  = [model_rdms[t][idx] for t in traits]
    X   = np.column_stack(Xs)              # shape (N, p)
    X   = np.column_stack([np.ones_like(y), X])   # prepend intercept

    # NNLS
    coef, rnorm = nnls(X, y)                   
    #coef, _ = nnls(X, y)                   # coef[0] = intercept
    betas = dict(zip(traits, coef[1:]))

    


    # pseudo-r2 for full model vs null (y̅)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    # compute pseudo F-statistic
    p = len(traits) 
    N = len(y)
    ss_reg = ss_tot - ss_res
    f_stat = (ss_reg / p) / (ss_res / (N - p - 1))

    
    # permutation test
    pseudo_t, permutation_p_value = permutation_test(X, y)

    return betas, r2, pseudo_t, permutation_p_value, float(f_stat)

# 3) run multi-regression for each subject 
def run_multi_for_subject(sub):
    func_dir   = os.path.join(cleaned_root, sub, "func")
    run_pat    = os.path.join(
        func_dir,
        f"{sub}_task-{STIMULUS_LABEL_SAVE_STRING}_run-*_*cleaned_desc-masked_bold.nii.gz"
    )
    single_pat = os.path.join(
        func_dir,
        f"{sub}_task-{STIMULUS_LABEL_SAVE_STRING}_cleaned_desc-masked_bold.nii.gz"
    )
    bold_files = sorted(glob.glob(run_pat)) + sorted(glob.glob(single_pat))
    if not bold_files:
        return

    rows = []
    for bf in bold_files:
        m   = re.search(r"_run-(\d+)_", os.path.basename(bf))
        run = m.group(1) if m else "NA"
        if (sub, run) in exclude_sub_runs: continue

        bold_img  = nib.load(bf)
        bold_data = bold_img.get_fdata()
        atlas_res = resample_to_img(atlas_img, bold_img, interpolation="nearest")
        atlas_dat = atlas_res.get_fdata()

        for parcel_id in range(1, n_rois+1):   
            mask = atlas_dat == parcel_id
            if not mask.any(): continue
            neural_rdm = 1 - np.corrcoef(bold_data[mask,:].T).astype(np.float32)
            betas, r2, pseudo_t, permutation_p_value, f_stat = rsa_multi_nnls(neural_rdm, model_rdms, traits)
            parcel_label = labels[parcel_id]     # ← NEW column
            # build row: sub, run, parcel, β₁…β₁₃, F_stat
            row = [sub, run, parcel_id, parcel_label] + [betas[t] for t in traits] + [r2] + [pseudo_t] + [permutation_p_value]+[float(f_stat)]
            rows.append(row)

    # write out into a new multi_regression folder
    out_base = os.path.join(
        deriv_dir, "RSA_stats",
        STIMULUS_LABEL_SAVE_STRING, "multi_regression", "subject_results"
    )
    os.makedirs(out_base, exist_ok=True)

    header = ["subject","run","parcel_num", "parcel_label"] + ALL_TRAIT_SAVE_STRS + ["r2"] + ["pseudo_t"] + ["permutation_p_value"]+["f_stat"]
    out_csv = os.path.join(
        out_base,
        f"{sub}_{STIMULUS_LABEL_SAVE_STRING}_multi_parcel_RSA_NNLS_{model_key}{smoothing_setting}.csv"
    )
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    



# %%
# 4) callfor each subject
for sub in subjects:
    run_multi_for_subject(sub)
    # break             # comment out to run all subjects

print("ALL MULTI-REG DONE ")



