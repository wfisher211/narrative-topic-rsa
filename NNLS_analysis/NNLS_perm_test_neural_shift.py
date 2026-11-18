# %%
import numpy as np
import random
from scipy.optimize import nnls
import os, glob, re, csv
import glob
import nibabel as nib
import statsmodels.api as sm
from nilearn import datasets
from nilearn.image import resample_to_img
import pandas as pd
from scipy.stats import ttest_1samp
from nilearn import plotting, datasets
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import json

# %%
# Set random seeds for reproducibility
np.random.seed(42)

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
model_key = "personality_5"   # options: all_13, mental_8, personality_5, trait_9
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
STIMULUS_LABEL_SAVE_STRING = "notthefallintact"          # e.g., ["slumlordreach", "pieman", "black", "forgot", "reachforstars", "notthefallintact"]

# Set window shift amount
shift_window = 20  #shifts the window by x TRs 

# Set smoothing setting to either use smoothed trait RDMs or the un-smoothed RDMs
smoothing_setting = "_no_smoothing"    # set to _no_smoothing or set to "" for smoothed RDMs

# Parcel of interest for power analysis
PARCEL_OF_INTEREST = 68          # LH_TempPar_1


# %%
# Expected # subject-run CSVs that should exist after NNLS per-parcel analysis
EXPECTED_CSV_COUNTS = {
    "slumlordreach":   17,
    "pieman":          75,
    "black":           46,
    "forgot":          46,
    "reachforstars":   17,
    "notthefallintact":54,
}

# %%
# ──────────────────────────────────────────────────────────────
# 0) PATHS & I/O
# ──────────────────────────────────────────────────────────────
root_dir  = "/burg/psych/users/gjf2118/narratives-fmri/fmriprep"          # ←  same as in cleaning script
deriv_dir = os.path.join(root_dir, "derivatives") #   (don’t hard-code “subjects” yet)



# output from your behaviour-model RSA
for trait_long, trait_save in zip(traits, ALL_TRAIT_SAVE_STRS):
    rdm_path = os.path.join(
        deriv_dir, "RDMs_behavior",
        f"{STIMULUS_LABEL_SAVE_STRING}_{trait_save}_RDM{smoothing_setting}.npy"     
    )
    model_rdm = np.load(rdm_path)


perm_dir = os.path.join(
    deriv_dir, "RSA_stats", STIMULUS_LABEL_SAVE_STRING, "perm_test_neural_shift"
)


#Output path for parcel results for later power analysis
SAVE_PATH = os.path.join(
    perm_dir,
    f"{STIMULUS_LABEL_SAVE_STRING}_parcel{PARCEL_OF_INTEREST}_{model_key}_subject_s1_perms.npy"
)

# %%
# ──────────────────────────────────────────────────────────────
# 1) SUBJECT / RUN FILTERS  (copy-paste verbatim)  ─────────────
# ──────────────────────────────────────────────────────────────
exclude_subs = {
    "sub-317","sub-335"
}
exclude_sub_runs = []
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
# FETCH SCHAEFER ATLAS  ─────────────────────────────────────
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

# %%
# ─── PERMUTATION TESTING Pipeline (s₁…sₙ) ────────────────────────────────
from scipy.optimize import nnls
from tqdm import tqdm

def vectorize_rdm(rdm):
    idx = np.tril_indices(rdm.shape[0], k=-1)
    return rdm[idx]

# 9.1) load all subject‐run CSVs with betas 
multi_csvs = glob.glob(os.path.join(
    deriv_dir, "RSA_stats", STIMULUS_LABEL_SAVE_STRING,
    "multi_regression","subject_results",
    f"*_{STIMULUS_LABEL_SAVE_STRING}_multi_parcel_RSA_NNLS_{model_key}{smoothing_setting}.csv"
))

# assert we have right number of files per stimulus 
n_expected = EXPECTED_CSV_COUNTS.get(STIMULUS_LABEL_SAVE_STRING)

if n_expected is not None:                       # we listed this stim above
    assert len(multi_csvs) == n_expected, (
        f"[{STIMULUS_LABEL_SAVE_STRING}] expected {n_expected} files "
        f"but found {len(multi_csvs)}"
    )
else:                                            # stim not in table → just sanity-check
    if len(multi_csvs) == 0:
        raise ValueError(f"No CSVs found for {STIMULUS_LABEL_SAVE_STRING}")



df_list = []
for fn in multi_csvs:
    df = pd.read_csv(fn)
    df['unit'] = df['subject'] + "_" + df['run'].astype(str)
    df_list.append(df)
all_df = pd.concat(df_list, ignore_index=True)

trait_cols = ALL_TRAIT_SAVE_STRS
n_traits  = len(trait_cols)

# 9.2) compute observed s_k per parcel for k=1…n_traits
observed = {}
# sort-and-sum top-k per row, then average by parcel
for k in range(1, n_traits+1):
    # descending sort, sum top k
    all_df[f's{k}'] = (
        np.sort(all_df[trait_cols].values, axis=1) # 1) sort each row ascending
        [:, ::-1]   # 2) reverse to descending 
        [:, :k]      # 3) take top-k
        .sum(axis=1)  # 4) sum across rows
    )
    observed[k] = all_df.groupby('parcel_num')[f's{k}'].mean()

# 9.3) precompute & cache every neural RDM (unchanged)
neural_rdm_cache = {}
for sub in subjects:
    func_dir = os.path.join(cleaned_root, sub, "func")
    run_pat  = os.path.join(func_dir,
        f"{sub}_task-{STIMULUS_LABEL_SAVE_STRING}_run-*_*cleaned_desc-masked_bold.nii.gz"
    )
    single_pat = os.path.join(func_dir,
        f"{sub}_task-{STIMULUS_LABEL_SAVE_STRING}_cleaned_desc-masked_bold.nii.gz"
    )
    bold_files = sorted(glob.glob(run_pat)) + sorted(glob.glob(single_pat))
    for bf in bold_files:
        m = re.search(r"_run-(\d+)_", os.path.basename(bf))
        run = m.group(1) if m else "NA"
        if (sub, run) in exclude_sub_runs: 
            continue
        # print("Attempting to load:", bf)  # debug line for when files get corrupted
        img = nib.load(bf)
        atlas_res = resample_to_img(atlas_img, img, interpolation='nearest')
        atlas_dat = atlas_res.get_fdata().astype(int)
        bold_dat  = img.get_fdata()
        for pid in range(1, n_rois+1):
            mask = atlas_dat == pid
            if not mask.any(): 
                continue
            # 1 - corr → RDM
            corr = np.corrcoef(bold_dat[mask,:].T)
            neural_rdm_cache[(sub, run, pid)] = (1 - corr).astype(np.float32)

    

# 9.4) load behavior RDMs (unchanged)
behavior_rdms = {
    trait: np.load(os.path.join(
        deriv_dir, "RDMs_behavior",
        f"{STIMULUS_LABEL_SAVE_STRING}_{sstr}_RDM{smoothing_setting}.npy"
    ))
    for trait, sstr in zip(traits, ALL_TRAIT_SAVE_STRS)
}

# 9.5) build null distributions for sₖ
n_perm = 100
parcel_ids = list(observed[1].index)
nulls = {k: {pid: [] for pid in parcel_ids} for k in range(1, n_traits+1)}

# 
from collections import defaultdict
subj_perm_vals = defaultdict(list)   # {(sub, run): [s1_perm0, s1_perm1, ...]}

n_tr   = next(iter(behavior_rdms.values())).shape[0]

assert n_tr == 160, f"Expected 160 timepoints, got {n_tr}"

for i in tqdm(range(n_perm), desc="Permutations"):
    
    
    # accumulate sums & counts for each k & parcel
    sums   = {k: {pid: 0.0 for pid in parcel_ids} for k in range(1, n_traits+1)}
    counts = {k: {pid: 0   for pid in parcel_ids} for k in range(1, n_traits+1)}

    for (sub, run, pid), rdm in neural_rdm_cache.items():
        # random circular TR shift ≥20
        shift = np.random.randint(shift_window, n_tr-shift_window)
        # apply circular shift to neural RDM
        rolled_rdm = np.roll(np.roll(rdm, shift, axis=0), shift, axis=1)
        y = vectorize_rdm(rolled_rdm)
        X = np.column_stack([np.ones_like(y)] +
                             [vectorize_rdm(behavior_rdms[t]) for t in traits])
        coef, _    = nnls(X, y)
        betas      = coef[1:]                     # drop intercept
        top_sorted = np.sort(betas)[::-1]         # descending

        # for each k, sum top-k betas
        for k in range(1, n_traits+1):
            s_k = top_sorted[:k].sum()
            sums[k][pid]   += s_k
            counts[k][pid] += 1

            if pid == PARCEL_OF_INTEREST and k == 1:
                # key = (subject, run). We'll pool across runs later → subject average
                subj_perm_vals[(sub, run)].append(s_k)

    # record mean null s_k for each parcel
    for k in range(1, n_traits+1):
        for pid in parcel_ids:
            nulls[k][pid].append(sums[k][pid] / counts[k][pid])


# ------------------------------------------------------------
# Build subject × permutation matrix for specific parcel   
# ------------------------------------------------------------
subjects_unique = sorted({sub for (sub, _run) in subj_perm_vals})
n_s   = len(subjects_unique)
n_p   = n_perm

perm_matrix = np.zeros((n_p, n_s), dtype=np.float32)

for s_idx, sub in enumerate(subjects_unique):

    # gather **all** run lists recorded for this subject
    run_arrays = [
        np.asarray(vals, dtype=np.float32)      # shape (n_perm,)
        for (s, r), vals in subj_perm_vals.items()
        if s == sub
    ]

    if not run_arrays:
        raise ValueError(f"No permutation s₁ values stored for subject {sub}")

    # stack → shape (n_runs_for_sub, n_perm)  then average over runs axis-0
    run_stack = np.vstack(run_arrays)
    perm_matrix[:, s_idx] = run_stack.mean(axis=0)

# ─── ensure the output directory exists ────────────────────────────
out_dir = os.path.dirname(SAVE_PATH)
os.makedirs(out_dir, exist_ok=True)

# save caches
np.save(SAVE_PATH, perm_matrix)
print(f"✅ Saved subject-level permutation s₁ values → {SAVE_PATH}")

# ------------------------------------------------------------
# Build OBSERVED s₁ vector (same parcel, averaged across available runs)
# ------------------------------------------------------------
obs_s1_by_subject = []

for sub in subjects_unique:
    rows = all_df[
        (all_df.subject == sub) &
        (all_df.parcel_num == PARCEL_OF_INTEREST)
    ]
    obs_s1_by_subject.append(rows['s1'].mean())

obs_s1 = np.asarray(obs_s1_by_subject, dtype=np.float32)
np.save(os.path.join(perm_dir,
        f"{STIMULUS_LABEL_SAVE_STRING}_parcel{PARCEL_OF_INTEREST}_{model_key}_observed_s1.npy"), obs_s1)
print(f"✅ Saved observed s₁ values for parcel {PARCEL_OF_INTEREST}")

# 9.6) compare observed vs null → p-values for every (k, parcel)
results = []
for k in range(1, n_traits+1):
    for pid, obs_val in observed[k].items():
        dist = np.array(nulls[k][pid])
        pval = (np.sum(dist >= obs_val) + 1) / (n_perm + 1)
        results.append((k, pid, obs_val, pval, pval < .05))

df_perm = pd.DataFrame(
    results,
    columns=['k', 'parcel_num', 'observed_s', 'p_value', 'significant_p05']
)

# 9.7) add parcel labels and save
parcel_labels = pd.DataFrame({
    'parcel_num': np.arange(1, len(labels)),
    'parcel_label': labels[1:]
})
# ─── define & create output folder ─────────────────────────────────────────
perm_dir = os.path.join(
    deriv_dir, "RSA_stats", STIMULUS_LABEL_SAVE_STRING, "perm_test_neural_shift"
)
os.makedirs(perm_dir, exist_ok=True)


df_perm = df_perm.merge(parcel_labels, on='parcel_num')
out_csv = os.path.join(perm_dir, f"{STIMULUS_LABEL_SAVE_STRING}_perm_test_{model_key}_allks{smoothing_setting}.csv")
df_perm.to_csv(out_csv, index=False)
print(f"✅ Permutation results for k=1…{n_traits} saved → {out_csv}")


# %%
# Load CSV file 
perm_csv_path = f"/Volumes/Passport/fmriprep/derivatives/RSA_stats/{STIMULUS_LABEL_SAVE_STRING}/perm_test_neural_shift/{STIMULUS_LABEL_SAVE_STRING}_perm_test_{model_key}_allks{smoothing_setting}.csv"
df_perm1 = pd.read_csv(perm_csv_path)

# Filter to only the k=1 results
df_perm1['k'] = df_perm1['k'].astype(int)
df_k1 = df_perm1[df_perm1['k'] == 1].copy()
df_k1['parcel_num'] = df_k1['parcel_num'].astype(int)


# 1) Extract the integer parcel map
atlas_data = atlas_resampled.get_fdata().astype(int)

# 2) Create maps
p_map               = np.zeros(atlas_data.shape, dtype=float)               # unthresholded
p_map_thresholded   = np.full(atlas_data.shape, np.nan, dtype=float)        # use NaN for masking
s_map               = np.zeros(atlas_data.shape, dtype=float)               # unthresholded
s_map_thresholded   = np.zeros(atlas_data.shape, dtype=float)               # 0 for non-sig

# 3) Fill in each parcel
for _, row in df_k1.iterrows():
    pid  = int(row['parcel_num'])
    pval = float(row['p_value'])
    sval = float(row['observed_s'])

    p_map[atlas_data == pid] = pval
    s_map[atlas_data == pid] = sval

    if pval < 0.05:
        p_map_thresholded[atlas_data == pid] = pval
        s_map_thresholded[atlas_data == pid] = sval

# 4) Wrap and save NIfTI images
p_img = nib.Nifti1Image(p_map, atlas_resampled.affine, atlas_resampled.header)
nib.save(p_img, os.path.join(perm_dir, f"{STIMULUS_LABEL_SAVE_STRING}_perm_pmap_{model_key}_k1{smoothing_setting}.nii.gz"))
print("✅ unthresholded k=1 p-value map saved")

p_img_thresh = nib.Nifti1Image(p_map_thresholded, atlas_resampled.affine, atlas_resampled.header)
nib.save(p_img_thresh, os.path.join(perm_dir, f"{STIMULUS_LABEL_SAVE_STRING}_perm_pmap_{model_key}_k1_thresh05{smoothing_setting}.nii.gz"))
print("✅ thresholded k=1 p-value map (NaN-masked) saved")

s_img = nib.Nifti1Image(s_map, atlas_resampled.affine, atlas_resampled.header)
nib.save(s_img, os.path.join(perm_dir, f"{STIMULUS_LABEL_SAVE_STRING}perm_smap_{model_key}_k1{smoothing_setting}.nii.gz"))
print("✅ k=1 observed s-value map saved")

s_img_thresh = nib.Nifti1Image(s_map_thresholded, atlas_resampled.affine, atlas_resampled.header)
nib.save(s_img_thresh, os.path.join(perm_dir, f"{STIMULUS_LABEL_SAVE_STRING}_perm_smap_{model_key}_k1_thresh05{smoothing_setting}.nii.gz"))
print("✅ thresholded k=1 s-value map saved")

# 5) Save -p map for MRIcroGL visualization
neg_p_map = -1.0 * p_map

neg_p_img = nib.Nifti1Image(neg_p_map, atlas_resampled.affine, atlas_resampled.header)
nib.save(neg_p_img, os.path.join(perm_dir, f"{STIMULUS_LABEL_SAVE_STRING}_perm_neg_pmap_{model_key}_k1{smoothing_setting}.nii.gz"))
print("✅ negative p-value map (p * -1) saved for visualization")

# Optional: Summary for verification
n_sig_parcels = df_k1[df_k1['p_value'] < 0.05]['parcel_num'].nunique()
print(f"Significant parcels (p < 0.05): {n_sig_parcels}")
print("Nonzero thresholded voxels in s_map:", np.count_nonzero(s_map_thresholded))
print("Non-NaN voxels in thresholded p_map:", np.count_nonzero(~np.isnan(p_map_thresholded)))




