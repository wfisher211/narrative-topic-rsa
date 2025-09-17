# %% [markdown]
# # Imports

# %%
# %% ─────────────────────────────────────────────
# 0) Imports & global config
# ------------------------------------------------
import os, re, glob
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from nilearn import datasets, image
from scipy.optimize import nnls
from tqdm.notebook import tqdm


# %%
np.random.seed(42)

# %% [markdown]
# # Parameters

# %%
# ---- stories to pool ---------------------------------------------------
STIMS = [ "slumlordreach", "pieman", "black", "forgot", "reachforstars", "notthefallintact"]                 # ["slumlordreach", "pieman", "black", "forgot", "reachforstars", "notthefallintact"]

# ---- trait models ------------------------------------------------------
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
# select which trait model to use here:
model_key = "all_13"               # options: all_13, mental_8, personality_5, trait_9 
traits    = TRAIT_SETS[model_key]
TRAIT_SAVE = [t.replace(" ","_").replace("-","_") for t in traits]

# ---- misc hyper-params -------------------------------------------------
root_dir  = Path("/burg/psych/users/gjf2118/narratives-fmri/fmriprep")
deriv_dir = root_dir / "derivatives"
shift_window      = 20
smoothing_setting = "_no_smoothing"     # options: " " or "_no_smoothing"
n_perm            = 100
PARCEL_OF_INTEREST = 68

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

# %% ─────────────────────────────────────────────
# 1) Gather all subject-level CSVs across stories
# ------------------------------------------------
df_list = []
for stim in STIMS:
    csvs = glob.glob(str(
        deriv_dir / "RSA_stats" / stim / "multi_regression" / "subject_results" /
        f"*_{stim}_multi_parcel_RSA_NNLS_{model_key}{smoothing_setting}.csv"
    ))
    
    # ── single-line hard checks ────────────────────────────────
    assert stim in EXPECTED_CSV_COUNTS, f"No expected count set for '{stim}'"
    assert len(csvs) == EXPECTED_CSV_COUNTS[stim], \
           f"[{stim}] expected {EXPECTED_CSV_COUNTS[stim]} CSVs, found {len(csvs)}"

    for fn in csvs:
        df = pd.read_csv(fn)
        df["stim"]    = stim
        df["subject"] = df["subject"] + "_" + stim            # unique ID
        df["unit"]    = df["subject"] + "_" + df["run"].astype(str)
        df_list.append(df)

all_df = pd.concat(df_list, ignore_index=True)
print("pooled CSV shape →", all_df.shape)

# %% ─────────────────────────────────────────────
# 2) Cache neural RDMs for ALL stims
# ------------------------------------------------
n_rois = 200
schaefer = datasets.fetch_atlas_schaefer_2018(
              n_rois=n_rois, yeo_networks=17, resolution_mm=2)
atlas_raw = nib.load(schaefer["maps"])
labels    = np.insert(
               [l.replace(b"17Networks_", b"").decode("utf-8")
                for l in schaefer["labels"]],
               0, "Background")

neural_cache = {}
atlas_img    = None      

for stim in STIMS:
    cleaned_root = deriv_dir / f"{stim}_cleaned"
    subs = sorted(
    s for s in os.listdir(cleaned_root) if s.startswith("sub-")
)
    for sub in subs:
        func = cleaned_root / sub / "func"
        bolds = glob.glob(str(func / f"{sub}_task-{stim}_run-*_*cleaned*.nii.gz"))
        bolds += glob.glob(str(func / f"{sub}_task-{stim}_cleaned_desc-masked_bold.nii.gz"))

        for bf in tqdm(bolds, desc=f"{stim}: caching RDMs", leave=False):
            run = re.search(r"_run-(\d+)_", os.path.basename(bf))
            run = run.group(1) if run else "NA"
            sub_id = f"{sub}_{stim}"

            img = nib.load(bf)
        
            # ── resample Schaefer atlas to this BOLD image every time ──
            atlas_res = image.resample_to_img(atlas_raw, img, interpolation="nearest")
            atlas_dat = atlas_res.get_fdata().astype(int)

        # keep one copy of resampled atlas for later NIfTI writing
            if atlas_img is None:
                atlas_img = atlas_res
            
            bold_dat  = img.get_fdata()

            for pid in range(1, n_rois + 1):
                mask = atlas_dat == pid
                if not mask.any():
                    continue
                rdm = 1.0 - np.corrcoef(bold_dat[mask, :].T)
                neural_cache[(sub_id, run, pid)] = rdm.astype(np.float32)

print("cached RDMs →", len(neural_cache))

# %% ─────────────────────────────────────────────
# 3) Load ALL behavior RDMs once
# ------------------------------------------------
behav_rdms = {}
for stim in STIMS:
    for trait, save in zip(traits, TRAIT_SAVE):
        behav_rdms[(stim, trait)] = np.load(
            deriv_dir / "RDMs_behavior" /
            f"{stim}_{save}_RDM{smoothing_setting}.npy"
        )

# get lower-triangular indices for vectorizing RDMs
vec_idx = np.tril_indices(behav_rdms[(STIMS[0], traits[0])].shape[0], k=-1)
def vec(mat): return mat[vec_idx]

# %% ─────────────────────────────────────────────
# 4) Compute observed sₖ per parcel
# ------------------------------------------------
trait_cols = TRAIT_SAVE
for k in range(1, len(trait_cols)+1):
    all_df[f"s{k}"] = (np.sort(all_df[trait_cols].values, 1)[:, ::-1][:, :k].sum(1))
observed = {k : all_df.groupby("parcel_num")[f"s{k}"].mean()
            for k in range(1, len(trait_cols)+1)}

# %% ─────────────────────────────────────────────
# 5) Build null distributions by permutation
# ------------------------------------------------
parcel_ids = list(observed[1].index)
nulls = {k:{pid:[] for pid in parcel_ids}
         for k in range(1, len(trait_cols)+1)}

for p in tqdm(range(n_perm), desc="permutations"):
    sums   = {k:np.zeros(len(parcel_ids)) for k in nulls}
    counts = {k:np.zeros(len(parcel_ids), int) for k in nulls}
    for key in sorted(neural_cache.keys()):          
        sub, run, pid = key
        nrdm = neural_cache[key]
        stim = sub.split("_")[-1]                 # get story label back
        shift = np.random.randint(shift_window, 160-shift_window)
        nrdm  = np.roll(np.roll(nrdm, shift, 0), shift, 1)

        X = np.column_stack([np.ones_like(vec(nrdm))] + [
                vec(behav_rdms[(stim, t)]) for t in traits])
        betas,_ = nnls(X, vec(nrdm))
        top = np.sort(betas[1:])[::-1]

        for k in nulls:
            s = top[:k].sum()
            sums[k][pid-1]   += s
            counts[k][pid-1] += 1
    for k in nulls:
        for i,pid in enumerate(parcel_ids):
            nulls[k][pid].append( sums[k][i] / counts[k][i] )

# %% ─────────────────────────────────────────────
# 6) p-values, save CSV + NIfTI maps (k=1 shown)
# ------------------------------------------------
rows = []
for k in nulls:
    for pid in parcel_ids:
        obs = observed[k][pid]
        dist = np.asarray(nulls[k][pid])
        p = ((dist >= obs).sum() + 1) / (n_perm + 1)
        rows.append((k,pid,obs,p,p<0.05))

df_perm = pd.DataFrame(rows, columns=[
            "k","parcel_num","observed_s","p_value","sig_p05"])
df_perm["parcel_label"] = df_perm["parcel_num"].map(
            {i:l for i,l in enumerate(labels)})

out_dir = deriv_dir / "RSA_stats" / "ALL_STIMS" / "perm_test_neural_shift"
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / f"ALL_STIMS_perm_test_{model_key}_allks{smoothing_setting}.csv"
df_perm.to_csv(csv_path, index=False)
print("CSV →", csv_path)



# %%
# ---- build p/s NIfTI for k = 1 -----------------------------------------
df_k1 = df_perm[df_perm.k == 1]
atlas_dat = atlas_img.get_fdata().astype(int)
shape = atlas_dat.shape

p_map  = np.zeros(shape,float); p_thr = np.full(shape,np.nan,float)
s_map  = np.zeros(shape,float); s_thr = np.zeros(shape,float)

for _,r in df_k1.iterrows():
    pid = int(r.parcel_num)
    p,s = r.p_value, r.observed_s
    p_map[atlas_dat==pid] = p
    s_map[atlas_dat==pid] = s
    if p<0.05:
        p_thr[atlas_dat==pid] = p
        s_thr[atlas_dat==pid] = s

def save(img, name): nib.save(img, out_dir/name)

save(nib.Nifti1Image(p_map,  atlas_img.affine, atlas_img.header),
     f"ALL_STIMS_perm_pmap_{model_key}_k1{smoothing_setting}.nii.gz")
save(nib.Nifti1Image(p_thr,  atlas_img.affine, atlas_img.header),
     f"ALL_STIMS_perm_pmap_{model_key}_k1_thresh05{smoothing_setting}.nii.gz")
save(nib.Nifti1Image(s_map,  atlas_img.affine, atlas_img.header),
     f"ALL_STIMS_perm_smap_{model_key}_k1{smoothing_setting}.nii.gz")
save(nib.Nifti1Image(s_thr,  atlas_img.affine, atlas_img.header),
     f"ALL_STIMS_perm_smap_{model_key}_k1_thresh05{smoothing_setting}.nii.gz")

print("✅ NIfTI maps saved to", out_dir)


