# %% [markdown]
# # Install necessary functions
# 

# %%
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import load_img, resample_to_img, index_img
from nilearn.masking import intersect_masks, apply_mask, unmask
from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds
import glob
import re
import sys
import argparse
import matplotlib.pyplot as plt


# %% [markdown]
# ## Set User Parameters
# -----------------------------
# Set Args
# -----------------------------
ap = argparse.ArgumentParser(description="Clean timeseries for one stimulus.")
ap.add_argument("--stim", required=True, help="Stimulus label (e.g., pieman)")
ap.add_argument("--base_dir", default="/burg/psych/users/gjf2118/narratives-fmri/fmriprep",
                help="Path to fmriprep root")
args = ap.parse_args()

base_dir    = '/burg/psych/users/wf2315/narratives-fmri/fmriprep'
task_label = args.stim                  # matches fMRIPrep file naming (task-*)
stim_label = args.stim                  # used for output dir naming

# Special case: reachforstars uses the slumlordreach intersect mask
INTERSECT_FROM = {"reachforstars": "slumlordreach"}
mask_stim = INTERSECT_FROM.get(stim_label, stim_label)

# TR length
TR = 1.5

# TR windows per stimulus (inclusive of start, exclusive of stop)
TR_SLICES = {
    "slumlordreach":   slice(19, 179),
    "pieman":          slice(13, 173),
    "black":           slice(3, 163),
    "forgot":          slice(3, 163),
    "reachforstars":   slice(647, 807),   # uses slumlordreach mask but has its own outputs
    "notthefallintact":slice(19, 179),
}

# Fail fast if stim not configured
assert (stim_label in TR_SLICES), f"Stimulus '{stim_label}' missing in TR_SLICES"

tr_slice = TR_SLICES[stim_label]

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
assert task_label in EXCLUSIONS, (
    f"Stimulus '{task_label}' is not configured in EXCLUSIONS."
)

_ex = EXCLUSIONS[task_label]
exclude_subs     = set(_ex.get("subs", set()))
exclude_sub_runs = set(_ex.get("sub_runs", set()))

# Which BOLD space / resolution to use
preferred_space = 'MNI152NLin2009cAsym'
preferred_res   = 'native'
# -----------------------------------------------------------------------------
# %%
EXPECTED_COUNT = {
    "slumlordreach":   17,
    "pieman":          75,
    "black":           46,
    "forgot":          46,
    "reachforstars":   17,
    "notthefallintact":54,
}

ALLOWED_SHAPES = {
    (65, 77, 49, 160),
    (78, 93, 78, 160),
}
# -----------------------------------------------------------------------------
mask_path = os.path.join(
    base_dir, "derivatives", f"{mask_stim}_masks",
    f"group_task-{mask_stim}_intersect_mask.nii.gz"
)
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Missing intersect mask: {mask_path}")
intersect = load_img(mask_path)
# %%
# -----------------------------------------------------------------------------
# BUILD LIST OF SUBJECTS
# -----------------------------------------------------------------------------
all_subs = sorted(d for d in os.listdir(base_dir)
                  if d.startswith('sub-')
                  and os.path.isdir(os.path.join(base_dir, d)))

subjects = [s for s in all_subs if s not in exclude_subs]

# %%
# -----------------------------------------------------------------------------
# PROCESS EACH VALID BOLD RUN (native‑res, already aligned)
# -----------------------------------------------------------------------------
space_res = f"space-{preferred_space}_res-{preferred_res}"

cleaned_count = 0

for sub in subjects:
    func_dir = os.path.join(base_dir, sub, "func")
    # collect preproc BOLD files
    run_pat    = os.path.join(func_dir,
                   f"{sub}_task-{task_label}_run-*_{space_res}_desc-preproc_bold.nii.gz")
    single_pat = os.path.join(func_dir,
                   f"{sub}_task-{task_label}_{space_res}_desc-preproc_bold.nii.gz")
    bold_files = sorted(glob.glob(run_pat)) + sorted(glob.glob(single_pat))
    if not bold_files:
        continue

    for bf in bold_files:
        run_m = re.search(r"_run-(\d+)_", os.path.basename(bf))
        run   = run_m.group(1) if run_m else None
        if run and (sub,run) in exclude_sub_runs:
           # print(f"  • skipping {sub} run-{run}")
            continue

        # flexible confounds lookup
        if run:
            conf_pat = os.path.join(func_dir,
                f"{sub}_task-{task_label}_run-{run}_desc-confounds*.tsv")
        else:
            conf_pat = os.path.join(func_dir,
                f"{sub}_task-{task_label}_desc-confounds*.tsv")

        confs = sorted(glob.glob(conf_pat))

        # Assert we have confounds for each file
        assert confs, f"[{os.path.basename(bf)}] Missing confounds (checked {conf_pat})"


        conf_file = confs[0]

        # print(f"\nProcessing {sub}{'_run-'+run if run else ''}: {os.path.basename(bf)}")
        bold_img = load_img(bf)

        # ── SANITY CHECK alignment ─────────────────────────────────────────────
        # Make sure shapes and affines exactly match the intersect mask; otherwise skip.
        assert bold_img.shape[:3] == intersect.shape, \
               f"[{sub}{run}] shape mismatch: bold {bold_img.shape[:3]} vs mask {intersect.shape}"
        assert np.allclose(bold_img.affine, intersect.affine), \
               f"[{sub}{run}] affine mismatch between bold and mask"

        # ─────────────────────────────────────────────────────────────────────────



        # -----------------------------------------------------------------------------
        # LOAD CONFOUNDS via nilearn’s fmriprep interface
        # -----------------------------------------------------------------------------
        # print("Loading confounds…")
        confounds_df, sample_mask = load_confounds(
        img_files    = bf,
        strategy     = ('motion', 'high_pass', 'wm_csf'),
        motion       = 'full',
        wm_csf       = 'basic',
        global_signal= 'basic',
        compcor      = 'anat_combined',
        n_compcor    = 'all',
        ica_aroma    = 'full',
        scrub        = 5,
        fd_threshold = 0.2,
        std_dvars_threshold = 3,
        demean       = True
        )

        # extract TR and mask‐then‐clean
        tr = float(bold_img.header.get_zooms()[3])
        ts = apply_mask(bold_img, intersect)
        assert tr in (1500,1.5), f"TR mismatch: {tr} vs 1500"


        # full‐length clean with every kwarg spelled out
        ts_clean = clean(
        signals         = ts,
        runs            = None,
        detrend         = True,
        standardize     = 'zscore_sample',    # use sample‐std zscore
        sample_mask     = None,
        confounds       = confounds_df,
        standardize_confounds = 'zscore_sample',
        low_pass        = 0.1,
        high_pass       = 0.01,
        t_r             = TR,
        ensure_finite   = True,
        )

        # back to 4D NIfTI
        cleaned_img = unmask(ts_clean, intersect)

        # now select only our TR window via index_img
        sel_img = index_img(cleaned_img, tr_slice)

        # --- ADD THIS LINE to fix TR in the header ---
        sel_img.header.set_zooms(sel_img.header.get_zooms()[:3] + (TR,))

        # sanity check: should be 160 TRs
        assert sel_img.shape in ALLOWED_SHAPES, (
            f"unexpected shape {sel_img.shape}, "
            f"must be one of {ALLOWED_SHAPES}"
        )
        assert sel_img.shape[-1] == 160 , f'unexpected TR length {sel_img.shape} should be 160 TRs long' 

        # save into derivatives/{task_label}_cleaned
        run_tag   = f"_run-{run}" if run else ""
        deriv_dir = os.path.join(base_dir, "derivatives", f"{stim_label}_cleaned")
        out_dir   = os.path.join(deriv_dir, sub, "func")
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir,
                     f"{sub}_task-{stim_label}{run_tag}_cleaned_desc-masked_bold.nii.gz")
        sel_img.to_filename(out_file)
        # print("  → saved to:", out_file)
        
        
        cleaned_count += 1

# print("\nALL DONE.")


# look up the expected count for this stimulus
expected = EXPECTED_COUNT.get(stim_label)
assert expected is not None, f" No expected count defined for stim '{stim_label}'"
# now assert dynamically
assert cleaned_count == expected, (
    f"Expected to clean {expected} files for stim '{stim_label}', "
    f"but cleaned {cleaned_count}."
)
print(f"[ok] Cleaned {cleaned_count} files for '{stim_label}' (mask from '{mask_stim}')")




