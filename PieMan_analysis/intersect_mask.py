import os
import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import sys

import nibabel as nib

from nilearn.image import load_img, resample_to_img, index_img
from nilearn.masking import intersect_masks, apply_mask, unmask
from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds




# -----------------------------------------------------------------------------
# 1) USER PARAMETERS
# -----------------------------------------------------------------------------

# Where your fmriprep lives
base_dir    = '/burg/psych/users/gjf2118/narratives-fmri/fmriprep'
task_label  = 'pieman'
output_mask = os.path.join(base_dir,
                           f'group_task-{task_label}_intersect_mask.nii.gz')

# TR indices to keep
tr_slice = slice(13, 173)
TR = 1.5

# Exclusions
exclude_subs     = {'sub-001','sub-021','sub-022','sub-038','sub-056','sub-068','sub-069'}
exclude_sub_runs = {('sub-002','2'),('sub-003','2'),('sub-004','2'),
                    ('sub-005','2'),('sub-006','2'),('sub-008','2'),
                    ('sub-010','2'),('sub-011','2'),('sub-012','2'),
                    ('sub-013','2'),('sub-014','2'),('sub-015','2'),
                    ('sub-016','2')}

# Which BOLD space / resolution to use
preferred_space = 'MNI152NLin2009cAsym'
preferred_res   = 'native'

# *** New: set this if you only want to process one subject ***
# e.g. target_subject = 'sub-002' 
# or target_subject = None  to run on all (minus excluded)

# -----------------------------------------------------------------------------
# 2) BUILD LIST OF SUBJECTS
# -----------------------------------------------------------------------------
all_subs = sorted(d for d in os.listdir(base_dir)
                  if d.startswith('sub-')
                  and os.path.isdir(os.path.join(base_dir, d)))


subjects = [s for s in all_subs if s not in exclude_subs]


# -----------------------------------------------------------------------------
# 3) MAKE GROUP INTERSECT MASK
# -----------------------------------------------------------------------------
# where to save the mask
deriv_mask_dir = os.path.join(base_dir, 'derivatives', 'pieMan_masks')
os.makedirs(deriv_mask_dir, exist_ok=True)
# redefine output_mask to live under derivatives/
output_mask = os.path.join(
    deriv_mask_dir,
    f'group_task-{task_label}_intersect_mask.nii.gz'
)

mask_files = []
space_res  = f"space-{preferred_space}_res-{preferred_res}"
for sub in subjects:
    func_dir = os.path.join(base_dir, sub, 'func')
    # try run-level then single-run masks
    patterns = [
        os.path.join(func_dir,
                     f"{sub}_task-{task_label}_run-1_{space_res}_desc-brain_mask.nii.gz"),
        os.path.join(func_dir,
                     f"{sub}_task-{task_label}_{space_res}_desc-brain_mask.nii.gz"),
    ]
    files = sorted(glob.glob(patterns[0])) or sorted(glob.glob(patterns[1]))
    # make sure files are found
    
    # if not files:
    #     print(f"Skipping: {sub} no mask found")
    
    # get file we want (ie., select run-1, not run-2)
    for fp in files:
        m   = re.search(r"_run-(\d+)_", fp)
        run = m.group(1) if m else None
        if run and (sub,run) in exclude_sub_runs:
            print(f"  • skipping {sub} run-{run}")
            continue
        mask_files.append(fp)

if not mask_files:
    raise RuntimeError("No masks found after exclusions!")

print(f"Found {len(mask_files)} masks, intersecting…")
intersect = intersect_masks(mask_files,
                           threshold=1.0,    
                           connected=False)
intersect.to_filename(output_mask)

mask_img = load_img(output_mask)

print("Saved group intersect mask to:", output_mask)


# for future stims (not Pieman) - 
# add a dictaionry at the start to get num subs
assert len(mask_files) == 75, f"Expected 75 masks, found {len(mask_files)}"




