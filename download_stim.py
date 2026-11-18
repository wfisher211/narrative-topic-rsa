# download_stim.py
"""Fetch the fMRIPrep files for one Narratives story.

Usage:
    python download_stim.py <path-to-fmriprep-dataset> <stim>
Example:
    python download_stim.py C:/Users/William/fmriprep pieman
"""
import argparse
import os
from glob import glob

import datalad.api as dl

STIMS = ["forgot", "black", "pieman", "notthefallintact", "slumlordreach", "reachforstars"]
SPACE = "space-MNI152NLin2009cAsym"

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dataset", help="path to the installed fmriprep dataset")
parser.add_argument("stim", choices=STIMS)
parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
args = parser.parse_args()

patterns = [
    f"sub-*/func/*task-{args.stim}_*{SPACE}*_desc-preproc_bold.nii.gz",
    f"sub-*/func/*task-{args.stim}_*{SPACE}*_desc-preproc_bold.json",
    f"sub-*/func/*task-{args.stim}_*{SPACE}*_desc-brain_mask.nii.gz",
    f"sub-*/func/*task-{args.stim}_*desc-confounds*.tsv",
    f"sub-*/func/*task-{args.stim}_*desc-confounds*.json",
]

ds = os.path.abspath(args.dataset)
files = sorted({os.path.join(ds, f) for p in patterns for f in glob(p, root_dir=ds)})
print(f"{len(files)} files for task-{args.stim} (expect 5 per run)")

if args.yes or input("Download? [y/N] ").lower() == "y":
    dl.get(path=files, dataset=ds, jobs=4)