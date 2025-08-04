# narrative-topic-rsa
Narratives Dataset Analysis Pipeline
## Narratives Analysis Pipeline

This repository provides a complete pipeline for performing representational similarity analysis (RSA) and non‐negative least squares (NNLS) modeling on the Hasson Narratives dataset derivatives (fMRIprep outputs). You’ll find steps for data download, preprocessing, trait RDM creation, NNLS regression, permutation testing, and visualization.

---

### Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Data Download](#data-download)  
3. [Data Preparation](#data-preparation)  
   - 3.1 [Pulling Preprocessed BOLD & Confounds](#pulling-preprocessed-bold--confounds)  
   - 3.2 [Building Intersect Masks](#building-intersect-masks)  
4. [Preprocessing BOLD Runs](#preprocessing-bold-runs)  
5. [Trait RDM Construction](#trait-rdm-construction)  
6. [NNLS Multivariate Regression](#nnls-multivariate-regression)  
7. [Permutation Testing](#permutation-testing)  
8. [Visualization](#visualization)  
9. [Contact & License](#contact--license)  

---

## Prerequisites

- **Operating System**: macOS or Linux  
- **Python**: ≥ 3.8  
- **Jupyter**: Notebook support  
- **Packages** (install via `pip install -r requirements.txt`):  
  - `datalad`, `numpy`, `pandas`, `nibabel`, `nilearn`, `scipy`, `statsmodels`, `tqdm`  

---

## Data Download

1. **Install with DataLad**  
   
   datalad install https://datasets.datalad.org/labs/hasson/narratives/derivatives/fmriprep

2.	**Fetch only the files you need**
    Adjust the stimulus label (forgot, black, pieman, notthefallintact, slumlordreach, reachforstars):

    datalad get \
    sub-*/func/task-<STIM>_space-MNI152NLin2009cAsym*_{
      desc-preproc_bold.nii.gz,
      desc-preproc_bold.json,
      desc-brain_mask.nii.gz
    } \
    sub-*/func/*task-<STIM>*_desc-confounds_regressors.{tsv,json}

## Data Preparation

 3.1 Pulling Preprocessed BOLD & Confounds
 Run the above datalad get command separately for each stimulus of interest (replace <STIM>).

 3.2 Building Intersect Masks
 Use intersect_mask.ipynb to compute a group‐level intersection mask for each stimulus.
 Set relevant subject exclusions for each stimulus:


pieman:
exclude_subs = {
    "sub-001","sub-021","sub-022","sub-038","sub-056","sub-068","sub-069"
}
exclude_sub_runs = {
    ("sub-002","2"),("sub-003","2"),("sub-004","2"),("sub-005","2"),("sub-006","2"),
    ("sub-008","2"),("sub-010","2"),("sub-011","2"),("sub-012","2"),("sub-013","2"),
    ("sub-014","2"),("sub-015","2"),("sub-016","2")

notthefallintact:
exclude_subs = {"sub-317","sub-335"}

slumlordreach and reachforthestars:
exclude_subs = {"sub-139"}

## Preprocessing BOLD Runs

4. Run preprocessing/all_subjects_clean_BOLD.ipynb for each stimulus, specifying:
	•	tr_slice to match story timing (1.5 s TR bins):

  e.g., for “pieman”:
tr_slice = slice(13, 173)

 notthefallintact:
tr_slice = slice(19, 179)

 forgot & black:
tr_slice = slice(3, 163)

 slumlordreach & reachforstars:
tr_slice = slice(19, 179)  # reachforstars uses (647, 807) if full-run indexing

  

## Trait RDM Construction
5. Trait RDM Construction

For each stimulus, generate trait-based representational dissimilarity matrices (RDMs) using updated_trait_rating_rdm_script.ipynb:
	1.	Set stimulus labels in the notebook:

pieman = 1Pieman 
slumlordreach = 2Slumlord
reachforstars = 3ReachStars
notthefallintact = 4NotTheFall
black = 5Black
forgot = 6Forgot

ex.
STIMULUS_LABEL = "1Pieman"           # maps to pieman
STIMULUS_LABEL_SAVE_STRING = "pieman"



## Multivariate Regression
6. NNLS Multivariate Regression

Load your trait RDMs and run non‐negative least squares (NNLS) models via NNLS_analysis/NNLS_mult_reg.ipynb:
	•	Models to run per stimulus:
	•	all_13
	•	mental_8
	•	personality_5
	•	trait_9
	•	Control smoothing via:
 smoothing_setting = ""         # for smoothed data
 or
 smoothing_setting = "_no_smoothing"

 This generates subject‐level .csv of parcel‐wise β‐estimates.


## Permutation Testing
7. Permutation Testing

   Two options:
	1.	Per‐Stimulus Permutation
Run NNLS_perm_test_neural_shift_parcel_saved.ipynb for each stimulus-model combination.
	2.	Aggregate Across All Stimuli
Run aggregate_NNLS_perm_test.ipynb to combine all subjects and stimuli into a single permutation test.

Each notebook produces:
	•	p-map (.nii.gz) per parcel
	•	s-map (observed summed βs)


## Visualization
8. Visualization
Use plotting_visuals_code.ipynb to display:
	•	p-map overlays
	•	1 – p inverted maps with continuous colorbars

   
