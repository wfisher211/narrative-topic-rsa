#!/bin/bash
#SBATCH --job-name=int_msk
#SBATCH --account=psych                 # ← adjust to your allocation
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=10:00:00                 
#SBATCH --array=0-4                     # six stimuli but only 5 masks (slumlordreach is the same mask for slumlordreach and reachforstars)
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=wf2315@columbia.edu # ← optional

module purge
module load singularity


# six labels, index via SLURM_ARRAY_TASK_ID
STIMS=(slumlordreach pieman black forgot notthefallintact)
STIM=${STIMS[$SLURM_ARRAY_TASK_ID]}

CONTAINER=/burg/psych/users/software/fajardgb_pyfmri.sif

# run inside the container
singularity exec ${CONTAINER} python all_intersect_mask.py --stim "$STIM"

echo '****** DONE *******'