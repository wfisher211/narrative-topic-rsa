#!/bin/sh
#SBATCH --account=psych
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=trait_rdm
#SBATCH --mail-type=END
#SBATCH --mail-user=wf2315@columbia.edu
#SBATCH --output=slurm/slurm_%x_%a_%j.out
#SBATCH --array=0-5

module purge #start clean
module load singularity #load modules

# six labels, index via SLURM_ARRAY_TASK_ID
STIMS=(slumlordreach pieman black forgot reachforstars notthefallintact)
STIM=${STIMS[$SLURM_ARRAY_TASK_ID]}

# set paths
CONTAINER=/burg/psych/users/software/fajardgb_pyfmri.sif

# run the script w the singularity container
singularity exec ${CONTAINER} python -u trait_rdm.py --stim "$STIM"

echo '****** DONE *******'