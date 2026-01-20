#!/bin/bash

### Job Configuration ###
[cite_start]#SBATCH --partition=main             # Partition name (use 'main' generally) [cite: 64, 68]
[cite_start]#SBATCH --job-name=biomarker_run     # Job name [cite: 66]
[cite_start]#SBATCH --output=logs_%J.out         # Output log file (%J is the job ID) [cite: 67]
[cite_start]#SBATCH --time=0-10:00:00            # Time limit (Days-Hours:Min:Sec) - Adjust as needed [cite: 65]
[cite_start]#SBATCH --gpus=1                     # Request 1 GPU [cite: 77]
[cite_start]#SBATCH --cpus-per-task=4            # Request 4 CPUs (sufficient for 1 GPU) [cite: 99]
[cite_start]#SBATCH --mem=24G                    # Request RAM (24G is usually standard per GPU) [cite: 97, 237]

### Execution ###

# 1. Print some info to the log (helps with debugging)
echo "Job ID: $SLURM_JOBID"
[cite_start]echo "Running on node: $SLURM_JOB_NODELIST" [cite: 84]
echo "Current directory: $(pwd)"

# 2. Load the system Anaconda module
[cite_start]module load anaconda [cite: 86]

# 3. Activate your environment
# Replace 'my_env_name' with the actual name from your environment.yaml file
[cite_start]source activate my_env_name [cite: 88]

# 4. Run your code
# '-u' forces python to print to the log file immediately instead of buffering
python -u main.py