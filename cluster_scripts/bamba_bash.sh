#!/bin/bash
# 
#SBATCH --account=edu             # Replace with your SLURM account name
#SBATCH --job-name=GetModelBamba    # Job name
#SBATCH -c 2                      # Number of CPU cores
#SBATCH --mem=40G          
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --output=bamba_output.txt # Output file
#MAY NEED TO INSTALL PACKAGEs? 
# Optional: Load Python module if needed
# module load python/3.10
echo "Running on node: $(hostname)"

#go to directory: 
cd ~/foundation-model-stack

export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
#install packages: 
pip install --user -e .
pip install --user pynvml rouge_score transformers ibm-fms

# Run the Python script
python3 bamba_benchmarking.py
