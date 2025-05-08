#!/bin/bash
#SBATCH --account=edu                   # Replace with your SLURM account name
#SBATCH --job-name=GetModelBamba       # Job name
#SBATCH -c 2                            # Number of CPU cores
#SBATCH --mem=40G                       
#SBATCH --gres=gpu:h100:1               # Request an H100 GPU
#SBATCH --time=03:10:00
#SBATCH --output=bamba_output_acc.txt   # Output file

echo "Running on node: $(hostname)"

# Navigate to your repo directory
cd ~/HPML/foundation-model-stack

# Install dependencies
pip install --user -e .
pip install --user pynvml rouge_score
pip install --user ibm-fms

# Run your script (no arguments)
python3 bamba_acc_512.py

scp ~/Desktop/HPML_Cluster/qa_for_accuracy512.json sbk2176@insomnia.rcs.columbia.edu:/insomnia001/home/sbk2176/HPML/foundation-model-stack/

scp sbk2176@insomnia.rcs.columbia.edu:/insomnia001/home/sbk2176/HPML/foundation-model-stack/default_predictions_acc.tsv ~/Desktop/
scp sbk2176@insomnia.rcs.columbia.edu:/insomnia001/home/sbk2176/HPML/foundation-model-stack/default_metrics_acc.csv  ~/Desktop/
