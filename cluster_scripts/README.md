# Columbia Insomnia Cluster Scripts for Bamba Benchmarking

This directory contains pairs of Python scripts and accompanying Bash job scripts used to run experiments with Bamba on Columbia University's Insomnia GPU cluster.

Each Python script defines a benchmarking or inference procedure (either accuracy metrics or speed/latency/memory metrics), while the corresponding Bash script contains the SLURM directives and environment setup needed to run the job on the cluster.

### Contents
- `*.py` — Python scripts for model inference, benchmarking, and metric logging (e.g., latency, memory usage).
- `*.sh` — SLURM-compatible Bash scripts used to submit jobs to the Insomnia cluster queue.


