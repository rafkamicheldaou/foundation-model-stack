# Divide, Tritron, Conquer (Indepdently) Exploring Chunked Sates in Bamba
Team Members: Rafka Daou, Maria Garmonina, Sarah Korb

The goal of this project is to explore how to reduce memory costs and improve throughput in the current Bamba implementation by modifying the model’s chunking mechanism for processing input sequences.

We evaluated Bamba’s Structured State-Space Model (SSM) performance by varying:

- Chunking strategies (default vs. optimized)
- Inference settings (use_cache=True vs. False)
- Model depth (reduced layers for profiling vs. full depth for accuracy) 

## Outline for Running Bamba-9B Benchmarks: 
| Benchmark Type | # Layers | Use_Cache |
|--------------| ---------- | ------------------ |
| Accuracy        | 32 | True |
|Performance  | 4 | False |


#### Approach

#### Inference latency

| Model | # GPUs | Median latency (ms) |
| ----- | ----------- | ----- |
| 7B | 1 | 14ms |
| 13B | 1 | 22ms |
| 70B | 8 | 30ms |


## Structure and contents of this Repository

* `HPML_Final` Colab notebook that contains: Initialization of the Bamba model architecture, and benchmarking code for both performance and accuracy. The notebook is designed to run sequentially — each cell is arranged in the order of execution. You can open and run it in Google Colab with no additional setup. At the end of the notebook, we include benchmarking code that logs and visualizes key performance metrics, allowing for direct comparison across different configurations.

* `GPT_Score` Colab notebook to evaluate the accuracy of our model outputs. This notebook uses GPTScore, a metric that computes the negative log-likelihood of a generated output given a reference — effectively measuring how fluent, coherent, and relevant the model's responses are. The notebook loads our model outputs and references, computes the GPTScore, and then visualizes the results through plots such as average and harmonic mean scores across different chunking strategies.

* `fms/modulues/indep_ssm.py` - 
* `fms/modulues/default_optimized_ssm.py` -

  
