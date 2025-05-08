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
![image](https://github.com/user-attachments/assets/beedd1ca-9235-40d0-bc9a-df47f21d1cf8)

#### Experimental Evaluation: Latency/Throughput/ Memory Usage

The triton-optimzied implementation constantly achieves lower latency and higher throughput compared to the baseline. This improvement is due to reduced kernel overhead, coalesced memory access and efficient fusion of operations. The independent chunking variant reduces latency by removing interdependencies. 
![image](https://github.com/user-attachments/assets/61458f48-bb41-4afa-996f-0af9f25af8f0)

The Triton-optimzied implementation reduces peak memory usage compared to the defualt, especially at lower chunks. It also achieves more stable and memory efficient memory bandwith over increasing chunk sizes. Meanwhile, the independent chunking variant slightly increases memory load but there is an evident tradeoff between reuse and parallelism.
![image](https://github.com/user-attachments/assets/53d742bf-a1d6-4d17-8d63-6199b6a2c0c4)

When evaluating accuracy, the Bamba-9B default model produced a wider and lower-scoring distribution, while our default optimized version, achieved higher GPTScores with less variance. This is likely caused by the numerical stability introduced by the Triton kernel when performing low level operations. 

![image](https://github.com/user-attachments/assets/09e4a29e-92e4-4b2b-8f89-d8bd16433d9a)

In addition to evaluating accuracy and various model architecture, we ran a series of experiments focused specifically on compiler-level speedups. Rather than changing the model itself, we used different modes and options within torch.compile. The configuration that delivered the best results combined max-autotune with epilogue_fusion, achieving an average latency of 11.88 seconds and throughput of 8.4 tokens per second. Max-autotunes is designed to search for the most efficient kernel implementation, while epilogue_fusion reduces GPU overhead by fusing post-processing steps into a single kernel. 

![image](https://github.com/user-attachments/assets/463dde6b-bdc6-49a0-acc5-e2b8afb48161)
## Structure and contents of this Repository

* `HPML_Final` Colab notebook that contains: Initialization of the Bamba model architecture, and benchmarking code for both performance and accuracy. The notebook is designed to run sequentially — each cell is arranged in the order of execution. You can open and run it in Google Colab with no additional setup. At the end of the notebook, we include benchmarking code that logs and visualizes key performance metrics, allowing for direct comparison across different configurations.

* `GPT_Score` Colab notebook to evaluate the accuracy of our model outputs. This notebook uses GPTScore, a metric that computes the negative log-likelihood of a generated output given a reference — effectively measuring how fluent, coherent, and relevant the model's responses are. The notebook loads our model outputs and references, computes the GPTScore, and then visualizes the results through plots such as average and harmonic mean scores across different chunking strategies.

* `./cluster_scripts` -  This contains all the code that was run on the insomnnia cluster to benchmark accuracy and performance including our bash scripts.

* `./benchmarking_data` -  This contains a folder of data used to test benchmarking and accuracy. The data in this folder varies by token length 
  
* `fms/modulues/indep_ssm.py` - Module removes inter-chunk recurrence by eliminating state caching and cross chunk dependencies. This enables parallel processing across chunks and reduces latency. 
* `fms/modulues/default_optimized_ssm.py` - Module implements the standard SSM with architectural optimizations to reduce runtime bottlenecks. This version preserves the original autoregressive behavior while significantly improving performance through low-level memory and kernel tuning. 
  
## Wandb Project Board: 
 

  
