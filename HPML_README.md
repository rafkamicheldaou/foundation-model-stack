Project Title: Divide, Tritron, Conquer (Indepdently)
Exploring Chunked Sates in Bamba
Team Members: Rafka Daou, Maria Garmonina, Sarah Korb


The goal of this project is to explore how to reduce memory costs and improve throughput in the current Bamba implementation by modifying
the model’s chunking mechanism for processing input sequences.

We evaluated Bamba’s Structured State-Space Model (SSM) performance by varying:
- Chunking strategies (default vs. optimized)
- Inference settings (use_cache=True vs. False)
- Model depth (reduced layers for profiling vs. full depth for accuracy)



**Outline for Running Our Code: 
**
This GitHub repository includes a Colab notebook titled HPML_Final, which contains:
- Initialization of the Bamba model architecture
- Benchmarking code for both performance and accuracy
The notebook is designed to run sequentially — each cell is arranged in the order of execution. You can open and run it in Google Colab with no additional setup.

It is important to choose the right model configuration depending on your evaluation goal.
If you want to benchmark performance — including latency, throughput, and memory usage — it is necessary to reduce the number of layers to speed up benchmarking
and set use_cache=False to expose the behavior of the SSM block. On the other hand, when benchmarking accuracy, you should use the full model with use_cache=True.
This configuration is essential to reflect actual deployment performance.

At the end of the notebook, we include benchmarking code that logs and visualizes key performance metrics, allowing for direct comparison across different configurations.


We also created a separate notebook titled gpt_score to evaluate the accuracy of our model outputs. This notebook uses GPTScore, a metric that computes the negative log-likelihood
of a generated output given a reference — effectively measuring how fluent, coherent, and relevant the model's responses are.
The notebook loads our model outputs and references, computes the GPTScore, and then visualizes the results through plots such as average and harmonic mean scores across different
chunking strategies.
