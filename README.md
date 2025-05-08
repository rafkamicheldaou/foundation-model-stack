# Divide, Tritron, Conquer (Indepdently) Exploring Chunked Sates in Bamba
Team Members: Rafka Daou, Maria Garmonina, Sarah Korb

The goal of this project is to explore how to reduce memory costs and improve throughput in the current Bamba implementation by modifying the model’s chunking mechanism for processing input sequences.

We evaluated Bamba’s Structured State-Space Model (SSM) performance by varying:

- Chunking strategies (default vs. optimized)
- Inference settings (use_cache=True vs. False)
- Model depth (reduced layers for profiling vs. full depth for accuracy) 

This GitHub repository includes a Colab notebook titled HPML_Final, which contains:

Initialization of the Bamba model architecture
Benchmarking code for both performance and accuracy The notebook is designed to run sequentially — each cell is arranged in the order of execution. You can open and run it in Google Colab with no additional setup.
It is important to choose the right model configuration depending on your evaluation goal.

## Outline for Running Bamba-9B Benchmarks: 
| Benchmark Type | # Layers | Use_Cache |
|--------------| ---------- | ------------------ |
| Accuracy        | 32 | True |
|Performance  | 4 | False |

At the end of the notebook, we include benchmarking code that logs and visualizes key performance metrics, allowing for direct comparison across different configurations.

We also created a separate notebook titled gpt_score to evaluate the accuracy of our model outputs. This notebook uses GPTScore, a metric that computes the negative log-likelihood of a generated output given a reference — effectively measuring how fluent, coherent, and relevant the model's responses are. The notebook loads our model outputs and references, computes the GPTScore, and then visualizes the results through plots such as average and harmonic mean scores across different chunking strategies.

#### Approach
Our approach for inference optimization is to use PyTorch compile, accelerated transformers, and tensor parallelism. PyTorch compile compiles the code into optimized kernels, accelerated transformers leverages `scaled_dot_product_attention` (SDPA) for accelerating attention computation while saving memory, and tensor parallelism is necessary for larger models.

To enable the Llama models to compile, we had to reimplement `RoPE` encodings without complex numbers. With this change, Llama model inference is able to leverage model compilation for latency reduction.

#### Inference latency
We measured inference latencies with 1024 token prompt and generation of 256 tokens on AWS P4de instance nodes with 8 80G A100 GPUs and report the median latency in the below table.
| Model | # GPUs | Median latency (ms) |
| ----- | ----------- | ----- |
| 7B | 1 | 14ms |
| 13B | 1 | 22ms |
| 70B | 8 | 30ms |

If you would like to reproduce the latencies, you can run the `scripts/benchmark_inference.py` and the details are described in [inference](./scripts).

For more information on reproducing the benchmarks and running some examples, see [here](scripts/README.md)

## HF Model Support

The support for HF models is provided by our HF model adapter. One can obtain similar latencies as tabulated above with HF models using our HF model adapter:

```python
from fms.models import get_model
from fms.models.hf import to_hf_api
import torch
from transformers import pipeline
# fms model
llama = get_model("llama", "13b")

# huggingface model backed by fms internals
llama_hf = to_hf_api(llama)

# compile the model -- in HF, the decoder only
llama_hf.decoder = torch.compile(llama_hf.decoder)

# generate some text -- the first time will be slow since the model needs to be compiled, but subsequent generations should be faster.
llama_generator = pipeline(task="text-generation", model=llama_hf, tokenizer=tokenizer)
llama_generator("""q: how are you? a: I am good. How about you? q: What is the weather like today? a:""")
```

A detailed example is provided [here](./notebooks/hf_adapted_inference.ipynb).

## Tuning

To fine-tune LLaMA, use the `scripts/train_causal.py` training script. Here's
an example of that command.
```
torchrun --nproc_per_node=2 \
        scripts/train_causal.py \
        --architecture=llama \
        --variant=7b \
        --tokenizer=~/models/tokenizer.model \
        --model_path=~/models/7B/ \
        --report_steps=10 \
        --checkpoint_format=meta \
        --distributed=fsdp
```
See options in the script for other ways to train and tune.

## Structure and contents of this Repository

* `fms/models/` - Pure pytorch implementations of popular model architectures, without requiring any specific common interface beyond `nn.Module`. Each model configuration is registered with `fms.models.register_model()` so that instances can be obtained through `fms.models.get_model('architecture', 'variant', '/path/to/data')`. Each model can also register sources/formats/versions of data to load (e.g. checkpoints provided by meta, HF, or trained from this repo). Users of the repo (e.g. `fms-extras`) can register their own model architectures as well.
* `fms/models/hf/` - Adapters that compose our native PyTorch FMS model architecture implementations in HF-compatible wrapper interfaces. Each FMS model implements an adapter, and adapted instances are obtained via `fms.models.hf.to_hf_api(model)`
* `fms/datasets/` - Code for loading data for pre-training and fine-tuning. Individual datasets are retrieved by `fms.datasets.get_dataset('name', tokenizer, 'optional path or other data reference')`. The expected tokenizer conforms to an `fms.utils.tokenizers.BaseTokenizer` interface.
* `fms/modules/` - Components extending `nn.Module` used in our model architecture implementations. Each Module has a corresponding `TPModule` so that modules can be sharded using a tensor-parallel distribution strategy. FMS modules should all support `torch.compile` without graph breaks.
* `fms/training/` - Pre-training and fine-tuning code.
* `fms/utils/` - Other operators useful in working with LLMs. These include a `generate()` function, `Tensor` subclasses, code for dealing with LLM checkpoints that might be saved/sharded in a variety of formats, tokenization code, and various other useful helper functions.
* `scripts/` - Various scripts for inference, benchmarking, and evaluation, as well as an entry-point for tuning/training.
