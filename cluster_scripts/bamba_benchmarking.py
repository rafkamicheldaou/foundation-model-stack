
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
import torch.nn.functional as F
import psutil
import pynvml
import pandas as pd
from torch.profiler import profile, ProfilerActivity
from fms.utils.tokenizers import get_tokenizer
from fms.utils.generation import generate
import torch



# overwrite the SSM variant - 
import fms.modules.ssm
from fms.modules.indep_simple_ssm import SSM as DefaultOptimized # line that imports and overwrites.
fms.modules.ssm.SSM = DefaultOptimized
from fms.models import get_model

# Load Bamba model
model = get_model(
    "hf_configured",
    "ibm-ai-platform/Bamba-9B",
    device_type="cuda",
    data_type=torch.bfloat16,
    nlayers=4,
)
model.config.attn_layer_indices = []


print("Number of layers:", len(model.base_model.layers), flush=True)
print("Config nlayers:", model.config.nlayers, flush=True)
print("Attention layers indices:", model.config.attn_layer_indices, flush=True)

layer = next(
    block for block in model.base_model.layers
    if hasattr(block, "ssm")
)

with open("/insomnia001/home/rd3111/foundation-model-stack/test.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

def ids_for_prompt(prompt, tokenizer, device):
    toks = tokenizer.tokenize(prompt)
    ids  = tokenizer.convert_tokens_to_ids(toks)
    return torch.tensor(ids, dtype=torch.long, device=device)

def decode_ids(ids):
    toks  = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(toks)

device = torch.device("cuda")
tokenizer = get_tokenizer("ibm-ai-platform/Bamba-9B")

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
model.compile()

records = []
MAX_NEW_TOKENS = 100
# TO DO: REPLACE DATA PATH
log_path = "/insomnia001/home/rd3111/foundation-model-stack/chunked_ssm_4_layers_first10.txt" # REPLACE WITH DATA PATH
with open(log_path, "a", encoding="utf-8") as log_file:
    for idx, item in enumerate(qa_pairs[:70], start=1):
        inputs = ids_for_prompt(item["prompt"], tokenizer, device)

        # system stats before
        cpu0 = psutil.cpu_percent(None)
        io0  = psutil.cpu_times_percent(None).iowait
        gpu0 = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        torch.cuda.reset_peak_memory_stats()

        # profile the generate step
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            torch.cuda.synchronize()
            t_start = time.time()

            out_ids, times = generate(
                model,
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=False,
                timing="per-token",
            )

            torch.cuda.synchronize()
            t_end = time.time()

        # system stats after
        cpu1 = psutil.cpu_percent(None)
        io1  = psutil.cpu_times_percent(None).iowait
        gpu1 = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        # derive metrics
        t_first     = times[0]
        t_mean      = sum(times[1:]) / len(times[1:])
        total_time  = t_end - t_start
        throughput  = MAX_NEW_TOKENS / total_time
        mem_bw      = peak_mem / total_time
        total_flops = sum(evt.flops for evt in prof.key_averages() if hasattr(evt, "flops"))
        top_ops     = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)

        rec = {
            "id": idx,
            "total_latency_s": total_time,
            "first_token_s": t_first,
            "mean_inter_token_s": t_mean,
            "throughput_tok_s": throughput,
            "peak_mem_MB": peak_mem,
            "mem_bw_MBps": mem_bw,
            "cpu_start_%": cpu0, "cpu_end_%": cpu1,
            "gpu_start_%": gpu0, "gpu_end_%": gpu1,
            "io_wait_diff_%": io1 - io0,
            "total_flops": total_flops,
            "profiler_top_ops": top_ops,
            "output": decode_ids(out_ids),
            "num_chunks": item["num_chunks"],
            "token_len": item["token_len"],
            "prompt": item["prompt"]
        }

        log_file.write(f"{rec}\n")
        records.append(rec)

        print(
            f"{idx}/160 | tot={total_time:.3f}s "
            f"| first={t_first:.3f}s | inter={t_mean:.4f}s | thr={throughput:.1f} tok/s",flush=True
        )

# write results to csv - TO DO: Replace data path
df = pd.DataFrame(records)
df.to_csv("/insomnia001/home/rd3111/foundation-model-stack/indep_optimized.csv", index=False)


