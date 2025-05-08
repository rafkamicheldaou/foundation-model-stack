import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch.nn.functional as F
import psutil
import pynvml
import pandas as pd

from fms.utils.tokenizers import get_tokenizer
from fms.utils.generation import generate

import fms.modules.ssm

# This file runs Bamba with inputs of 256 tokens from a dataset and records the models output 
# We vary the ssm module used in Bamba to see if it affects the quality of outputs given the same set of prompts
# All inputs, reference prompts, and model output are written to a tsv file and saved
# This file is later used in GPTscore scripts to score the Bamba model on its output quality

#Here, pick module to use:
from fms.modules.default_optimized import SSM as OptimizedDefaultSSM
# from fms.modules.ssm import SSM as DefaultSSM

# Overwrite the originals in the source module
fms.modules.ssm.SSM = OptimizedDefaultSSM
chunking = "OptimizedDefaultSSM"

# Now load Bamba:
# Using pretrained, and default layers = 32
from fms.models import get_model
import torch
model = get_model(
    "hf_pretrained",
    #"hf_configured",
    "ibm-ai-platform/Bamba-9B-v2",
    device_type="cuda",
    #data_type=torch.float32,
    #nlayers=32,
)
model.config.attn_layer_indices = []

#Checking layers 
print("Number of layers:", len(model.base_model.layers), flush=True)
print("Config nlayers:", model.config.nlayers, flush=True)
print("Attention layers indices:", model.config.attn_layer_indices, flush=True)

layer = next(
    block for block in model.base_model.layers
    if hasattr(block, "ssm")
)

#Checking correct ssm module
print(isinstance(layer.ssm, OptimizedDefaultSSM),
      layer.ssm.__class__.__module__ + "." + layer.ssm.__class__.__name__)

# DATA LOADING:

#Pairs of qu and a of length 256 tokens:
with open("/insomnia001/home/sbk2176/HPML/foundation-model-stack/qa_for_accuracy_256.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

import csv

#Set device to cuda, compile model and tokenizer
device = torch.device("cuda")
tokenizer = get_tokenizer("ibm-ai-platform/Bamba-9B-v2")
model.compile()

# HELPER FUNCTIONS

#Converting prompts into ids for model input 
def ids_for_prompt(prompt: str, tokenizer, device):
    toks = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(toks)
    if tokenizer.bos_token_id != tokenizer.eos_token_id: 
        ids = [tokenizer.bos_token_id] + ids
    return torch.tensor(ids, dtype=torch.long, device=device)

#Decoding output back into words
def decode_ids(ids: torch.Tensor):
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    return tokenizer.convert_tokens_to_string(toks)

# Path to store logs/output
log_path = f"/insomnia001/home/sbk2176/HPML/foundation-model-stack/accuracy_chunkedssm_32layers_{chunking}.tsv"
write_header = not os.path.exists(log_path)

# Open tsv file to begin recording:
with open(log_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow(["id", "prompt", "reference", "prediction"])

    with torch.no_grad():
        # For every pair, convert to ids and record lenghts 
        for idx, item in enumerate(qa_pairs[:15], start=1):
            prompt = item["prompt"]
            inputs = ids_for_prompt(prompt, tokenizer, device)
            prompt_len = inputs.size(0)
            prompt_len = inputs.size(0)
            answer_len = len(tokenizer.tokenize(item["answer"]))
            total_len = prompt_len + answer_len

            print(f"Prompt {idx} | Prompt tokens: {prompt_len} | Answer tokens: {answer_len} | Total: {total_len}",flush=True)
            
            #Generate output, limit length to that of reference answer in datatset 
            out_ids = generate(
                model,
                inputs,
                max_new_tokens=len(tokenizer.tokenize(item["answer"])),
                use_cache=True,
                timing="",
                eos_token_id=tokenizer.eos_token_id,
            )
            
            #decoding output
            new_ids = out_ids[prompt_len:]
            output_text = decode_ids(new_ids)
            reference = item["answer"]

            #writing everything to the tsv file 
            writer.writerow([idx, prompt, reference, output_text])
            print(f"{idx:3d}: {output_text}…")